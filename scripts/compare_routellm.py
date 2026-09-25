"""Score RouteLLM's pre-trained routers on a stored mdb-steer benchmark run (no LLM calls).

Every benchmark query already has both models' graded, priced answers in `telemetry`, so a router
only has to say strong or weak per query. We sweep each router's threshold and compare it against
mdb-steer's router, random routing and the hindsight ceiling on the same rows.

    pip install -e ".[routellm]"
    python scripts/compare_routellm.py                     # bert only: local, no API key
    python scripts/compare_routellm.py --routers bert,mf   # mf needs OpenAI or Azure OpenAI embeddings

Why not `import routellm`? Importing its routers module builds an `OpenAI()` client at import time,
which fails without OPENAI_API_KEY even for the local BERT router. So we load the same Hugging Face
checkpoints directly and reproduce RouteLLM's `calculate_strong_win_rate` for each router.

`mf` embeds queries with `text-embedding-3-small` (what it was trained on). Configure either:
    OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_EMBED_DEPLOYMENT [, AZURE_OPENAI_API_VERSION]

Scores are cached in the `routellm_scores` collection, so re-runs are instant.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

from mdb_steer import features as F
from mdb_steer.config import Settings
from mdb_steer.evaluation import replay, summarize
from mdb_steer.pipeline import _rescored_features
from mdb_steer.store import Store

BERT_CHECKPOINT = "routellm/bert_gpt4_augmented"
MF_CHECKPOINT = "routellm/mf_gpt4_augmented"
# MF scores a (strong, weak) pair from its training vocabulary; these are RouteLLM's defaults.
MF_STRONG_ID, MF_WEAK_ID = 24, 36  # gpt-4-1106-preview, mixtral-8x7b-instruct-v0.1


class BertRouter:
    """RouteLLM `bert`: 3-way classifier (strong wins / tie / weak wins); P(strong) = 1 - P(tie or weak)."""

    def __init__(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT)
        self.model = AutoModelForSequenceClassification.from_pretrained(BERT_CHECKPOINT, num_labels=3).eval()

    def score(self, prompt: str) -> float:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits[0], dim=-1)
        return float(1 - probs[-2:].sum())


class MFRouter:
    """RouteLLM `mf`: sigmoid(s(strong, q) - s(weak, q)), s = classifier(normalise(P[m]) * proj(embed(q)))."""

    def __init__(self) -> None:
        import torch
        from huggingface_hub import hf_hub_download

        try:
            from safetensors.torch import load_file

            state = load_file(hf_hub_download(MF_CHECKPOINT, "model.safetensors"))
        except Exception:
            state = torch.load(hf_hub_download(MF_CHECKPOINT, "pytorch_model.bin"), map_location="cpu")
        self.P = state["P.weight"]
        self.proj = state["text_proj.0.weight"]
        self.classifier = state["classifier.0.weight"]
        self.embed, self.embed_model = _embedding_client()

    def score(self, prompt: str) -> float:
        import torch

        vec = self.embed.embeddings.create(input=[prompt], model=self.embed_model).data[0].embedding
        q = self.proj @ torch.tensor(vec, dtype=self.proj.dtype)
        models = torch.nn.functional.normalize(self.P[[MF_STRONG_ID, MF_WEAK_ID]], p=2, dim=1)
        logits = (models * q) @ self.classifier.T
        return float(torch.sigmoid(logits[0] - logits[1]))


def _embedding_client():
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        )
        return client, os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"]
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        return OpenAI(), "text-embedding-3-small"
    raise SystemExit("mf needs OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT/_API_KEY/_EMBED_DEPLOYMENT; use --routers bert to stay keyless")


ROUTERS = {"bert": BertRouter, "mf": MFRouter}


def routellm_scores(store: Store, name: str, rows: list[dict]) -> dict[str, float]:
    """P(strong) per query_id, cached in MongoDB so each query is scored once per router."""
    cache = store.runs.database["routellm_scores"]
    cached = {d["query_id"]: d["score"] for d in cache.find({"router": name})}
    todo = [r for r in rows if r["query_id"] not in cached]
    if todo:
        router = ROUTERS[name]()
        for i, r in enumerate(todo, 1):
            cached[r["query_id"]] = router.score(r["query"])
            cache.update_one({"router": name, "query_id": r["query_id"]}, {"$set": {"score": cached[r["query_id"]]}}, upsert=True)
            print(f"  {name}: scored {i}/{len(todo)}", end="\r")
        print()
    return cached


def token_priced(rows: list[dict], ratio: float) -> list[dict]:
    """Per-token pricing (output 3x input), weak model `ratio` times cheaper; same as blog_stats.py."""
    def cost(attempt: dict, unit: float) -> float:
        return unit * (attempt["prompt_tokens"] + 3 * attempt["completion_tokens"])
    return [
        r | {"strong": r["strong"] | {"cost_usd": cost(r["strong"], 1.0)}, "weak": r["weak"] | {"cost_usd": cost(r["weak"], 1 / ratio)}}
        for r in rows
    ]


def curve(rows: list[dict], settings: Settings, grid: list[float]) -> list[dict]:
    points = []
    for t in grid:
        s = replace(settings, threshold=t)
        summary = summarize(replay(rows, s, t), s)
        points.append({
            "threshold": t,
            "offload": summary["strategies"]["router"]["offload"] * 100,
            "savings": summary["cost_savings_pct"],
            "drop": summary["quality_drop_pct"],
            "lift": summary["lift_over_random"],
            "passed": summary["passed"],
        })
    return points


def auc(rows: list[dict]) -> float:
    """AUC of p_strong for the `weak_fails` label: 0.5 is random, 1.0 ranks every weak failure first."""
    pos = [r["p_strong"] for r in rows if r["weak"]["score"] < 1.0]
    neg = [r["p_strong"] for r in rows if r["weak"]["score"] >= 1.0]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def best(points: list[dict]) -> dict | None:
    passing = [p for p in points if p["passed"] and p["offload"] > 0]
    return max(passing, key=lambda p: p["savings"]) if passing else None


def report(label: str, rows_by_router: dict[str, list[dict]], settings: Settings, grid: list[float], verbose: bool) -> None:
    any_rows = next(iter(rows_by_router.values()))
    st = summarize(replay(any_rows, settings, 0.5), settings)["strategies"]
    base = st["all_strong"]["cost_usd"]
    print(f"\n=== {label} ===")
    print(f"  hindsight ceiling: {(base - st['hindsight']['cost_usd']) / base * 100:5.1f}% saving at "
          f"{st['hindsight']['offload'] * 100:.0f}% offload   all-weak: {(base - st['all_weak']['cost_usd']) / base * 100:5.1f}% saving, "
          f"{(st['all_strong']['quality'] - st['all_weak']['quality']) / st['all_strong']['quality'] * 100:.1f}% quality drop")
    print(f"  best point within the {settings.quality_guardrail_pct:g}% guardrail:")
    for name, rows in rows_by_router.items():
        points = curve(rows, settings, grid)
        b = best(points)
        beats = sum(p["lift"] > 0 for p in points if 0 < p["offload"] < 100)
        mixed = sum(0 < p["offload"] < 100 for p in points)
        if b:
            print(f"    {name:<14} t={b['threshold']:.2f}  offload {b['offload']:5.1f}%  saving {b['savings']:5.1f}%  "
                  f"drop {b['drop']:4.1f}%  lift {b['lift']:+.3f}   beats random at {beats}/{mixed} thresholds")
        else:
            print(f"    {name:<14} no threshold passes the guardrail   beats random at {beats}/{mixed} thresholds")
        if verbose:
            for p in points:
                print(f"      t={p['threshold']:.2f}  offload {p['offload']:5.1f}%  saving {p['savings']:6.1f}%  "
                      f"drop {p['drop']:5.1f}%  lift {p['lift']:+.3f}  {'PASS' if p['passed'] else 'fail'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_id", nargs="?", help="benchmark run (default: latest)")
    ap.add_argument("--routers", default="bert", help="comma-separated RouteLLM routers: bert (keyless), mf (needs embeddings key)")
    ap.add_argument("--ratios", default="10", help="extra per-token strong:weak price ratios to report, e.g. 5,10,20")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the full threshold curve for each router")
    args = ap.parse_args()

    settings = Settings()
    store = Store(settings.mongodb_uri, settings.db_name)
    run_id = args.run_id or store.runs.find_one(sort=[("_id", -1)])["run_id"]
    rows = list(store.telemetry.find({"run_id": run_id}))
    if not rows:
        raise SystemExit(f"no telemetry for run {run_id!r}; run `benchmark` first")
    print(f"run {run_id}: {len(rows)} held-out queries")

    model = store.latest_router_model()
    by_router = {"mdb-steer": [
        r | {"p_strong": F.predict(model, [_rescored_features(r, settings)[f] for f in model["features"]])} for r in rows
    ]}
    for name in [n.strip() for n in args.routers.split(",") if n.strip()]:
        if name not in ROUTERS:
            raise SystemExit(f"unknown router {name!r}; choose from {', '.join(ROUTERS)}")
        scores = routellm_scores(store, name, rows)
        by_router[f"routellm-{name}"] = [r | {"p_strong": scores[r["query_id"]]} for r in rows]

    # RouteLLM scores bunch in a narrow band, so sweep every observed score as well as a fixed grid;
    # that puts all routers on the same offload axis.
    grid = sorted({round(0.05 * i, 2) for i in range(21)} | {round(r["p_strong"], 4) for rs in by_router.values() for r in rs})

    print("AUC for 'weak model fails' (threshold-free; 0.5 = random):")
    for name, rs in by_router.items():
        print(f"  {name:<14} {auc(rs):.3f}")

    report("CPU compute cost (measured)", by_router, settings, grid, args.verbose)
    for ratio in [float(x) for x in args.ratios.split(",") if x.strip()]:
        priced = {n: token_priced(rs, ratio) for n, rs in by_router.items()}
        report(f"per-token pricing, weak {ratio:g}x cheaper", priced, settings, grid, args.verbose)

if __name__ == "__main__":
    main()
