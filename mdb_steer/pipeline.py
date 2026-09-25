"""Calibration and benchmark workflows."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from mdb_steer import features as F
from mdb_steer import judge
from mdb_steer.config import Settings
from mdb_steer.evaluation import replay, summarize
from mdb_steer.llm import Ollama
from mdb_steer.router import Router, knn_vote
from mdb_steer.store import Store

Log = Callable[[str], None]


@dataclass(frozen=True)
class Attempt:
    model: str
    answer: str
    score: float
    verdict: str
    reason: str
    latency_ms: float
    compute_ms: float
    load_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


def load_dataset(path: Path) -> list[dict[str, str]]:
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for item in items:
        missing = {"id", "query", "reference"} - item.keys()
        if missing:
            raise ValueError(f"{path}: item {item.get('id', '?')} missing {sorted(missing)}")
    return items


def run_attempt(llm: Ollama, settings: Settings, model: str, item: dict[str, str]) -> Attempt:
    completion = llm.chat(model, item["query"], max_tokens=settings.max_tokens)
    graded = judge.grade(llm, settings.judge_model, item["query"], item["reference"], completion.text)
    return Attempt(
        model=model,
        answer=completion.text,
        score=graded.score,
        verdict=graded.verdict,
        reason=graded.reason,
        latency_ms=completion.latency_ms,
        compute_ms=completion.compute_ms,
        load_ms=completion.load_ms,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        cost_usd=settings.cost_usd(completion.compute_ms),
    )


def calibrate(
    store: Store, llm: Ollama, settings: Settings, path: Path, log: Log = print, *, force: bool = False
) -> int:
    """Run both models on every calibration item, grade them, index the results, and fit the router."""
    items = load_dataset(path)
    # Resume support: skip items already calibrated with server-side compute timings.
    done = {d["_id"] for d in store.calibration.find({"attempts.strong.compute_ms": {"$exists": True}}, {"_id": 1})}
    dimensions = 0
    for i, item in enumerate(items, 1):
        if item["id"] in done and not force:
            continue
        strong = run_attempt(llm, settings, settings.strong_model, item)
        weak = run_attempt(llm, settings, settings.weak_model, item)
        embedding = llm.embed(settings.embed_model, item["query"])
        dimensions = len(embedding)
        store.upsert_calibration(
            {
                "_id": item["id"],
                "query": item["query"],
                "reference": item["reference"],
                "category": item.get("category"),
                "embedding": embedding,
                "scores": {"strong": strong.score, "weak": weak.score},
                "attempts": {"strong": asdict(strong), "weak": asdict(weak)},
                "calibrated_at": time.time(),
            }
        )
        log(f"  [{i:>3}/{len(items)}] {item['id']:<6} strong={strong.score:.2f} weak={weak.score:.2f}")

    if dimensions or done:
        log("  waiting for vector index...")
        store.ensure_vector_index(dimensions or len(store.calibration.find_one({}, {"embedding": 1})["embedding"]))
        log_fit(Router(store, llm, settings).fit(), log)
    return len(items)


def log_fit(model: dict[str, Any], log: Log) -> None:
    weights = "  ".join(f"{f}={w:+.2f}" for f, w in zip(model["features"], model["weights"]))
    chosen = next(c for c in model["cv_curve"] if c["threshold"] == model["threshold"])
    log(f"  fitted router on {model['n']} queries ({model['positives']} labelled strong-needed, {model.get('label', 'strong_wins')}): "
        f"train accuracy {model['train_accuracy']:.0%}, cross-validated {model['cv_accuracy']:.0%}")
    log(f"  weights: {weights}  bias={model['bias']:+.2f}")
    log(f"  threshold {model['threshold']:.2f} (cross-validated): offload {chosen['offload']:.0%}, "
        f"savings {chosen['cost_savings_pct']:.1f}%, quality drop {chosen['quality_drop_pct']:.2f}%"
        f"{'' if chosen['passed'] else '  [no threshold meets the guardrail]'}")


def regrade(store: Store, llm: Ollama, settings: Settings, path: Path, log: Log = print) -> int:
    """Re-judge stored calibration answers with the current judge, without regenerating them."""
    references = {item["id"]: item["reference"] for item in load_dataset(path)}
    docs = list(store.calibration.find({}, {"embedding": 0}))
    for i, doc in enumerate(docs, 1):
        reference = references.get(doc["_id"], doc.get("reference"))
        if reference is None:
            log(f"  [{i:>3}/{len(docs)}] {doc['_id']:<6} skipped: no reference")
            continue
        update: dict[str, Any] = {"reference": reference}
        for side in ("strong", "weak"):
            graded = judge.grade(llm, settings.judge_model, doc["query"], reference, doc["attempts"][side]["answer"])
            update |= {
                f"scores.{side}": graded.score,
                f"attempts.{side}.score": graded.score,
                f"attempts.{side}.verdict": graded.verdict,
                f"attempts.{side}.reason": graded.reason,
            }
        store.calibration.update_one({"_id": doc["_id"]}, {"$set": update})
        log(f"  [{i:>3}/{len(docs)}] {doc['_id']:<6} strong={update['scores.strong']:.1f} weak={update['scores.weak']:.1f}")
    return len(docs)


def benchmark(store: Store, llm: Ollama, settings: Settings, path: Path, log: Log = print) -> dict[str, Any]:
    """Route each held-out query, and also run both models so every strategy can be compared."""
    router = Router(store, llm, settings)
    # Pin the effective threshold (explicit or cross-validated) so the summary records it.
    settings = replace(settings, threshold=router.threshold)
    run_id = f"run-{int(time.time())}"
    rows: list[dict[str, Any]] = []

    for i, item in enumerate(load_dataset(path), 1):
        decision = router.route(item["query"])
        strong = run_attempt(llm, settings, settings.strong_model, item)
        weak = run_attempt(llm, settings, settings.weak_model, item)
        chosen = strong if decision.model == settings.strong_model else weak

        row = {
            "run_id": run_id,
            "query_id": item["id"],
            "category": item.get("category"),
            "query": item["query"],
            "model_chosen": decision.model,
            "p_strong": decision.p_strong,
            "router_overhead_ms": decision.overhead_ms,
            "router_compute_ms": decision.compute_ms,
            "router_fitted": decision.fitted,
            "features": decision.features,
            "neighbors": decision.neighbors,
            "chosen": asdict(chosen),
            "strong": asdict(strong),
            "weak": asdict(weak),
            "timestamp": time.time(),
        }
        store.telemetry.insert_one(dict(row))
        rows.append(row)
        log(f"  [{i:>3}] {item['id']:<6} p_strong={decision.p_strong:.2f} -> {decision.model}")

    summary = summarize(rows, settings)
    summary["run_id"] = run_id
    store.runs.insert_one(dict(summary))
    return summary


def sweep(
    store: Store, settings: Settings, run_id: str | None, thresholds: list[float], *, rescore: bool = False
) -> list[dict[str, Any]]:
    """Replay a stored benchmark run at different thresholds. No model calls: telemetry already
    holds routing features and both models' attempts for every query.

    With `rescore`, p_strong is recomputed from the stored features using the latest fitted router,
    so a new fit can be evaluated on the held-out set without re-running any model."""
    if run_id is None:
        latest = store.runs.find_one(sort=[("_id", -1)])
        if latest is None:
            raise LookupError("no benchmark runs found; run `benchmark` first")
        run_id = latest["run_id"]
    rows = list(store.telemetry.find({"run_id": run_id}))
    if not rows:
        raise LookupError(f"no telemetry for run {run_id!r}")
    if rescore:
        model = store.latest_router_model()
        if model is None:
            raise LookupError("no fitted router model; run `fit` first")
        missing = [r["query_id"] for r in rows if "features" not in r]
        if missing:
            raise LookupError(f"run {run_id!r} has no stored features for {missing}; re-run `benchmark`")
        rows = [r | {"p_strong": F.predict(model, [_rescored_features(r, settings)[f] for f in model["features"]])} for r in rows]

    points = []
    for t in thresholds:
        s = replace(settings, threshold=t)
        summary = summarize(replay(rows, s, t), s)
        points.append({"threshold": t, "run_id": run_id} | {
            k: summary[k] for k in ("quality_drop_pct", "cost_savings_pct", "lift_over_random", "passed")
        } | {"offload": summary["strategies"]["router"]["offload"]})
    return points


def _rescored_features(row: dict[str, Any], settings: Settings) -> dict[str, float]:
    """Stored features, with the kNN vote recomputed from stored neighbours under the current label."""
    neighbors = [(n["similarity"], {"strong": n["strong"], "weak": n["weak"]}) for n in row.get("neighbors", [])]
    return row["features"] | ({"knn_p_strong": knn_vote(neighbors, settings)} if neighbors else {})
