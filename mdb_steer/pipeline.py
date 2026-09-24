"""Calibration and benchmark workflows."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from mdb_steer import features as F
from mdb_steer import judge
from mdb_steer.config import Settings
from mdb_steer.llm import Ollama
from mdb_steer.router import Router
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


def calibrate(store: Store, llm: Ollama, settings: Settings, path: Path, log: Log = print) -> int:
    """Run both models on every calibration item, grade them, index the results, and fit the router."""
    items = load_dataset(path)
    dimensions = 0
    for i, item in enumerate(items, 1):
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

    if dimensions:
        log("  waiting for vector index...")
        store.ensure_vector_index(dimensions)
        log_fit(Router(store, llm, settings).fit(), log)
    return len(items)


def log_fit(model: dict[str, Any], log: Log) -> None:
    weights = "  ".join(f"{f}={w:+.2f}" for f, w in zip(model["features"], model["weights"]))
    log(f"  fitted router on {model['n']} queries ({model['positives']} strong wins), "
        f"train accuracy {model['train_accuracy']:.0%}\n  weights: {weights}  bias={model['bias']:+.2f}")


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


def summarize(rows: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    if not rows:
        raise ValueError("benchmark dataset is empty")

    strong_q = [r["strong"]["score"] for r in rows]
    weak_q = [r["weak"]["score"] for r in rows]
    strong_c = [r["strong"]["cost_usd"] for r in rows]
    weak_c = [r["weak"]["cost_usd"] for r in rows]
    offload = sum(r["model_chosen"] == settings.weak_model for r in rows) / len(rows)

    # Oracle: cheapest model that achieves the best score on each query.
    oracle = [("weak" if r["weak"]["score"] >= r["strong"]["score"] else "strong") for r in rows]

    strategies = {
        "all_strong": {"quality": mean(strong_q), "cost_usd": sum(strong_c), "offload": 0.0},
        "all_weak": {"quality": mean(weak_q), "cost_usd": sum(weak_c), "offload": 1.0},
        "router": {
            "quality": mean(r["chosen"]["score"] for r in rows),
            # Routing is not free: include the self-confidence call's compute.
            "cost_usd": sum(r["chosen"]["cost_usd"] + settings.cost_usd(r.get("router_compute_ms", 0.0)) for r in rows),
            "offload": offload,
        },
        # Expected value of routing the same share of traffic to the weak model at random.
        "random": {
            "quality": (1 - offload) * mean(strong_q) + offload * mean(weak_q),
            "cost_usd": (1 - offload) * sum(strong_c) + offload * sum(weak_c),
            "offload": offload,
        },
        "oracle": {
            "quality": mean(r[m]["score"] for r, m in zip(rows, oracle)),
            "cost_usd": sum(r[m]["cost_usd"] for r, m in zip(rows, oracle)),
            "offload": oracle.count("weak") / len(rows),
        },
    }

    base, routed = strategies["all_strong"], strategies["router"]
    quality_drop_pct = (base["quality"] - routed["quality"]) / base["quality"] * 100 if base["quality"] else 0.0
    cost_savings_pct = (base["cost_usd"] - routed["cost_usd"]) / base["cost_usd"] * 100 if base["cost_usd"] else 0.0

    return {
        "n_queries": len(rows),
        "strategies": strategies,
        "quality_drop_pct": quality_drop_pct,
        "cost_savings_pct": cost_savings_pct,
        "lift_over_random": routed["quality"] - strategies["random"]["quality"],
        "mean_router_overhead_ms": mean(r["router_overhead_ms"] for r in rows),
        "guardrail_pct": settings.quality_guardrail_pct,
        "passed": quality_drop_pct <= settings.quality_guardrail_pct,
        "settings": asdict(settings),
        "rows": [
            {k: r[k] for k in ("query_id", "category", "model_chosen", "p_strong")}
            | {"strong": r["strong"]["score"], "weak": r["weak"]["score"]}
            for r in rows
        ],
    }


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
        rows = [r | {"p_strong": F.predict(model, [r["features"][f] for f in model["features"]])} for r in rows]

    points = []
    for t in thresholds:
        s = replace(settings, threshold=t)
        replayed = [
            r | {"model_chosen": s.strong_model if r["p_strong"] >= t else s.weak_model,
                 "chosen": r["strong"] if r["p_strong"] >= t else r["weak"]}
            for r in rows
        ]
        summary = summarize(replayed, s)
        points.append({"threshold": t, "run_id": run_id} | {
            k: summary[k] for k in ("quality_drop_pct", "cost_savings_pct", "lift_over_random", "passed")
        } | {"offload": summary["strategies"]["router"]["offload"]})
    return points
