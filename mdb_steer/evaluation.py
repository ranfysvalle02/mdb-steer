"""Scoring a routing policy against the alternatives (all-strong, all-weak, random, oracle).

Rows are dicts holding both models' attempts (`strong`, `weak`), the routed choice
(`model_chosen`, `chosen`) and optional `router_compute_ms`. Used by the benchmark, threshold
sweeps, and cross-validated threshold selection in `fit`.
"""

from __future__ import annotations

from dataclasses import asdict
from statistics import mean
from typing import Any

from mdb_steer.config import Settings


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
        "mean_router_overhead_ms": mean(r.get("router_overhead_ms", 0.0) for r in rows),
        "threshold": settings.threshold,
        "guardrail_pct": settings.quality_guardrail_pct,
        "passed": quality_drop_pct <= settings.quality_guardrail_pct,
        "settings": asdict(settings),
        "rows": [
            {k: r.get(k) for k in ("query_id", "category", "model_chosen", "p_strong")}
            | {"strong": r["strong"]["score"], "weak": r["weak"]["score"]}
            for r in rows
        ],
    }


def replay(rows: list[dict[str, Any]], settings: Settings, threshold: float) -> list[dict[str, Any]]:
    """Re-route stored rows (which carry `p_strong`) at a different threshold."""
    return [
        r | ({"model_chosen": settings.strong_model, "chosen": r["strong"]} if r["p_strong"] >= threshold
             else {"model_chosen": settings.weak_model, "chosen": r["weak"]})
        for r in rows
    ]


def select_threshold(
    rows: list[dict[str, Any]], settings: Settings, grid: list[float]
) -> tuple[float, list[dict[str, Any]]]:
    """Pick the threshold with the largest cost saving whose quality drop stays within the guardrail.

    Falls back to the lowest threshold (most traffic on the strong model) if none qualifies.
    Returns the chosen threshold and the full curve for inspection.
    """
    curve = []
    for t in grid:
        summary = summarize(replay(rows, settings, t), settings)
        curve.append({
            "threshold": t,
            "offload": summary["strategies"]["router"]["offload"],
            "cost_savings_pct": summary["cost_savings_pct"],
            "quality_drop_pct": summary["quality_drop_pct"],
            "passed": summary["passed"],
        })
    passing = [c for c in curve if c["passed"]]
    best = max(passing, key=lambda c: (c["cost_savings_pct"], c["threshold"])) if passing else min(curve, key=lambda c: c["threshold"])
    return best["threshold"], curve
