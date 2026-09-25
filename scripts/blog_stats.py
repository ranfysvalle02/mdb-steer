"""Reproduce the bootstrap CIs and cloud-pricing tables in blog.md / blog2.md / review.md.

Reads a stored benchmark run from MongoDB (no model calls), rescored with the latest fitted router:

    python scripts/blog_stats.py [RUN_ID]
"""

from __future__ import annotations

import random
import sys
from dataclasses import replace

from mdb_steer import features as F
from mdb_steer.config import Settings
from mdb_steer.evaluation import replay, summarize
from mdb_steer.pipeline import _rescored_features
from mdb_steer.store import Store

RESAMPLES = 2000


def stats(rows, settings, threshold):
    summary = summarize(replay(rows, settings, threshold), replace(settings, threshold=threshold))
    st = summary["strategies"]
    base = st["all_strong"]["cost_usd"]
    return {
        "hindsight_savings": (base - st["hindsight"]["cost_usd"]) / base * 100,
        "all_weak_savings": (base - st["all_weak"]["cost_usd"]) / base * 100,
        "router_savings": summary["cost_savings_pct"],
        "quality_drop": summary["quality_drop_pct"],
        "lift_over_random": summary["lift_over_random"],
    }


def ci(values):
    xs = sorted(values)
    return xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]


def token_priced(rows, ratio):
    """Price each attempt per token (output at 3x input), with the weak model `ratio` times cheaper."""
    def cost(attempt, unit):
        return unit * (attempt["prompt_tokens"] + 3 * attempt["completion_tokens"])
    return [
        r | {"strong": r["strong"] | {"cost_usd": cost(r["strong"], 1.0)}, "weak": r["weak"] | {"cost_usd": cost(r["weak"], 1 / ratio)}}
        for r in rows
    ]


def main() -> None:
    settings = Settings()
    store = Store(settings.mongodb_uri, settings.db_name)
    run_id = sys.argv[1] if len(sys.argv) > 1 else store.runs.find_one(sort=[("_id", -1)])["run_id"]
    model = store.latest_router_model()
    rows = [
        r | {"p_strong": F.predict(model, [_rescored_features(r, settings)[f] for f in model["features"]])}
        for r in store.telemetry.find({"run_id": run_id})
    ]
    rng = random.Random(0)
    print(f"run {run_id}: {len(rows)} queries, label {model.get('label')}, cross-validated threshold {model.get('threshold')}")

    for threshold in (0.40, model.get("threshold", 0.5)):
        point = stats(rows, settings, threshold)
        boots = [stats([rng.choice(rows) for _ in rows], settings, threshold) for _ in range(RESAMPLES)]
        print(f"\nthreshold {threshold:.2f}")
        for key, value in point.items():
            lo, hi = ci(b[key] for b in boots)
            print(f"  {key:<18} {value:7.2f}  95% CI [{lo:7.2f}, {hi:7.2f}]  P(>0)={sum(b[key] > 0 for b in boots) / RESAMPLES:.2f}")
        print(f"  P(quality drop <= {settings.quality_guardrail_pct}%) = "
              f"{sum(b['quality_drop'] <= settings.quality_guardrail_pct for b in boots) / RESAMPLES:.2f}")

    print("\ntoken pricing (strong:weak per-token price ratio), router at threshold 0.40")
    for ratio in (1.35, 2, 5, 10, 20):
        priced = token_priced(rows, ratio)
        point = stats(priced, settings, 0.40)
        boots = [stats([rng.choice(priced) for _ in priced], settings, 0.40) for _ in range(RESAMPLES // 2)]
        (hlo, hhi), (rlo, rhi) = ci(b["hindsight_savings"] for b in boots), ci(b["router_savings"] for b in boots)
        print(f"  {ratio:>5}x  hindsight {point['hindsight_savings']:5.1f}% [{hlo:5.1f}, {hhi:5.1f}]  "
              f"all-weak {point['all_weak_savings']:5.1f}%  router {point['router_savings']:5.1f}% [{rlo:5.1f}, {rhi:5.1f}]")


if __name__ == "__main__":
    main()
