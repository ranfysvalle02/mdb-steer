"""Command-line entry point: `python -m mdb_steer {calibrate,benchmark,route}`."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from mdb_steer.config import Settings
from mdb_steer.llm import Ollama
from mdb_steer.pipeline import benchmark, calibrate, log_fit, regrade, sweep
from mdb_steer.router import Router
from mdb_steer.store import Store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mdb-steer", description=__doc__)
    parser.add_argument("--threshold", type=float, help="override ROUTER_THRESHOLD")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser("calibrate", help="run both models on the calibration set and build the vector index")
    p_cal.add_argument("--data", type=Path, default=Path("data/calibration.jsonl"))
    p_cal.add_argument("--force", action="store_true", help="re-run items that are already calibrated")

    p_regrade = sub.add_parser("regrade", help="re-judge stored calibration answers with the current judge")
    p_regrade.add_argument("--data", type=Path, default=Path("data/calibration.jsonl"))

    sub.add_parser("fit", help="refit the router's feature weights on the current calibration set")

    p_bench = sub.add_parser("benchmark", help="evaluate routing on held-out queries; exits 1 if the guardrail fails")
    p_bench.add_argument("--data", type=Path, default=Path("data/benchmark.jsonl"))

    p_sweep = sub.add_parser("sweep", help="replay a stored benchmark run across thresholds (no model calls)")
    p_sweep.add_argument("--run-id", help="defaults to the latest run")
    p_sweep.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    p_sweep.add_argument("--rescore", action="store_true", help="recompute p_strong with the latest fitted router")

    p_route = sub.add_parser("route", help="route a single query and optionally answer it")
    p_route.add_argument("query")
    p_route.add_argument("--answer", action="store_true", help="also execute the query on the chosen model")

    args = parser.parse_args(argv)

    settings = Settings()
    if args.threshold is not None:
        settings = replace(settings, threshold=args.threshold)
    store = Store(settings.mongodb_uri, settings.db_name)
    store.ping()
    llm = Ollama(settings.ollama_url)

    if args.command == "calibrate":
        print(f"Calibrating {settings.strong_model} vs {settings.weak_model} (judge: {settings.judge_model})")
        n = calibrate(store, llm, settings, args.data, force=args.force)
        print(f"Indexed {n} calibration queries.")
        return 0

    if args.command == "regrade":
        print(f"Regrading calibration answers (judge: {settings.judge_model})")
        n = regrade(store, llm, settings, args.data)
        print(f"Regraded {n} calibration queries.")
        return 0

    if args.command == "fit":
        log_fit(Router(store, llm, settings).fit(), print)
        return 0

    if args.command == "benchmark":
        threshold = settings.threshold if settings.threshold is not None else "cross-validated"
        print(f"Benchmarking router (threshold={threshold}, k={settings.k})")
        summary = benchmark(store, llm, settings, args.data)
        _print_report(summary)
        return 0 if summary["passed"] else 1

    if args.command == "sweep":
        points = sweep(store, settings, args.run_id, [float(t) for t in args.thresholds.split(",")], rescore=args.rescore)
        print(f"run {points[0]['run_id']}\n")
        print(f"{'threshold':>9} {'offload':>8} {'savings':>8} {'q drop':>8} {'vs random':>10}  guardrail")
        for p in points:
            print(
                f"{p['threshold']:>9.2f} {p['offload']:>8.0%} {p['cost_savings_pct']:>7.1f}% "
                f"{p['quality_drop_pct']:>7.2f}% {p['lift_over_random']:>+10.3f}  {'PASS' if p['passed'] else 'FAIL'}"
            )
        return 0

    decision = Router(store, llm, settings).route(args.query)
    print(f"model:    {decision.model}")
    print(f"p_strong: {decision.p_strong:.3f}  ({'fitted' if decision.fitted else 'kNN only'}, overhead {decision.overhead_ms:.0f} ms)")
    print("features: " + "  ".join(f"{k}={v:.2f}" for k, v in decision.features.items()))
    for n in decision.neighbors:
        print(f"  {n['id']:<6} sim={n['similarity']:.3f} strong={n['strong']:.2f} weak={n['weak']:.2f}")
    if args.answer:
        print("\n" + llm.chat(decision.model, args.query, max_tokens=settings.max_tokens).text)
    return 0


def _print_report(summary: dict[str, Any]) -> None:
    print(f"\n{'ID':<6} {'category':<12} {'p_strong':>8}  {'routed to':<16} {'strong':>6} {'weak':>6}")
    for r in summary["rows"]:
        print(
            f"{r['query_id']:<6} {r['category'] or '-':<12} {r['p_strong']:>8.2f}  "
            f"{r['model_chosen']:<16} {r['strong']:>6.2f} {r['weak']:>6.2f}"
        )

    print(f"\n{'strategy':<12} {'quality':>8} {'cost (USD)':>12} {'offload':>8}")
    for name, s in summary["strategies"].items():
        print(f"{name:<12} {s['quality']:>8.3f} {s['cost_usd']:>12.6f} {s['offload']:>8.0%}")

    verdict = "PASS" if summary["passed"] else "FAIL"
    print(
        f"\nqueries={summary['n_queries']}  cost savings={summary['cost_savings_pct']:.1f}%  "
        f"quality drop={summary['quality_drop_pct']:.2f}%  lift over random={summary['lift_over_random']:+.3f}  "
        f"router overhead={summary['mean_router_overhead_ms']:.0f} ms"
    )
    print(f"guardrail (<= {summary['guardrail_pct']:.1f}% drop): {verdict}  [run {summary['run_id']}]")


if __name__ == "__main__":
    sys.exit(main())
