import pytest

from mdb_steer.evaluation import replay, select_threshold, summarize

from .conftest import row


def rows():
    # q1/q2 easy (weak matches strong), q3/q4 hard (strong wins). The router scores them correctly.
    return [row("q1", 1.0, 1.0, 0.1), row("q2", 1.0, 1.0, 0.2), row("q3", 1.0, 0.0, 0.8), row("q4", 1.0, 0.0, 0.9)]


def test_summarize_strategies(settings) -> None:
    summary = summarize(replay(rows(), settings, 0.5), settings)
    st = summary["strategies"]
    assert st["all_strong"] == {"quality": 1.0, "cost_usd": pytest.approx(0.004), "offload": 0.0}
    assert st["all_weak"]["quality"] == 0.5
    assert st["router"]["quality"] == 1.0 and st["router"]["offload"] == 0.5
    assert st["random"]["quality"] == pytest.approx(0.75)
    assert st["hindsight"]["offload"] == 0.5
    assert summary["lift_over_random"] == pytest.approx(0.25)
    assert summary["passed"] is True
    assert summary["cost_savings_pct"] == pytest.approx(40.0)  # 2 * 1000 -> 2 * 200 of 4000


def test_router_compute_is_charged(settings) -> None:
    charged = [r | {"router_compute_ms": 500} for r in rows()]
    free = summarize(replay(rows(), settings, 0.5), settings)
    paid = summarize(replay(charged, settings, 0.5), settings)
    assert paid["strategies"]["router"]["cost_usd"] == pytest.approx(free["strategies"]["router"]["cost_usd"] + 0.002)


def test_guardrail_fails_when_quality_drops(settings) -> None:
    summary = summarize(replay(rows(), settings, 0.95), settings)  # everything to weak
    assert summary["quality_drop_pct"] == pytest.approx(50.0)
    assert summary["passed"] is False


def test_select_threshold_maximises_savings_within_guardrail(settings) -> None:
    threshold, curve = select_threshold(rows(), settings, [0.05, 0.5, 0.95])
    assert threshold == 0.5
    assert [c["passed"] for c in curve] == [True, True, False]


def test_select_threshold_falls_back_to_most_conservative(settings) -> None:
    bad = [row("q1", 1.0, 0.0, 0.9), row("q2", 1.0, 0.0, 0.8)]  # router is confidently wrong
    threshold, _ = select_threshold(bad, settings, [0.95, 0.99])
    assert threshold == 0.95


def test_summarize_rejects_empty(settings) -> None:
    with pytest.raises(ValueError):
        summarize([], settings)
