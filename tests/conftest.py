from __future__ import annotations

from typing import Any

import pytest

from mdb_steer.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        strong_model="strong", weak_model="weak", embed_model="embed", judge_model="strong",
        threshold=None, k=3, win_margin=0.1, router_label="strong_wins", gpu_cost_per_hour=3.6, quality_guardrail_pct=5.0,
        self_confidence=False, cv_folds=2,
    )


def attempt(score: float, compute_ms: float) -> dict[str, Any]:
    # gpu_cost_per_hour=3.6 makes cost_usd == compute_ms / 1e6: easy to reason about.
    return {"score": score, "compute_ms": compute_ms, "cost_usd": compute_ms / 1e6}


def row(qid: str, strong: float, weak: float, p_strong: float, strong_ms: float = 1000, weak_ms: float = 200) -> dict[str, Any]:
    return {"query_id": qid, "category": "c", "p_strong": p_strong,
            "strong": attempt(strong, strong_ms), "weak": attempt(weak, weak_ms)}
