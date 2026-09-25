"""Runtime settings, sourced from environment variables with local-dev defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str = field(default_factory=lambda: _env("MONGODB_URI", "mongodb://localhost:27017/?directConnection=true"))
    db_name: str = field(default_factory=lambda: _env("DB_NAME", "mdb_steer"))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))

    strong_model: str = field(default_factory=lambda: _env("STRONG_MODEL", "llama3.1:8b"))
    weak_model: str = field(default_factory=lambda: _env("WEAK_MODEL", "llama3.2:1b"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "nomic-embed-text"))
    # Defaults to the strong model. Point this at a third model to avoid self-grading bias.
    judge_model: str = field(default_factory=lambda: _env("JUDGE_MODEL", _env("STRONG_MODEL", "llama3.1:8b")))

    # Route to the strong model when P(strong needed) >= threshold. Unset: use the threshold `fit`
    # selected by cross-validation on the calibration set (never tuned on the benchmark).
    threshold: float | None = field(
        default_factory=lambda: float(os.environ["ROUTER_THRESHOLD"]) if os.getenv("ROUTER_THRESHOLD") else None
    )
    # Number of calibration neighbours consulted per query.
    k: int = field(default_factory=lambda: int(_env("ROUTER_K", "5")))
    # What the router learns to predict for each calibration query:
    #   "weak_fails"  - the weak model's answer was not fully correct (default; one judge verdict)
    #   "strong_wins" - strong beat weak by more than win_margin (difference of two noisy verdicts;
    #                   on our data this label was unpredictable, AUC ~0.5)
    router_label: str = field(default_factory=lambda: _env("ROUTER_LABEL", "weak_fails"))
    win_margin: float = field(default_factory=lambda: float(_env("ROUTER_WIN_MARGIN", "0.1")))

    # Ask the weak model to rate its own confidence as a routing feature (one short extra call).
    # Off by default: on our data it carried no signal and added ~0.5 s per query.
    self_confidence: bool = field(default_factory=lambda: _env("ROUTER_SELF_CONFIDENCE", "false").lower() == "true")
    # Folds for cross-validated threshold selection in `fit`.
    cv_folds: int = field(default_factory=lambda: int(_env("ROUTER_CV_FOLDS", "5")))
    # L2 regularisation for the logistic combiner; keeps weights sane on small calibration sets.
    l2: float = field(default_factory=lambda: float(_env("ROUTER_L2", "0.01")))

    max_tokens: int = field(default_factory=lambda: int(_env("MAX_TOKENS", "768")))
    # Cost is server-reported model compute time (excluding load) priced at this hourly rate.
    gpu_cost_per_hour: float = field(default_factory=lambda: float(_env("GPU_COST_PER_HOUR", "1.00")))
    # Maximum acceptable quality drop (percent) of routed traffic vs. all-strong.
    quality_guardrail_pct: float = field(default_factory=lambda: float(_env("QUALITY_GUARDRAIL_PCT", "5.0")))

    def cost_usd(self, compute_ms: float) -> float:
        return compute_ms / 3_600_000 * self.gpu_cost_per_hour
