"""Learned router.

For an incoming query we compute:
  - knn_p_strong: similarity-weighted share of the k nearest calibration queries (Atlas Vector
    Search) where the strong model beat the weak one by more than `win_margin`;
  - text difficulty features (multi-step wording, length, numbers);
  - weak_uncertainty: the weak model's own low confidence (optional, costs one short call).

A logistic model fitted on the calibration set turns these into P(strong wins). Nothing about the
incoming query's difficulty is given to the router: it only sees the text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from mdb_steer import features as F
from mdb_steer.config import Settings
from mdb_steer.llm import Ollama
from mdb_steer.store import Store


@dataclass(frozen=True)
class RouteDecision:
    model: str
    p_strong: float
    features: dict[str, float]
    overhead_ms: float  # wall-clock added before the answer starts
    compute_ms: float  # billable model compute spent on routing (self-confidence call)
    neighbors: list[dict[str, Any]]
    fitted: bool  # False when falling back to the raw kNN vote


class Router:
    def __init__(self, store: Store, llm: Ollama, settings: Settings) -> None:
        self._store = store
        self._llm = llm
        self._settings = settings
        self._model = store.latest_router_model()

    def route(self, query: str) -> RouteDecision:
        start = time.perf_counter()
        feats, neighbors, compute_ms = self.features(query)
        if self._model:
            p_strong = F.predict(self._model, [feats[f] for f in self._model["features"]])
        else:
            p_strong = feats["knn_p_strong"]
        overhead_ms = (time.perf_counter() - start) * 1000

        s = self._settings
        return RouteDecision(
            model=s.strong_model if p_strong >= s.threshold else s.weak_model,
            p_strong=p_strong,
            features=feats,
            overhead_ms=overhead_ms,
            compute_ms=compute_ms,
            neighbors=[
                {"id": n["_id"], "similarity": round(n["similarity"], 4), "strong": n["scores"]["strong"], "weak": n["scores"]["weak"]}
                for n in neighbors
            ],
            fitted=self._model is not None,
        )

    def features(
        self, query: str, *, embedding: list[float] | None = None, exclude_id: str | None = None
    ) -> tuple[dict[str, float], list[dict[str, Any]], float]:
        s = self._settings
        if embedding is None:
            embedding = self._llm.embed(s.embed_model, query)
        # When featurising a calibration query for training, drop the query itself (leave-one-out).
        neighbors = self._store.nearest(embedding, s.k + (1 if exclude_id else 0))
        neighbors = [n for n in neighbors if n["_id"] != exclude_id][: s.k]

        feats = {"knn_p_strong": self.knn_p_strong(neighbors), **F.text_features(query), "weak_uncertainty": 0.5}
        compute_ms = 0.0
        if s.self_confidence:
            feats["weak_uncertainty"], completion = F.weak_uncertainty(self._llm, s.weak_model, query)
            compute_ms = completion.compute_ms
        return feats, neighbors, compute_ms

    def knn_p_strong(self, neighbors: list[dict[str, Any]]) -> float:
        # With no evidence, fail safe towards quality.
        total = sum(n["similarity"] for n in neighbors)
        if total <= 0:
            return 1.0
        margin = self._settings.win_margin
        wins = sum(n["similarity"] for n in neighbors if n["scores"]["strong"] - n["scores"]["weak"] > margin)
        return wins / total

    def fit(self) -> dict[str, Any]:
        """Fit the logistic combiner on the calibration set using leave-one-out kNN features."""
        s = self._settings
        docs = list(self._store.calibration.find({}, {"query": 1, "embedding": 1, "scores": 1}))
        if len(docs) < 5:
            raise ValueError(f"need at least 5 calibration queries to fit, have {len(docs)}")

        X, y = [], []
        for doc in docs:
            feats, _, _ = self.features(doc["query"], embedding=doc["embedding"], exclude_id=doc["_id"])
            X.append([feats[f] for f in F.FEATURES])
            y.append(int(doc["scores"]["strong"] - doc["scores"]["weak"] > s.win_margin))

        model = {"features": list(F.FEATURES), **F.fit_logistic(X, y, l2=s.l2)}
        preds = [F.predict(model, x) >= s.threshold for x in X]
        model |= {
            "n": len(docs),
            "training_rows": [dict(zip(F.FEATURES, x)) | {"id": d["_id"], "label": t} for x, t, d in zip(X, y, docs)],
            "positives": sum(y),
            "train_accuracy": sum(p == bool(t) for p, t in zip(preds, y)) / len(y),
            "self_confidence": s.self_confidence,
            "fitted_at": time.time(),
        }
        self._store.save_router_model(model)
        self._model = model
        return model
