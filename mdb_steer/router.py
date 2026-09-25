"""Learned router.

Each calibration query gets a label saying the strong model was needed (see `needs_strong`). For an
incoming query we compute:
  - knn_p_strong: similarity-weighted share of the k nearest calibration queries (Atlas Vector
    Search) whose label says the strong model was needed;
  - text difficulty features (multi-step wording, length, numbers);
  - weak_uncertainty: the weak model's own low confidence (optional, costs one short call).

A logistic model fitted on the calibration set turns these into P(strong needed), `p_strong`. Nothing about the
incoming query's difficulty is given to the router: it only sees the text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from mdb_steer import features as F
from mdb_steer.config import Settings
from mdb_steer.evaluation import select_threshold
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
            model=s.strong_model if p_strong >= self.threshold else s.weak_model,
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
        return knn_vote([(n["similarity"], n["scores"]) for n in neighbors], self._settings)

    @property
    def threshold(self) -> float:
        """Explicit ROUTER_THRESHOLD wins; otherwise the cross-validated one from the fitted model."""
        if self._settings.threshold is not None:
            return self._settings.threshold
        return self._model.get("threshold", 0.5) if self._model else 0.5

    def fit(self) -> dict[str, Any]:
        """Fit the logistic combiner on the calibration set and choose the routing threshold.

        Features use leave-one-out kNN (a query never votes for itself). The threshold is chosen from
        out-of-fold predictions (k-fold CV) so it is never tuned on the benchmark set.
        """
        s = self._settings
        docs = sorted(
            self._store.calibration.find({}, {"query": 1, "category": 1, "embedding": 1, "scores": 1, "attempts": 1}),
            key=lambda d: d["_id"],
        )
        if len(docs) < 2 * s.cv_folds:
            raise ValueError(f"need at least {2 * s.cv_folds} calibration queries to fit, have {len(docs)}")

        names = F.feature_names(s.self_confidence)
        X, y, router_ms = [], [], []
        for doc in docs:
            feats, _, compute_ms = self.features(doc["query"], embedding=doc["embedding"], exclude_id=doc["_id"])
            X.append([feats[f] for f in names])
            y.append(int(needs_strong(doc["scores"], s)))
            router_ms.append(compute_ms)

        # Out-of-fold P(strong wins) for every calibration query.
        oof = [0.0] * len(docs)
        for fold in range(s.cv_folds):
            train = [i for i in range(len(docs)) if i % s.cv_folds != fold]
            fold_model = F.fit_logistic([X[i] for i in train], [y[i] for i in train], l2=s.l2)
            for i in range(fold, len(docs), s.cv_folds):
                oof[i] = F.predict(fold_model, X[i])

        cv_rows = [
            {
                "query_id": d["_id"],
                "category": d.get("category"),
                "p_strong": p,
                "strong": d["attempts"]["strong"],
                "weak": d["attempts"]["weak"],
                "router_compute_ms": ms,
            }
            for d, p, ms in zip(docs, oof, router_ms)
        ]
        grid = [round(0.05 * i, 2) for i in range(1, 20)]
        threshold, curve = select_threshold(cv_rows, s, grid)

        model = {"features": names, **F.fit_logistic(X, y, l2=s.l2)}
        preds = [F.predict(model, x) >= threshold for x in X]
        model |= {
            "threshold": threshold,
            "cv_curve": curve,
            "cv_accuracy": sum((p >= threshold) == bool(t) for p, t in zip(oof, y)) / len(y),
            "train_accuracy": sum(p == bool(t) for p, t in zip(preds, y)) / len(y),
            "n": len(docs),
            "positives": sum(y),
            "training_rows": [dict(zip(names, x)) | {"id": d["_id"], "label": t} for x, t, d in zip(X, y, docs)],
            "self_confidence": s.self_confidence,
            "label": s.router_label,
            "fitted_at": time.time(),
        }
        self._store.save_router_model(model)
        self._model = model
        return model


def needs_strong(scores: dict[str, float], settings: Settings) -> bool:
    """Training label: did this calibration query need the strong model?"""
    if settings.router_label == "strong_wins":
        return scores["strong"] - scores["weak"] > settings.win_margin
    if settings.router_label == "weak_fails":
        return scores["weak"] < 1.0
    raise ValueError(f"unknown ROUTER_LABEL {settings.router_label!r}")


def knn_vote(neighbors: list[tuple[float, dict[str, float]]], settings: Settings) -> float:
    """Similarity-weighted share of (similarity, scores) neighbours that needed the strong model."""
    total = sum(sim for sim, _ in neighbors)
    if total <= 0:
        return 1.0  # no evidence: fail safe towards quality
    return sum(sim for sim, scores in neighbors if needs_strong(scores, settings)) / total
