from dataclasses import replace

from mdb_steer.router import Router, needs_strong


class FakeStore:
    def __init__(self, neighbors, model=None):
        self._neighbors, self._model = neighbors, model

    def latest_router_model(self):
        return self._model

    def nearest(self, embedding, k):
        return self._neighbors[:k]


class FakeLLM:
    def embed(self, model, text):
        return [0.0]


def n(_id, sim, strong, weak):
    return {"_id": _id, "similarity": sim, "scores": {"strong": strong, "weak": weak}}


def test_knn_vote_is_similarity_weighted(settings) -> None:
    router = Router(FakeStore([]), FakeLLM(), settings)
    vote = router.knn_p_strong([n("a", 0.9, 1.0, 0.0), n("b", 0.3, 1.0, 1.0)])
    assert vote == 0.9 / 1.2


def test_knn_vote_respects_win_margin(settings) -> None:
    router = Router(FakeStore([]), FakeLLM(), settings)
    assert router.knn_p_strong([n("a", 1.0, 1.0, 0.95)]) == 0.0


def test_no_evidence_fails_safe_to_strong(settings) -> None:
    decision = Router(FakeStore([]), FakeLLM(), settings).route("anything")
    assert decision.p_strong == 1.0 and decision.model == "strong" and not decision.fitted


def test_leave_one_out_excludes_self(settings) -> None:
    store = FakeStore([n("self", 1.0, 1.0, 0.0), n("x", 0.8, 1.0, 1.0), n("y", 0.7, 1.0, 1.0), n("z", 0.6, 1.0, 1.0)])
    _, neighbors, _ = Router(store, FakeLLM(), settings).features("q", embedding=[0.0], exclude_id="self")
    assert [m["_id"] for m in neighbors] == ["x", "y", "z"]


def test_threshold_precedence(settings) -> None:
    model = {"features": ["knn_p_strong"], "weights": [0.0], "bias": 0.0, "threshold": 0.3}
    assert Router(FakeStore([], model), FakeLLM(), settings).threshold == 0.3
    assert Router(FakeStore([], model), FakeLLM(), replace(settings, threshold=0.7)).threshold == 0.7
    assert Router(FakeStore([]), FakeLLM(), settings).threshold == 0.5


def test_fitted_model_drives_decision(settings) -> None:
    # Strongly negative bias: P(strong) ~ 0 regardless of features -> weak.
    model = {"features": ["knn_p_strong"], "weights": [0.0], "bias": -10.0, "threshold": 0.5}
    decision = Router(FakeStore([n("a", 1.0, 1.0, 0.0)], model), FakeLLM(), settings).route("q")
    assert decision.fitted and decision.model == "weak"


def test_weak_fails_label(settings) -> None:
    s = replace(settings, router_label="weak_fails")
    assert needs_strong({"strong": 0.0, "weak": 0.5}, s)  # weak not fully right, even though strong also failed
    assert not needs_strong({"strong": 1.0, "weak": 1.0}, s)
    assert not needs_strong({"strong": 0.0, "weak": 0.5}, settings)  # strong_wins: no win when both fail
    assert Router(FakeStore([]), FakeLLM(), s).knn_p_strong([n("a", 0.6, 1.0, 1.0), n("b", 0.4, 1.0, 0.0)]) == 0.4
