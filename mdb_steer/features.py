"""Routing features and the tiny logistic model that combines them.

Neighbour similarity tells us *what topic* a query is about, but not how hard it is: a grammar
fix and a syllogism can sit next to easy queries in embedding space. So alongside the kNN vote we
add cheap difficulty signals from the text, plus the weak model's own confidence.
"""

from __future__ import annotations

import math
import re

from mdb_steer.llm import Completion, Ollama

FEATURES = ("knn_p_strong", "multi_step", "length", "numeric", "weak_uncertainty")

# Wording that tends to signal multi-step reasoning, derivation, or non-trivial code.
_MULTI_STEP = re.compile(
    r"\b(explain|why|implement|design|prove|derive|optimi[sz]e|compare|step|steps|how many|"
    r"probability|if|then|all but|at least|each|every|consensus|concurren\w*|thread-safe|"
    r"algorithm|trade-?offs?)\b",
    re.IGNORECASE,
)

_CONFIDENCE_PROMPT = """Before answering, rate how confident you are that you can answer the question
below completely and correctly. Reply with a single integer from 0 (certainly wrong) to 10
(certainly right) and nothing else.

Question:
{question}"""


def text_features(query: str) -> dict[str, float]:
    return {
        "multi_step": min(1.0, len(_MULTI_STEP.findall(query)) / 3),
        "length": min(1.0, len(query.split()) / 40),
        "numeric": 1.0 if re.search(r"\d", query) else 0.0,
    }


def weak_uncertainty(llm: Ollama, weak_model: str, query: str) -> tuple[float, Completion]:
    """1 - (weak model's self-rated confidence / 10). Unparseable replies count as 0.5."""
    completion = llm.chat(weak_model, _CONFIDENCE_PROMPT.format(question=query), max_tokens=4)
    match = re.search(r"\d+", completion.text)
    if not match:
        return 0.5, completion
    return 1.0 - min(int(match.group()), 10) / 10, completion


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def fit_logistic(
    X: list[list[float]], y: list[int], *, l2: float = 0.01, lr: float = 0.5, epochs: int = 4000
) -> dict[str, list[float] | float]:
    """Class-balanced, L2-regularised logistic regression on standardised features.

    Balancing matters because strong-model wins are the minority class: unweighted, the model can
    score well by always predicting "weak is fine". Returns the weights plus the scaling used, so
    `predict` can apply the same standardisation.
    """
    n, d = len(X), len(X[0])
    means = [sum(x[j] for x in X) / n for j in range(d)]
    stds = [max(1e-6, math.sqrt(sum((x[j] - means[j]) ** 2 for x in X) / n)) for j in range(d)]
    Z = [[(x[j] - means[j]) / stds[j] for j in range(d)] for x in X]

    pos = sum(y)
    class_weight = {1: n / (2 * max(pos, 1)), 0: n / (2 * max(n - pos, 1))}

    w, b = [0.0] * d, 0.0
    for _ in range(epochs):
        grad_w, grad_b = [0.0] * d, 0.0
        for zi, yi in zip(Z, y):
            err = (_sigmoid(sum(wj * zj for wj, zj in zip(w, zi)) + b) - yi) * class_weight[yi]
            grad_b += err
            for j in range(d):
                grad_w[j] += err * zi[j]
        w = [wj - lr * (gj / n + l2 * wj) for wj, gj in zip(w, grad_w)]
        b -= lr * grad_b / n
    return {"weights": w, "bias": b, "means": means, "stds": stds}


def predict(model: dict, x: list[float]) -> float:
    z = [(xi - m) / sd for xi, m, sd in zip(x, model.get("means", [0.0] * len(x)), model.get("stds", [1.0] * len(x)))]
    return _sigmoid(sum(w * zi for w, zi in zip(model["weights"], z)) + model["bias"])
