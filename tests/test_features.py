from mdb_steer.features import feature_names, fit_logistic, predict, text_features


def test_text_features_separate_easy_from_multi_step() -> None:
    easy = text_features("What is the capital of Japan?")
    hard = text_features("Explain why switching wins in Monty Hall, step by step, and compute the probability.")
    assert easy["multi_step"] == 0.0
    assert hard["multi_step"] == 1.0
    assert hard["length"] > easy["length"]
    assert text_features("What is 15% of 200?")["numeric"] == 1.0


def test_feature_names_drop_self_confidence_when_disabled() -> None:
    assert "weak_uncertainty" not in feature_names(False)
    assert "weak_uncertainty" in feature_names(True)


def test_fit_logistic_learns_signal_and_ignores_noise() -> None:
    X = [[0.0, 0.1], [0.1, 0.9], [0.9, 0.2], [1.0, 0.8]] * 5
    y = [0, 0, 1, 1] * 5
    model = fit_logistic(X, y)
    assert predict(model, [1.0, 0.5]) > 0.9
    assert predict(model, [0.0, 0.5]) < 0.1
    assert abs(model["weights"][0]) > 5 * abs(model["weights"][1])


def test_class_balancing_prevents_majority_collapse() -> None:
    # 1 positive per 5 rows: an unbalanced fit would push every prediction below 0.5.
    X = [[1.0]] + [[0.0]] * 4
    model = fit_logistic(X * 4, [1, 0, 0, 0, 0] * 4)
    assert predict(model, [1.0]) > 0.5 > predict(model, [0.0])


def test_constant_feature_does_not_blow_up() -> None:
    model = fit_logistic([[0.5, 0.0], [0.5, 1.0]] * 3, [0, 1] * 3)
    assert 0.0 < predict(model, [0.5, 0.5]) < 1.0


def test_predict_accepts_unstandardised_legacy_models() -> None:
    assert predict({"weights": [0.0], "bias": 0.0}, [123.0]) == 0.5
