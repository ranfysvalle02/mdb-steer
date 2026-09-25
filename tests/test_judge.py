import pytest

from mdb_steer.judge import _parse


@pytest.mark.parametrize(
    ("text", "score", "verdict"),
    [
        ('{"reason": "matches", "verdict": "CORRECT"}', 1.0, "CORRECT"),
        ('{"reason": "cut off", "verdict": "partial"}', 0.5, "PARTIAL"),
        ('{"reason": "wrong", "verdict": "INCORRECT"}', 0.0, "INCORRECT"),
        ("The answer is INCORRECT.", 0.0, "INCORRECT"),  # INCORRECT must win over its substring CORRECT
        ("Looks CORRECT to me", 1.0, "CORRECT"),
        ('{"verdict": "MAYBE"}', 0.0, "INCORRECT"),
        ("no verdict at all", 0.0, "INCORRECT"),
    ],
)
def test_parse(text: str, score: float, verdict: str) -> None:
    grade = _parse(text)
    assert (grade.score, grade.verdict) == (score, verdict)
