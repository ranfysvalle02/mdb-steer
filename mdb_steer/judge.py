"""LLM-as-judge: grades a candidate answer against a reference answer.

Small judges compress 0-10 scales (wrong answers still get 8/10), so we ask for a categorical
verdict on the *final answer* instead and map it to a score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from mdb_steer.llm import Ollama

VERDICT_SCORES = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}

_SYSTEM = "You are a strict, impartial grader. You respond with JSON only."

_PROMPT = """Decide whether the candidate answer is correct, using the reference answer as ground truth.

Question:
{question}

Reference answer (ground truth):
{reference}

Candidate answer:
{candidate}

Rules:
- Judge the candidate's final answer and key claims against the reference. Ignore wording, length and formatting.
- Any factual error, wrong number, wrong conclusion, or broken code makes it INCORRECT, even if the method looks right.
- PARTIAL only if the final answer is right but a key point from the reference is missing, or the answer is cut off after reaching it.
- CORRECT only if it agrees with the reference on everything that matters.

Respond with JSON: {{"reason": "<one sentence>", "verdict": "CORRECT" | "PARTIAL" | "INCORRECT"}}"""


@dataclass(frozen=True)
class Grade:
    score: float  # 1.0 correct, 0.5 partial, 0.0 incorrect
    verdict: str
    reason: str


def grade(llm: Ollama, judge_model: str, question: str, reference: str, candidate: str) -> Grade:
    completion = llm.chat(
        judge_model,
        _PROMPT.format(question=question, reference=reference, candidate=candidate),
        system=_SYSTEM,
        max_tokens=200,
        json_mode=True,
    )
    return _parse(completion.text)


def _parse(text: str) -> Grade:
    try:
        data = json.loads(text)
        verdict, reason = str(data["verdict"]).strip().upper(), str(data.get("reason", ""))
    except (ValueError, KeyError, TypeError):
        # Scan free text for a verdict word; check INCORRECT first since it contains CORRECT.
        upper = text.upper()
        verdict = next((v for v in ("INCORRECT", "PARTIAL", "CORRECT") if v in upper), "INCORRECT")
        reason = f"parsed from non-JSON judge output: {text[:120]!r}"
    if verdict not in VERDICT_SCORES:
        verdict, reason = "INCORRECT", f"unknown verdict {verdict!r}"
    return Grade(score=VERDICT_SCORES[verdict], verdict=verdict, reason=reason)
