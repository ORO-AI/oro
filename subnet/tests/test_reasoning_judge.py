"""ReasoningJudge must not turn a failed judge call into a zero score."""

from unittest.mock import patch

from validator.reasoning_judge import ReasoningJudge


def test_credit_failure_is_missing_and_opens_the_circuit():
    judge = ReasoningJudge("token", "openrouter", "https://api.example.com")
    failure = {
        "score": 0.0,
        "explanation": "",
        "model": "",
        "inference_failed": 1,
        "inference_total": 1,
        "inference_402": 1,
    }
    with patch(
        "src.agent.reasoning_scorer.score_reasoning_quality",
        return_value=failure,
    ) as scorer:
        first = judge.score([{"step": 1}], "p1")
        second = judge.score([{"step": 2}], "p2")

    assert first["reasoning_score"] is None
    assert second["reasoning_score"] is None
    assert scorer.call_count == 1
