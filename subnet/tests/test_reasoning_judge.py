"""ReasoningJudge maps scorer outcomes to problem-level coverage states."""

from unittest.mock import patch

from validator.reasoning_judge import ReasoningJudge


def _judge() -> ReasoningJudge:
    return ReasoningJudge(
        inference_access_token="token",
        inference_provider="openrouter",
        backend_base_url="https://api.example.com",
    )


def _scorer_result(*, valid: bool, failure_class: str | None):
    return {
        "score": 0.9 if valid else 0.0,
        "explanation": "ok" if valid else "",
        "model": "judge" if valid else "",
        "valid": valid,
        "failure_class": failure_class,
        "inference_failed": 0 if valid else 1,
        "inference_total": 1,
        "inference_402": 1 if failure_class == "insufficient_miner_credits" else 0,
        "inference_402_in_flight": 0,
        "inference_402_credits": (
            1 if failure_class == "insufficient_miner_credits" else 0
        ),
        "inference_403": 0,
    }


def test_credit_failure_is_not_a_zero_sample_and_skips_following_calls():
    judge = _judge()
    with patch(
        "src.agent.reasoning_scorer.score_reasoning_quality",
        return_value=_scorer_result(
            valid=False, failure_class="insufficient_miner_credits"
        ),
    ) as scorer:
        first = judge.score([{"step": 1}], "p1")
        second = judge.score([{"step": 2}], "p2")

    assert first["reasoning_score"] is None
    assert first["reasoning_judgment_status"] == "failed"
    assert first["reasoning_failure_class"] == "insufficient_miner_credits"
    assert second["reasoning_score"] is None
    assert second["reasoning_judgment_status"] == "skipped"
    assert second["reasoning_failure_class"] == "insufficient_miner_credits"
    assert scorer.call_count == 1


def test_valid_judgment_is_the_only_path_to_a_reasoning_sample():
    judge = _judge()
    with patch(
        "src.agent.reasoning_scorer.score_reasoning_quality",
        return_value=_scorer_result(valid=True, failure_class=None),
    ):
        result = judge.score([{"step": 1}], "p1")

    assert result["reasoning_score"] == 0.9
    assert result["reasoning_judgment_status"] == "valid"
    assert result["reasoning_failure_class"] is None
