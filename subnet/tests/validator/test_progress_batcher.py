"""Tests for progress_batcher helpers."""

from uuid import uuid4

from oro_sdk.models import ProblemStatus
from oro_sdk.types import UNSET

from validator.progress_batcher import problem_result_to_update
from validator.types import ProblemResult


def _make_result(**overrides) -> ProblemResult:
    defaults = dict(
        problem_id=str(uuid4()),
        category="product",
        status=ProblemStatus.SUCCESS,
        score=1.0,
        score_dict={},
        inference_failures=0,
        inference_total=0,
    )
    defaults.update(overrides)
    return ProblemResult(**defaults)


class TestProblemResultToUpdate:
    def test_carries_score_and_status(self):
        r = _make_result(score=1.0, status=ProblemStatus.SUCCESS)
        u = problem_result_to_update(r)
        assert u.score == 1.0
        assert u.status == ProblemStatus.SUCCESS

    def test_zero_score_preserved(self):
        r = _make_result(score=0.0, status=ProblemStatus.FAILED)
        u = problem_result_to_update(r)
        assert u.score == 0.0
        assert u.status == ProblemStatus.FAILED

    def test_logs_s3_key_default_unset(self):
        r = _make_result()
        u = problem_result_to_update(r)
        assert u.logs_s3_key is UNSET

    def test_logs_s3_key_passthrough(self):
        r = _make_result()
        u = problem_result_to_update(r, logs_s3_key="k/path/x.json")
        assert u.logs_s3_key == "k/path/x.json"

    def test_reasoning_summary_only_when_scored(self):
        r_no = _make_result(reasoning_score=None)
        assert problem_result_to_update(r_no).score_components_summary is None

        r_yes = _make_result(
            reasoning_score=0.9,
            reasoning_explanation="looks good",
            reasoning_model="anthropic/claude-haiku-4.5",
        )
        u = problem_result_to_update(r_yes)
        assert u.score_components_summary == {
            "reasoning_explanation": "looks good",
            "reasoning_model": "anthropic/claude-haiku-4.5",
        }

    def test_inference_counts_gated_on_total(self):
        r_zero = _make_result(inference_failures=0, inference_total=0)
        u_zero = problem_result_to_update(r_zero)
        assert u_zero.inference_failure_count is None
        assert u_zero.inference_total is None

        r_run = _make_result(inference_failures=1, inference_total=5)
        u_run = problem_result_to_update(r_run)
        assert u_run.inference_failure_count == 1
        assert u_run.inference_total == 5
