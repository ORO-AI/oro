"""Tests for timestamp penalty in format scoring."""

import pytest
from unittest.mock import patch
from src.agent.problem_scorer import ProblemScorer, TIMESTAMP_MISSING_PENALTY

PENALTY = TIMESTAMP_MISSING_PENALTY


def _make_step(timestamp=None):
    """Build a minimal step dict with valid format tags."""
    step = {
        "completion": {
            "message": {
                "think": "reasoning",
                "tool_call": [{"name": "find_product", "parameters": {"q": "test"}}],
                "response": "answer",
            },
        },
        "extra_info": {"step": 1},
    }
    if timestamp is not None:
        step["extra_info"]["timestamp"] = timestamp
    return step


def _score_format(steps):
    """Score format for a list of steps, mocking out product lookup."""
    with patch("src.agent.problem_scorer.get_product", return_value=None):
        scorer = ProblemScorer("product", {"q": {}}, {})
        return scorer.score_problem("q", steps)["format"]


@pytest.mark.parametrize("timestamp,expected", [
    (1711234567000, 1.0),
    (1711234567.0, 1.0),
    (None, PENALTY),
    ("not a number", PENALTY),
    (0, PENALTY),
    (-1, PENALTY),
], ids=["valid_int", "valid_float", "missing", "wrong_type", "zero", "negative"])
def test_single_step_timestamp_scoring(timestamp, expected):
    assert _score_format([_make_step(timestamp=timestamp)]) == expected


def test_mixed_steps_average_correctly():
    steps = [
        _make_step(timestamp=1711234567000),
        _make_step(timestamp=None),
    ]
    assert _score_format(steps) == (1.0 + PENALTY) / 2
