"""Tests for reasoning quality scoring via LLM judge."""

from unittest.mock import patch, MagicMock

from reasoning_scorer import (
    score_reasoning_quality,
    format_trajectory_for_judge,
    parse_judge_response,
    JUDGE_MODELS,
)


def _make_dialogue(steps):
    """Build a dialogue from a list of (think_text, tool_calls) tuples."""
    dialogue = []
    for think, tools in steps:
        tool_calls = [
            {"name": t, "parameters": {"q": "test"}, "result": {"data": "test"}}
            for t in tools
        ]
        step = {
            "completion": {
                "message": {
                    "think": think,
                    "tool_call": tool_calls,
                }
            },
            "extra_info": {"problem_id": "test-id", "query": "find yellow dishwashing liquid"},
        }
        dialogue.append(step)
    return dialogue


REGEX_AGENT = _make_dialogue([
    ("Processing.", ["find_product"]),
    ("Done.", ["recommend_product", "terminate"]),
])

REASONING_AGENT = _make_dialogue([
    (
        "Task=product. Looking for yellow eco-friendly dishwashing liquid in price range 27-81.",
        ["find_product", "view_product_information"],
    ),
    (
        "Reviewing product attributes. Product 4395270855 has yellow color, eco-friendly, "
        "antibacterial. Best match based on available data.",
        ["recommend_product"],
    ),
    ("Product recommended. Terminating.", ["terminate"]),
])


class TestFormatTrajectoryForJudge:
    def test_formats_think_and_tools(self):
        text = format_trajectory_for_judge(REASONING_AGENT)
        assert "Task=product" in text
        assert "find_product" in text
        assert "view_product_information" in text

    def test_empty_dialogue(self):
        text = format_trajectory_for_judge([])
        assert text == ""

    def test_includes_query(self):
        text = format_trajectory_for_judge(REASONING_AGENT)
        assert "yellow dishwashing liquid" in text


class TestParseJudgeResponse:
    def test_parses_json_with_explanation(self):
        resp = parse_judge_response('{"reasoning_quality": 0.85, "explanation": "Good analysis"}')
        assert resp["score"] == 0.85
        assert resp["explanation"] == "Good analysis"

    def test_parses_json_without_explanation(self):
        resp = parse_judge_response('{"reasoning_quality": 0.7}')
        assert resp["score"] == 0.7
        assert resp["explanation"] == ""

    def test_clamps_above_one(self):
        resp = parse_judge_response('{"reasoning_quality": 1.5}')
        assert resp["score"] == 1.0

    def test_clamps_below_zero(self):
        resp = parse_judge_response('{"reasoning_quality": -0.5}')
        assert resp["score"] == 0.0

    def test_returns_zero_on_garbage(self):
        resp = parse_judge_response("no score here at all")
        assert resp["score"] == 0.0

    def test_returns_zero_on_empty(self):
        resp = parse_judge_response("")
        assert resp["score"] == 0.0

    def test_extracts_json_after_think_block(self):
        """The judge wraps reasoning in <think> tags with numbers like 0.9,
        then outputs JSON. We must use the JSON, not numbers from <think>."""
        response = (
            '<think>\nThe score should be around 0.9 or 1.0. '
            'The verification is weak so maybe 0.5.\n</think>\n\n'
            '{"reasoning_quality": 0.85, "explanation": "Good but shallow"}'
        )
        resp = parse_judge_response(response)
        assert resp["score"] == 0.85

    def test_uses_last_json_match(self):
        """If <think> mentions a JSON-like snippet, use the last one."""
        response = (
            '<think>I initially thought {"reasoning_quality": 0.3} but '
            'reconsidered.</think>\n'
            '{"reasoning_quality": 0.9, "explanation": "Actually good"}'
        )
        resp = parse_judge_response(response)
        assert resp["score"] == 0.9


class TestScoreReasoningQuality:
    @patch("reasoning_scorer.requests.post")
    def test_returns_dict_on_success(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": '{"reasoning_quality": 0.8, "explanation": "Strong reasoning"}'}}]
            },
        )
        result = score_reasoning_quality(REASONING_AGENT, api_key="test-key")
        assert result["score"] == 0.8
        assert result["explanation"] == "Strong reasoning"
        assert result["inference_failed"] == 0
        assert result["inference_total"] == 1

    @patch("reasoning_scorer.time.sleep")
    @patch("reasoning_scorer.requests.post")
    def test_swaps_model_on_429(self, mock_post, _mock_sleep):
        mock_post.side_effect = [
            MagicMock(status_code=429, text="rate limited"),
            MagicMock(
                status_code=200,
                json=lambda: {
                    "choices": [{"message": {"content": '{"reasoning_quality": 0.6, "explanation": "ok"}'}}]
                },
            ),
        ]
        result = score_reasoning_quality(REGEX_AGENT, api_key="test-key")
        assert result["score"] == 0.6
        assert result["inference_failed"] == 1
        assert result["inference_total"] == 2

    @patch("reasoning_scorer.time.sleep")
    @patch("reasoning_scorer.requests.post")
    def test_returns_zero_after_all_retries_exhausted(self, mock_post, _mock_sleep):
        mock_post.return_value = MagicMock(status_code=429, text="rate limited")
        result = score_reasoning_quality(REGEX_AGENT, api_key="test-key", max_retries=3)
        assert result["score"] == 0.0
        assert result["inference_failed"] == 3
        assert result["inference_total"] == 3

    def test_empty_dialogue_returns_zero(self):
        result = score_reasoning_quality([], api_key="test-key")
        assert result["score"] == 0.0

    def test_judge_models_is_nonempty(self):
        assert len(JUDGE_MODELS) >= 3
