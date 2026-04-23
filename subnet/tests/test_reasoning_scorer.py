"""Tests for reasoning quality scoring via LLM judge."""

from unittest.mock import patch, MagicMock

from reasoning_scorer import (
    _format_proxy_call,
    _select_models_by_utilization,
    _summarize_proxy_calls,
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

    def test_aggregates_proxy_calls_across_steps(self):
        """Proxy calls are distributed by timestamp across every step; the
        formatter must sum them for the judge, not read only step 0."""
        dialogue = _make_dialogue([("think", []), ("think", []), ("think", [])])
        dialogue[0]["extra_info"]["proxy_calls"] = [
            {"method": "POST", "path": "/inference/chat/completions",
             "params": {"model": "x"}, "status_code": 200, "duration_ms": 5000,
             "completion_tokens": 100}
        ]
        dialogue[1]["extra_info"]["proxy_calls"] = [
            {"method": "POST", "path": "/inference/chat/completions",
             "params": {"model": "x"}, "status_code": 200, "duration_ms": 5000,
             "completion_tokens": 100}
        ]
        dialogue[2]["extra_info"]["proxy_calls"] = [
            {"method": "GET", "path": "/search/find_product",
             "params": {"q": "test"}, "status_code": 200, "duration_ms": 150}
        ]
        text = format_trajectory_for_judge(dialogue)
        # Summary reflects the totals from all three steps, not just step 0
        assert "2 inference" in text
        assert "1 search" in text


class TestParseJudgeResponse:
    def test_parses_json_with_explanation(self):
        resp = parse_judge_response('{"reasoning_quality": 0.85, "explanation": "Good analysis"}')
        assert resp["score"] == 0.85
        assert resp["explanation"] == "Good analysis"
        assert resp["parsed"] is True

    def test_parses_json_without_explanation(self):
        resp = parse_judge_response('{"reasoning_quality": 0.7}')
        assert resp["score"] == 0.7
        assert resp["explanation"] == ""
        assert resp["parsed"] is True

    def test_clamps_above_one(self):
        resp = parse_judge_response('{"reasoning_quality": 1.5}')
        assert resp["score"] == 1.0
        assert resp["parsed"] is True

    def test_clamps_below_zero(self):
        resp = parse_judge_response('{"reasoning_quality": -0.5}')
        assert resp["score"] == 0.0
        assert resp["parsed"] is True

    def test_legitimate_zero_marked_parsed(self):
        """A judge genuinely scoring 0 (regex agent, 0 inference) returns
        valid JSON — callers must distinguish this from an unparseable
        response that also yields score=0."""
        resp = parse_judge_response('{"reasoning_quality": 0, "explanation": "0 inference calls"}')
        assert resp["score"] == 0.0
        assert resp["parsed"] is True

    def test_returns_zero_on_garbage(self):
        resp = parse_judge_response("no score here at all")
        assert resp["score"] == 0.0
        assert resp["parsed"] is False

    def test_returns_zero_on_empty(self):
        resp = parse_judge_response("")
        assert resp["score"] == 0.0
        assert resp["parsed"] is False

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

    @patch("reasoning_scorer.time.sleep")
    @patch("reasoning_scorer.requests.post")
    def test_rotates_on_unparseable_200(self, mock_post, _mock_sleep):
        """A 200 with empty/garbage content must not be surfaced as a
        legitimate 0.0 score — rotate model and retry."""
        mock_post.side_effect = [
            # First judge returns 200 OK but the content is empty
            MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": ""}}]}),
            # Second judge returns a well-formed score
            MagicMock(
                status_code=200,
                json=lambda: {
                    "choices": [{"message": {"content": '{"reasoning_quality": 0.75, "explanation": "ok"}'}}]
                },
            ),
        ]
        result = score_reasoning_quality(REASONING_AGENT, api_key="test-key")
        assert result["score"] == 0.75
        assert result["explanation"] == "ok"
        assert result["inference_failed"] == 1
        assert result["inference_total"] == 2

    @patch("reasoning_scorer.time.sleep")
    @patch("reasoning_scorer.requests.post")
    def test_handles_null_content_from_chutes(self, mock_post, _mock_sleep):
        """Chutes sometimes returns {'choices':[{'message':{'content':null}}]}.
        We must coerce to '' so the unparseable-200 retry path triggers
        without crashing on content[:200]."""
        mock_post.side_effect = [
            # First judge returns 200 OK with content=null (the shape that crashed pre-hotfix)
            MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": None}}]}),
            MagicMock(
                status_code=200,
                json=lambda: {
                    "choices": [{"message": {"content": '{"reasoning_quality": 0.6, "explanation": "ok"}'}}]
                },
            ),
        ]
        result = score_reasoning_quality(REASONING_AGENT, api_key="test-key")
        assert result["score"] == 0.6
        assert result["inference_failed"] == 1
        assert result["inference_total"] == 2

    @patch("reasoning_scorer.time.sleep")
    @patch("reasoning_scorer.requests.post")
    def test_returns_zero_when_all_retries_unparseable(self, mock_post, _mock_sleep):
        """If every rotated judge returns an unparseable 200, fall through
        to the max-retries exit path with score=0 and inference_failed
        counting every attempt — don't surface one of the spurious 0s."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": ""}}]},
        )
        result = score_reasoning_quality(REASONING_AGENT, api_key="test-key", max_retries=3)
        assert result["score"] == 0.0
        assert result["inference_failed"] == 3
        assert result["inference_total"] == 3

    @patch("reasoning_scorer.time.sleep")
    @patch("reasoning_scorer.requests.post")
    def test_keeps_legitimate_zero_without_retry(self, mock_post, _mock_sleep):
        """A well-formed JSON with reasoning_quality: 0 must be returned as-is
        — do NOT rotate models, that's a real regex-agent detection."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": '{"reasoning_quality": 0, "explanation": "0 inference calls"}'}}]
            },
        )
        result = score_reasoning_quality(REGEX_AGENT, api_key="test-key")
        assert result["score"] == 0.0
        assert result["explanation"] == "0 inference calls"
        assert result["inference_failed"] == 0
        assert result["inference_total"] == 1

    def test_empty_dialogue_returns_zero(self):
        result = score_reasoning_quality([], api_key="test-key")
        assert result["score"] == 0.0

    def test_judge_models_nonempty(self):
        assert len(JUDGE_MODELS) >= 1


class TestSelectModelsByUtilization:
    """Tests for _select_models_by_utilization routing logic."""

    def _mock_response(self, entries):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = entries
        return resp

    def test_sorts_by_utilization_ascending(self):
        # Make each JUDGE_MODELS entry have a distinct utilization, reversed
        # from the static order, and verify the returned list is sorted
        # least-loaded first.
        entries = [
            {"name": m, "utilization_current": 0.9 - i * 0.1}
            for i, m in enumerate(JUDGE_MODELS)
        ]
        with patch("reasoning_scorer.requests.get", return_value=self._mock_response(entries)):
            result = _select_models_by_utilization()
        assert result == list(reversed(JUDGE_MODELS))

    def test_missing_model_treated_as_fully_loaded(self):
        # Models not present in the utilization response default to 1.0, so
        # they sort to the end.
        entries = [{"name": JUDGE_MODELS[-1], "utilization_current": 0.1}]
        with patch("reasoning_scorer.requests.get", return_value=self._mock_response(entries)):
            result = _select_models_by_utilization()
        assert result[0] == JUDGE_MODELS[-1]

    def test_api_failure_returns_static_list(self):
        with patch("reasoning_scorer.requests.get", side_effect=Exception("boom")):
            result = _select_models_by_utilization()
        assert result == JUDGE_MODELS

    def test_non_200_returns_static_list(self):
        resp = MagicMock()
        resp.status_code = 500
        with patch("reasoning_scorer.requests.get", return_value=resp):
            result = _select_models_by_utilization()
        assert result == JUDGE_MODELS


class TestFormatProxyCall:
    def test_search_with_params(self):
        call = {
            "method": "GET",
            "path": "/search/find_product",
            "params": {"q": "wireless mouse", "price": "0-25"},
            "status_code": 200,
            "duration_ms": 150,
        }
        result = _format_proxy_call(call)
        assert "GET /search/find_product" in result
        assert "wireless mouse" in result
        assert "200" in result
        assert "150ms" in result

    def test_inference_with_model(self):
        call = {
            "method": "POST",
            "path": "/inference/chat/completions",
            "json_data": {"model": "deepseek-ai/DeepSeek-V3.2-TEE", "temperature": 0},
            "status_code": 200,
            "duration_ms": 2000,
        }
        result = _format_proxy_call(call)
        assert "POST /inference/chat/completions" in result
        assert "model=deepseek-ai/DeepSeek-V3.2-TEE" in result

    def test_inference_with_token_count(self):
        call = {
            "method": "POST",
            "path": "/inference/chat/completions",
            "json_data": {"model": "test-model"},
            "status_code": 200,
            "duration_ms": 5000,
            "response": {"usage": {"completion_tokens": 142, "prompt_tokens": 800}},
        }
        result = _format_proxy_call(call)
        assert "tokens=142" in result

    def test_truncates_long_params(self):
        call = {
            "method": "GET",
            "path": "/search/find_product",
            "params": {"q": "x" * 300},
            "status_code": 200,
            "duration_ms": 100,
        }
        result = _format_proxy_call(call)
        assert "..." in result

    def test_includes_returned_product_ids(self):
        call = {
            "method": "GET",
            "path": "/search/find_product",
            "params": {"q": "laptop"},
            "status_code": 200,
            "duration_ms": 100,
            "result_product_ids": ["123", "456", "789"],
        }
        result = _format_proxy_call(call)
        assert "returned product_ids: 123,456,789" in result

    def test_truncates_long_returned_product_ids_list(self):
        ids = [str(i) for i in range(30)]
        call = {
            "method": "GET",
            "path": "/search/find_product",
            "params": {"q": "laptop"},
            "status_code": 200,
            "duration_ms": 100,
            "result_product_ids": ids,
        }
        result = _format_proxy_call(call)
        assert "returned product_ids: 0,1,2" in result
        assert "(+10 more)" in result


class TestSummarizeProxyCalls:
    def test_empty_list(self):
        result = _summarize_proxy_calls([])
        assert "No proxy call data" in result

    def test_counts_and_shows_calls(self):
        calls = [
            {"method": "GET", "path": "/search/find_product", "params": {"q": "mouse"}, "status_code": 200, "duration_ms": 100},
            {"method": "GET", "path": "/search/find_product", "params": {"q": "keyboard"}, "status_code": 200, "duration_ms": 150},
            {"method": "GET", "path": "/search/view_product_information", "params": {"product_ids": "123"}, "status_code": 200, "duration_ms": 50},
            {"method": "POST", "path": "/inference/chat/completions", "json_data": {"model": "test-model"}, "status_code": 200, "duration_ms": 2000,
             "response": {"usage": {"completion_tokens": 95, "prompt_tokens": 500}}},
        ]
        result = _summarize_proxy_calls(calls)
        assert "2 search" in result
        assert "1 product views" in result
        assert "1 inference" in result
        assert "95 tokens generated" in result
        assert "Call sequence:" in result
        assert "mouse" in result
        assert "keyboard" in result
        assert "model=test-model" in result

    def test_zero_inference_warning(self):
        calls = [
            {"method": "GET", "path": "/search/find_product", "status_code": 200, "duration_ms": 100},
        ]
        result = _summarize_proxy_calls(calls)
        assert "0 inference" in result
        assert "WARNING" in result

    def test_counts_failed_calls(self):
        calls = [
            {"method": "POST", "path": "/inference/chat/completions", "status_code": 402, "duration_ms": 50},
            {"method": "POST", "path": "/inference/chat/completions", "status_code": 200, "duration_ms": 1000},
        ]
        result = _summarize_proxy_calls(calls)
        assert "1 failed" in result


class TestFormatTrajectoryWithProxyCalls:
    def test_includes_proxy_details(self):
        dialogue = [
            {
                "completion": {"message": {"think": "Analyzing.", "tool_call": []}},
                "extra_info": {
                    "query": "find a product",
                    "proxy_calls": [
                        {"method": "GET", "path": "/search/find_product", "params": {"q": "laptop"}, "status_code": 200, "duration_ms": 100},
                        {"method": "POST", "path": "/inference/chat/completions", "json_data": {"model": "test"}, "status_code": 200, "duration_ms": 500},
                    ],
                },
            }
        ]
        text = format_trajectory_for_judge(dialogue)
        assert "VERIFIED PROXY CALLS" in text
        assert "Call sequence:" in text
        assert "laptop" in text
        assert "1 inference" in text

    def test_no_proxy_calls_shows_unavailable(self):
        dialogue = [
            {
                "completion": {"message": {"think": "Thinking.", "tool_call": []}},
                "extra_info": {"query": "test"},
            }
        ]
        text = format_trajectory_for_judge(dialogue)
        assert "No proxy call data" in text
