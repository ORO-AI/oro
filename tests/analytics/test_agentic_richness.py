"""Unit tests for the agentic_richness calculator (ORO-1372).

Coverage:
  * axis-A default-agent XML pattern -> 1.0
  * axis-B no-tool-call pattern -> 0.0
  * multi-tool <tool_call> block, both matched
  * theatrical <tool_call> (next dispatch mismatches) -> not credited
  * stale targets from earlier step do not credit later dispatches
  * native message.tool_calls path (post-ORO-1162)
  * bundle-level aggregation across steps
  * empty bundle
  * unrecognised /search/* path stays out of the denominator
"""

from __future__ import annotations

from src.analytics.agentic_richness import (
    calc_agentic_richness,
    calc_agentic_richness_for_step,
)


def _inference(content: str = "", tool_calls: list[dict] | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content or None}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "kind": "summary",
        "path": "/inference/chat/completions",
        "response": {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]},
    }


def _dispatch(path: str) -> dict:
    return {"kind": "summary", "path": path}


def _counts(step: dict) -> tuple[int, int]:
    """Drop the tool_breakdown from the per-step return so tests can compare tuples."""
    llm, disp, _ = calc_agentic_richness_for_step(step)
    return llm, disp


def test_axis_a_single_xml_tool_call() -> None:
    step = {
        "extra_info": {
            "proxy_calls": [
                _inference('<think>x</think><tool_call>[{"name":"find_product","parameters":{}}]</tool_call>'),
                _dispatch("/search/find_product"),
            ]
        }
    }
    assert _counts(step) == (1, 1)


def test_axis_b_no_tool_call() -> None:
    step = {
        "extra_info": {
            "proxy_calls": [
                _inference("<think>narrate</think><response>shop</response>"),
                _dispatch("/search/find_product"),
                _dispatch("/search/view_product_information"),
            ]
        }
    }
    assert _counts(step) == (0, 2)


def test_multi_tool_block_both_matched() -> None:
    step = {
        "extra_info": {
            "proxy_calls": [
                _inference(
                    '<tool_call>[{"name":"find_product","parameters":{}},'
                    '{"name":"view_product_information","parameters":{}}]</tool_call>'
                ),
                _dispatch("/search/find_product"),
                _dispatch("/search/view_product_information"),
            ]
        }
    }
    assert _counts(step) == (2, 2)


def test_theatrical_tool_call_rejected() -> None:
    """LLM emits find_product but Python dispatches view first."""
    step = {
        "extra_info": {
            "proxy_calls": [
                _inference('<tool_call>[{"name":"find_product","parameters":{}}]</tool_call>'),
                _dispatch("/search/view_product_information"),  # mismatch eats the target
                _dispatch("/search/find_product"),              # too late to credit
            ]
        }
    }
    assert _counts(step) == (0, 2)


def test_stale_target_does_not_credit_after_new_inference() -> None:
    step = {
        "extra_info": {
            "proxy_calls": [
                _inference(
                    '<tool_call>[{"name":"find_product","parameters":{}},'
                    '{"name":"check_product_match","parameters":{}}]</tool_call>'
                ),
                _dispatch("/search/find_product"),               # 1/1
                _inference("<think>narrate</think><response>r</response>"),  # resets queue
                _dispatch("/search/check_product_match"),        # not credited
            ]
        }
    }
    assert _counts(step) == (1, 2)


def test_native_tool_calls_credited() -> None:
    step = {
        "extra_info": {
            "proxy_calls": [
                _inference(
                    "",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "find_product", "arguments": "{}"},
                        }
                    ],
                ),
                _dispatch("/search/find_product"),
            ]
        }
    }
    assert _counts(step) == (1, 1)


def test_bundle_aggregation() -> None:
    bundle = [
        {
            "extra_info": {
                "proxy_calls": [
                    _inference('<tool_call>[{"name":"find_product","parameters":{}}]</tool_call>'),
                    _dispatch("/search/find_product"),
                ]
            }
        },
        {
            "extra_info": {
                "proxy_calls": [
                    _inference("<think>x</think><response>y</response>"),
                    _dispatch("/search/find_product"),
                    _dispatch("/search/view_product_information"),
                ]
            }
        },
    ]
    result = calc_agentic_richness(bundle)
    assert result.llm_emitted_count == 1
    assert result.total_dispatch_count == 3
    assert abs(result.agentic_richness - 1 / 3) < 1e-9
    assert result.n_steps == 2
    assert result.tool_breakdown == {"find_product": 2, "view_product_information": 1}


def test_empty_bundle_returns_zero() -> None:
    result = calc_agentic_richness([])
    assert result.agentic_richness == 0.0
    assert result.total_dispatch_count == 0
    assert result.n_steps == 0


def test_unknown_proxy_path_ignored() -> None:
    """An unrecognised /search/* path doesn't pollute the denominator."""
    step = {
        "extra_info": {
            "proxy_calls": [
                _inference("<think>x</think>"),
                _dispatch("/search/some_new_endpoint"),
                _dispatch("/search/find_product"),
            ]
        }
    }
    assert _counts(step) == (0, 1)
