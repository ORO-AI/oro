from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from validator import simulator_completion
from validator.simulator_completion import SimulatorCompletion


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry backoff otherwise sleeps ~0.5s+1.0s per test run; skip it."""

    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(simulator_completion.asyncio, "sleep", _instant)


def test_forwards_user_simulator_request_through_inference_proxy() -> None:
    client = MagicMock()
    client.post.return_value = {
        "choices": [
            {
                "message": {"content": '{"action":"no_op","content":""}'},
                "finish_reason": "stop",
            }
        ]
    }
    completion = SimulatorCompletion("miner-token", client=client)
    messages = [{"role": "user", "content": "continue?"}]

    result = asyncio.run(
        completion(
            "mistralai/mistral-small-2603",
            messages,
            max_tokens=400,
            temperature=0.0,
        )
    )

    client.post.assert_called_once_with(
        "/inference/chat/completions",
        json_data={
            "model": "mistralai/mistral-small-2603",
            "messages": messages,
            "max_tokens": 400,
            "temperature": 0.0,
            "stream": False,
        },
    )
    assert result == {
        "text": '{"action":"no_op","content":""}',
        "tool_calls": [],
        "finish_reason": "stop",
    }


@pytest.mark.parametrize("response", [None, {}, {"choices": []}])
def test_raises_after_bounded_retries_exhaust(response) -> None:  # noqa: ANN001
    """A persistent bad response still raises, but only after retry budget spent."""
    client = MagicMock()
    client.post.return_value = response
    completion = SimulatorCompletion("miner-token", client=client)

    with pytest.raises(RuntimeError, match="returned no completion"):
        asyncio.run(completion("model", []))

    # Every attempt hits the proxy; the retry loop must not short-circuit
    # before the budget is spent.
    assert client.post.call_count == simulator_completion._MAX_ATTEMPTS


def test_retries_on_transient_none_then_recovers() -> None:
    """ProxyClient.post returning None (any non-200) is treated as retriable.

    ORO-2189: prod pattern — miner's OpenRouter/Chutes returns 429/5xx once,
    ProxyClient swallows the non-200 to None, next attempt succeeds. Must
    not sink the whole task into environment_error on a single blip.
    """
    good = {
        "choices": [
            {"message": {"content": "ok"}, "finish_reason": "stop"},
        ]
    }
    client = MagicMock()
    client.post.side_effect = [None, good]
    completion = SimulatorCompletion("miner-token", client=client)

    result = asyncio.run(completion("model", [{"role": "user", "content": "hi"}]))

    assert result["text"] == "ok"
    assert client.post.call_count == 2


def test_retries_on_malformed_body_then_recovers() -> None:
    """A 200 response with no ``choices[0].message`` also retries (provider quirk)."""
    good = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    client = MagicMock()
    client.post.side_effect = [{"choices": [{}]}, good]
    completion = SimulatorCompletion("miner-token", client=client)

    result = asyncio.run(completion("model", []))

    assert result["text"] == "ok"
    assert client.post.call_count == 2
