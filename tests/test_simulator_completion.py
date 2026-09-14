from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from src.agent.proxy_client import PostResult
from validator import simulator_completion
from validator.simulator_completion import (
    InferenceProviderError,
    SimulatorCompletion,
)


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry backoff otherwise sleeps ~0.5s+1.0s per test run; skip it."""

    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(simulator_completion.asyncio, "sleep", _instant)


def _ok(data: dict) -> PostResult:
    return PostResult(data=data, error=None)


def _err(status: int | None, body: str, kind: str = "upstream") -> PostResult:
    return PostResult(data=None, error={"kind": kind, "status": status, "body": body})


def test_forwards_user_simulator_request_through_inference_proxy() -> None:
    client = MagicMock()
    client.post_verbose.return_value = _ok(
        {
            "choices": [
                {
                    "message": {"content": '{"action":"no_op","content":""}'},
                    "finish_reason": "stop",
                }
            ]
        }
    )
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

    client.post_verbose.assert_called_once_with(
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


def test_raises_after_bounded_retries_exhaust_on_upstream_403() -> None:
    """Persistent upstream error → InferenceProviderError with the provider body."""
    client = MagicMock()
    client.post_verbose.return_value = _err(
        403, '{"error":{"message":"rate limit exceeded","code":"rate_limit_exceeded"}}'
    )
    completion = SimulatorCompletion("miner-token", client=client)

    with pytest.raises(InferenceProviderError) as excinfo:
        asyncio.run(completion("model", []))

    assert excinfo.value.status == 403
    assert "rate_limit_exceeded" in excinfo.value.body
    # Retry budget fully spent before escalating.
    assert client.post_verbose.call_count == simulator_completion._MAX_ATTEMPTS


def test_backoff_walks_full_schedule_before_giving_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successive failures sleep through the configured backoff schedule."""
    slept: list[float] = []

    async def _record(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(simulator_completion.asyncio, "sleep", _record)
    client = MagicMock()
    client.post_verbose.return_value = _err(503, "still down")
    completion = SimulatorCompletion("miner-token", client=client)

    with pytest.raises(InferenceProviderError):
        asyncio.run(completion("model", []))

    assert slept == list(
        simulator_completion._RETRY_BACKOFFS_S[: simulator_completion._MAX_ATTEMPTS - 1]
    )


def test_bails_out_early_when_wall_budget_would_be_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow first attempt must not let the next backoff push the total
    wall past ``simulator_timeout_s``. The retry loop bails cleanly
    instead of letting the outer future time out mid-sleep and
    quarantine an about-to-recover session."""
    slept: list[float] = []

    async def _record(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(simulator_completion.asyncio, "sleep", _record)
    # Fake ``time.monotonic`` so we can pin elapsed wall without waiting.
    now = [0.0]
    original_post = _err(429, "rate limited")

    def slow_post(*_args, **_kwargs):
        # Every request "takes" 30s of wall time.
        now[0] += 30.0
        return original_post

    monkeypatch.setattr(simulator_completion.time, "monotonic", lambda: now[0])
    client = MagicMock()
    client.post_verbose.side_effect = slow_post
    completion = SimulatorCompletion("miner-token", client=client)

    with pytest.raises(InferenceProviderError) as excinfo:
        asyncio.run(completion("model", []))

    # After attempt 1 (30s elapsed) + backoff 5s + 3s headroom = 38s < 55s → sleeps.
    # After attempt 2 (60s elapsed) + backoff 10s + 3s headroom = 73s > 55s → bail.
    assert slept == [simulator_completion._RETRY_BACKOFFS_S[0]]
    assert client.post_verbose.call_count == 2
    assert excinfo.value.status == 429


def test_retries_on_transient_upstream_then_recovers() -> None:
    """ORO-2189: a single provider blip (503 body) must not sink the task."""
    good = _ok({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    client = MagicMock()
    client.post_verbose.side_effect = [_err(503, '{"error":"upstream"}'), good]
    completion = SimulatorCompletion("miner-token", client=client)

    result = asyncio.run(completion("model", [{"role": "user", "content": "hi"}]))

    assert result["text"] == "ok"
    assert client.post_verbose.call_count == 2


def test_retries_on_network_error_then_recovers() -> None:
    """A network-error PostResult also retries — kind=network, status=None."""
    good = _ok({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    client = MagicMock()
    client.post_verbose.side_effect = [
        _err(None, "no response (network error or timeout)", kind="network"),
        good,
    ]
    completion = SimulatorCompletion("miner-token", client=client)

    result = asyncio.run(completion("model", []))

    assert result["text"] == "ok"
    assert client.post_verbose.call_count == 2


def test_terminal_error_from_network_failure_has_none_status() -> None:
    """When every attempt is a network error, InferenceProviderError still fires
    but with ``status=None`` so callers can distinguish "provider said 500"
    from "we never reached them"."""
    client = MagicMock()
    client.post_verbose.return_value = _err(
        None, "no response (network error or timeout)", kind="network"
    )
    completion = SimulatorCompletion("miner-token", client=client)

    with pytest.raises(InferenceProviderError) as excinfo:
        asyncio.run(completion("model", []))

    assert excinfo.value.status is None
    assert "network" in excinfo.value.body or "no response" in excinfo.value.body


def test_retries_on_malformed_body_then_recovers() -> None:
    """200 with no ``choices[0].message`` (ORO-2191 non-blocker case) — retries."""
    good = _ok({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    client = MagicMock()
    client.post_verbose.side_effect = [_ok({"choices": [{}]}), good]
    completion = SimulatorCompletion("miner-token", client=client)

    result = asyncio.run(completion("model", []))

    assert result["text"] == "ok"
    assert client.post_verbose.call_count == 2


def test_concurrent_sessions_do_not_race_on_error_signal() -> None:
    """The whole reason PostResult replaced last_error (ORO-2191 blocker):
    the failing session's error must be its own, even when another session
    on the same ProxyClient completes between the failing post and the
    caller reading the signal. Return-value semantics make this trivially
    safe; the test just guards against a regression to a shared attribute.
    """
    # Simulate two SimulatorCompletion instances sharing one client but
    # each getting a distinct PostResult on their own call. If a future
    # refactor accidentally reintroduces shared state, the second caller
    # would see the first's error.
    client = MagicMock()
    client.post_verbose.side_effect = [
        *[_err(429, "session-A body")] * simulator_completion._MAX_ATTEMPTS,
        *[_err(500, "session-B body")] * simulator_completion._MAX_ATTEMPTS,
    ]
    completion = SimulatorCompletion("miner-token", client=client)

    with pytest.raises(InferenceProviderError) as a:
        asyncio.run(completion("model", []))
    with pytest.raises(InferenceProviderError) as b:
        asyncio.run(completion("model", []))

    assert a.value.status == 429 and "session-A" in a.value.body
    assert b.value.status == 500 and "session-B" in b.value.body


@pytest.mark.parametrize("setting,expected", [("", None), ("false", False), ("true", True)])
def test_shopper_reasoning_setting_reaches_provider_request(monkeypatch, setting, expected):
    monkeypatch.setenv("ORO_SHOPPER_REASONING_ENABLED", setting)
    client = MagicMock()
    client.post_verbose.return_value = _ok({"choices": [{"message": {"content": '{}'}, "finish_reason": "stop"}]})
    completion = SimulatorCompletion("miner-token", client=client)
    asyncio.run(completion("deepseek/deepseek-v4.1-flash", [], max_tokens=400, temperature=0.0))
    payload = client.post_verbose.call_args.kwargs["json_data"]
    assert payload["model"] == "deepseek/deepseek-v4.1-flash"
    assert payload["max_tokens"] == 400
    assert payload["temperature"] == 0.0
    if expected is None:
        assert "reasoning" not in payload
    else:
        assert payload["reasoning"] == {"enabled": expected}


def test_invalid_shopper_reasoning_setting_fails_before_inference(monkeypatch):
    monkeypatch.setenv("ORO_SHOPPER_REASONING_ENABLED", "disabled")
    client = MagicMock()
    with pytest.raises(ValueError, match="ORO_SHOPPER_REASONING_ENABLED"):
        SimulatorCompletion("miner-token", client=client)
    client.post_verbose.assert_not_called()
