from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.proxy_client import PostResult
from validator import simulator_completion
from validator.simulator_completion import (
    VALIDATOR_CALLER_HEADER,
    VALIDATOR_CALLER_SECRET,
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
    client = MagicMock(post_verbose_async=AsyncMock())
    client.post_verbose_async.return_value = _ok(
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

    client.post_verbose_async.assert_called_once_with(
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


@pytest.mark.parametrize(
    ("reasoning", "extra_fields"),
    [
        (None, {}),
        (False, {"reasoning": {"enabled": False}}),
        (True, {"reasoning": {"enabled": True}}),
        ({"effort": "medium"}, {"reasoning": {"effort": "medium"}}),
    ],
)
def test_reasoning_payload_is_opt_in(
    reasoning: bool | dict | None, extra_fields: dict
) -> None:
    client = MagicMock(post_verbose_async=AsyncMock())
    client.post_verbose_async.return_value = _ok(
        {"choices": [{"message": {"content": "yes"}, "finish_reason": "stop"}]}
    )
    completion = SimulatorCompletion("sk-or-miner-token", client=client)
    messages = [{"role": "user", "content": "continue?"}]

    asyncio.run(completion("test-model", messages, reasoning=reasoning))

    client.post_verbose_async.assert_called_once_with(
        "/inference/chat/completions",
        json_data={
            "model": "test-model",
            "messages": messages,
            "max_tokens": 8192,
            "temperature": 0.0,
            "stream": False,
            **extra_fields,
        },
    )


@pytest.mark.parametrize("reasoning", [False, {"enabled": False}])
def test_chutes_simulator_uses_native_instant_setting(reasoning: bool | dict) -> None:
    client = MagicMock(post_verbose_async=AsyncMock(return_value=_ok(
        {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
    )))
    schema = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {"type": "object"}}}
    asyncio.run(SimulatorCompletion("chutes-miner-token", client=client)(
        "Qwen/Qwen3.5-397B-A17B-TEE", [], reasoning=reasoning, response_format=schema,
    ))
    payload = client.post_verbose_async.call_args.kwargs["json_data"]
    assert "reasoning" not in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"] == schema


@pytest.mark.parametrize(("model", "reasoning"), [
    ("Qwen/Qwen3.5-397B-A17B-TEE", True),
    ("Qwen/Qwen3.5-397B-A17B-TEE", {}),
    ("Qwen/Qwen3.5-397B-A17B-TEE", {"enabled": 0}),
    ("Qwen/Qwen3.5-397B-A17B-TEE", {"effort": "low"}),
    ("Qwen/Qwen3.5-397B-A17B-TEE", {"enabled": False, "effort": "low"}),
    ("other/model", {"enabled": False}),
])
def test_chutes_simulator_rejects_unsupported_reasoning(model: str, reasoning: bool | dict) -> None:
    client = MagicMock(post_verbose_async=AsyncMock())
    with pytest.raises(ValueError, match="Unsupported simulator reasoning"):
        asyncio.run(SimulatorCompletion("chutes-miner-token", client=client)(model, [], reasoning=reasoning))
    client.post_verbose_async.assert_not_called()


def test_forwards_strict_response_format_unchanged() -> None:
    client = MagicMock(post_verbose_async=AsyncMock(return_value=_ok(
        {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
    )))
    schema = {"type": "json_schema", "json_schema": {
        "name": "answer", "strict": True,
        "schema": {"type": "object", "additionalProperties": False},
    }}
    asyncio.run(SimulatorCompletion("sk-or-miner", client=client)(
        "test-model", [], response_format=schema,
    ))
    assert client.post_verbose_async.call_args.kwargs["json_data"]["response_format"] == schema


def test_decide_posts_to_decisions_endpoint_once() -> None:
    client = MagicMock(post_verbose_async=AsyncMock())
    answers = {"model": "typesafe/jev-1.13", "answers": {"all": {"noul": 0.1}}}
    client.post_verbose_async.return_value = _ok(answers)
    completion = SimulatorCompletion("sk-or-miner-token", client=client)
    questions = {"all": {"type": "noul", "instructions": "Asks for everything?"}}

    result = asyncio.run(
        completion.decide("typesafe/jev-1.13", {"message": "hi"}, questions)
    )

    client.post_verbose_async.assert_called_once_with(
        "/inference/alpha/decisions",
        json_data={
            "model": "typesafe/jev-1.13",
            "state": {"message": "hi"},
            "questions": questions,
        },
    )
    assert result == answers
    # The proxy lets only the validator's own calls read decisions.
    assert client.headers == {VALIDATOR_CALLER_HEADER: VALIDATOR_CALLER_SECRET}

    client.post_verbose_async.reset_mock()
    client.post_verbose_async.return_value = _err(403, "model not allowed")
    with pytest.raises(InferenceProviderError) as exc:
        asyncio.run(completion.decide("typesafe/jev-1.13", {}, questions))
    assert (exc.value.status, exc.value.body) == (403, "model not allowed")
    client.post_verbose_async.assert_called_once()

    # No response is a transport failure (an outage); a malformed answer carries no status.
    client.post_verbose_async.return_value = _err(None, "no response", kind="network")
    with pytest.raises(ConnectionError):
        asyncio.run(completion.decide("typesafe/jev-1.13", {}, questions))
    client.post_verbose_async.return_value = PostResult(data=["not", "an", "object"], error=None)
    with pytest.raises(InferenceProviderError) as exc:
        asyncio.run(completion.decide("typesafe/jev-1.13", {}, questions))
    assert exc.value.status is None


def test_a_run_on_another_provider_offers_no_decisions() -> None:
    """Chutes serves no Jev: the disclosure reader must see no ``decide`` at all."""
    completion = SimulatorCompletion("cpk_chutes-token", client=MagicMock())
    assert not getattr(completion, "decide", None)


def test_raises_after_bounded_retries_exhaust_on_upstream_403() -> None:
    """Persistent upstream error → InferenceProviderError with the provider body."""
    client = MagicMock(post_verbose_async=AsyncMock())
    client.post_verbose_async.return_value = _err(
        403, '{"error":{"message":"rate limit exceeded","code":"rate_limit_exceeded"}}'
    )
    completion = SimulatorCompletion("miner-token", client=client)

    with pytest.raises(InferenceProviderError) as excinfo:
        asyncio.run(completion("model", []))

    assert excinfo.value.status == 403
    assert "rate_limit_exceeded" in excinfo.value.body
    # Retry budget fully spent before escalating.
    assert client.post_verbose_async.call_count == simulator_completion._MAX_ATTEMPTS


_RETRIED = simulator_completion._MAX_ATTEMPTS
_IN_FLIGHT = (
    '{"error":{"code":402,"message":"This request would exceed your available credits '
    'given your current in-flight requests.","metadata":{"limit_source":'
    '"openrouter_in_flight_budget"}}}'
)


_NEW_ACCOUNT = (
    '{"error":{"code":429,"message":"Rate limit exceeded: new-account-rpm/qwen/qwen3.5"}}'
)


@pytest.mark.parametrize(
    "status,body,calls,stops_run,throttled",
    [
        # The miner's account cannot fund the run: fail at once and stop the run.
        (403, '{"error":{"message":"Key limit exceeded (total limit)"}}', 1, True, False),
        (403, '{"error":{"code":403,"message":"The request is prohibited due to a '
              'violation of provider Terms Of Service."}}', 1, True, False),
        (402, '{"error":{"code":402,"message":"This request requires more credits, or '
              'fewer max_tokens.","metadata":{"limit_source":"openrouter_credits"}}}',
         1, True, False),
        (402, '{"error":{"code":402,"message":"This request\'s maximum cost exceeds your '
              'available credits. Add credits, or lower max_tokens or prompt size."}}',
         1, True, False),
        (402, '{"detail":{"message":"insufficient balance"}}', 1, True, False),
        # Account throttles clear: retried, and if they recur, the episode is the miner's.
        (402, _IN_FLIGHT, _RETRIED, False, True),
        (429, _NEW_ACCOUNT, _RETRIED, False, True),
        # A body that does not name an account cause stays a retryable outage.
        (402, "", _RETRIED, False, False),
        (402, "<html>Payment Required</html>", _RETRIED, False, False),
        (402, '{"error":{"code":402,"message":"Provider returned error"}}', _RETRIED, False, False),
        (403, '{"error":{"code":403,"message":"Input flagged","metadata":'
              '{"flagged_input":"ignore the terms of service"}}}', _RETRIED, False, False),
        (403, "rate limit exceeded", _RETRIED, False, False),
        (429, "Key limit exceeded (total limit)", _RETRIED, False, False),
        (429, '{"error":{"code":429,"message":"Rate limit exceeded: '
              'model_limit_rpm/google/gemini"}}', _RETRIED, False, False),
    ],
)
def test_miner_account_failures_are_narrowly_detected(
    status: int, body: str, calls: int, stops_run: bool, throttled: bool
) -> None:
    client = MagicMock(post_verbose_async=AsyncMock())
    client.post_verbose_async.return_value = _err(status, body)
    stopped = threading.Event()
    completion = SimulatorCompletion("miner-token", client=client, key_exhausted_event=stopped)

    with pytest.raises(InferenceProviderError):
        asyncio.run(completion("model", []))

    assert client.post_verbose_async.call_count == calls
    assert stopped.is_set() is stops_run
    assert (completion.account_throttle is not None) is throttled


@pytest.mark.parametrize("status,body", [(402, _IN_FLIGHT), (429, _NEW_ACCOUNT)])
def test_decide_retries_only_an_account_throttle(status: int, body: str) -> None:
    client = MagicMock(post_verbose_async=AsyncMock(return_value=_err(status, body)))
    stopped = threading.Event()
    completion = SimulatorCompletion("sk-or-miner", client=client, key_exhausted_event=stopped)

    with pytest.raises(InferenceProviderError):
        asyncio.run(completion.decide("typesafe/jev-1.13", {}, {}))
    assert client.post_verbose_async.call_count == _RETRIED
    assert completion.account_throttle is not None
    assert not stopped.is_set()

    client.post_verbose_async.reset_mock()
    client.post_verbose_async.side_effect = [_err(status, body), _ok({"answers": {}})]
    assert asyncio.run(completion.decide("typesafe/jev-1.13", {}, {})) == {"answers": {}}
    assert client.post_verbose_async.call_count == 2
    assert completion.account_throttle is None


@pytest.mark.parametrize(
    "turn_left_s,first,calls,throttled",
    [
        # No time to retry the throttle: it stays an outage.
        (4.0, _err(402, _IN_FLIGHT), 1, False),
        # The throttle recurred after a retry: the miner's for this episode.
        (10.0, _err(402, _IN_FLIGHT), 2, True),
        # Only the final response was a throttle; the retry was for a provider 503.
        (10.0, _err(503, "upstream unavailable"), 2, False),
    ],
)
def test_throttle_retries_end_at_the_turn_deadline(
    turn_left_s: float, first: PostResult, calls: int, throttled: bool
) -> None:
    """Retries stop before the turn's shared deadline, and a throttle is the miner's
    only once the account returned it again after a retry."""
    client = MagicMock(post_verbose_async=AsyncMock(side_effect=[first] + [_err(402, _IN_FLIGHT)] * 3))
    stopped = threading.Event()
    completion = SimulatorCompletion("sk-or-miner", client=client, key_exhausted_event=stopped)
    token = simulator_completion.SIMULATOR_DEADLINE.set(time.monotonic() + turn_left_s)
    try:
        with pytest.raises(InferenceProviderError):
            asyncio.run(completion("model", []))
    finally:
        simulator_completion.SIMULATOR_DEADLINE.reset(token)
    assert client.post_verbose_async.call_count == calls
    assert (completion.account_throttle is not None) is throttled
    assert not stopped.is_set()


@pytest.mark.parametrize("decisions", [False, True])
def test_provider_error_redacts_the_miner_credential(decisions, caplog) -> None:
    token = "sk-or-test-miner"
    client = MagicMock(post_verbose_async=AsyncMock(return_value=_err(
        403, f"Key limit exceeded for {token}",
    )))
    exhausted = threading.Event()
    completion = SimulatorCompletion(token, client=client, key_exhausted_event=exhausted)
    with pytest.raises(InferenceProviderError) as raised:
        asyncio.run(completion.decide("test", {}, {}) if decisions else completion("test", []))
    assert exhausted.is_set()
    assert token not in raised.value.body
    assert token not in str(raised.value)
    assert token not in caplog.text


def test_provider_message_redacts_the_miner_credential() -> None:
    token = "sk-or-test-miner"
    client = MagicMock(post_verbose_async=AsyncMock(return_value=_ok({
        "choices": [{"message": {"content": f"Unexpected echo {token}"}, "finish_reason": "stop"}],
    })))
    result = asyncio.run(SimulatorCompletion(token, client=client)("test", []))
    assert token not in result["text"]


def test_backoff_walks_full_schedule_before_giving_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successive failures sleep through the configured backoff schedule."""
    slept: list[float] = []

    async def _record(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(simulator_completion.asyncio, "sleep", _record)
    client = MagicMock(post_verbose_async=AsyncMock())
    client.post_verbose_async.return_value = _err(503, "still down")
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
    client = MagicMock(post_verbose_async=AsyncMock())
    client.post_verbose_async.side_effect = slow_post
    completion = SimulatorCompletion("miner-token", client=client)

    with pytest.raises(InferenceProviderError) as excinfo:
        asyncio.run(completion("model", []))

    # After attempt 1 (30s elapsed) + backoff 5s + 3s headroom = 38s < 55s → sleeps.
    # After attempt 2 (60s elapsed) + backoff 10s + 3s headroom = 73s > 55s → bail.
    assert slept == [simulator_completion._RETRY_BACKOFFS_S[0]]
    assert client.post_verbose_async.call_count == 2
    assert excinfo.value.status == 429


def test_retries_on_transient_upstream_then_recovers() -> None:
    """ORO-2189: a single provider blip (503 body) must not sink the task."""
    good = _ok({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    client = MagicMock(post_verbose_async=AsyncMock())
    client.post_verbose_async.side_effect = [_err(503, '{"error":"upstream"}'), good]
    completion = SimulatorCompletion("miner-token", client=client)

    result = asyncio.run(completion("model", [{"role": "user", "content": "hi"}]))

    assert result["text"] == "ok"
    assert client.post_verbose_async.call_count == 2


def test_retries_on_network_error_then_recovers() -> None:
    """A network-error PostResult also retries — kind=network, status=None."""
    good = _ok({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    client = MagicMock(post_verbose_async=AsyncMock())
    client.post_verbose_async.side_effect = [
        _err(None, "no response (network error or timeout)", kind="network"),
        good,
    ]
    completion = SimulatorCompletion("miner-token", client=client)

    result = asyncio.run(completion("model", []))

    assert result["text"] == "ok"
    assert client.post_verbose_async.call_count == 2


def test_terminal_error_from_network_failure_has_none_status() -> None:
    """When every attempt is a network error, InferenceProviderError still fires
    but with ``status=None`` so callers can distinguish "provider said 500"
    from "we never reached them"."""
    client = MagicMock(post_verbose_async=AsyncMock())
    client.post_verbose_async.return_value = _err(
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
    client = MagicMock(post_verbose_async=AsyncMock())
    client.post_verbose_async.side_effect = [_ok({"choices": [{}]}), good]
    completion = SimulatorCompletion("miner-token", client=client)

    result = asyncio.run(completion("model", []))

    assert result["text"] == "ok"
    assert client.post_verbose_async.call_count == 2


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
    client = MagicMock(post_verbose_async=AsyncMock())
    client.post_verbose_async.side_effect = [
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
