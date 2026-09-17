"""Session isolation and validator-owned HTTP bridge coverage."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from oro_env_runtime.runtime import TOOL_CONTRACT_VERSION
from oro_env_runtime.schema import Event, LedgerEntry
from validator.env_pack_loader import LoadedPack
from validator.episode_emitter import replay_ledger
from validator.session_registry import (
    HarnessExecutionError,
    HarnessTimeoutError,
    InvalidSessionError,
    SessionRegistry,
)
from validator.session_service import SessionRuntime, create_session_app

pytest_plugins = ("tests.compat_fixture",)


@pytest.fixture
def registry(loaded_pack: LoadedPack) -> SessionRegistry:
    with SessionRegistry(loaded_pack) as value:
        yield value


def _start(
    registry: SessionRegistry,
    *,
    session_id: str = "session-1",
    task_id: str | None = None,
) -> dict:
    return registry.start(
        evaluation_run_id="eval-1",
        agent_version_id="agent-1",
        task_id=task_id or registry.loaded_pack.task_ids[0],
        session_id=session_id,
    )


def _call_envelope(
    registry: SessionRegistry,
    *,
    session_id: str = "session-1",
    task_id: str | None = None,
    call_id: str = "call-1",
    idempotency_key: str = "idem-1",
    turn: int = 1,
    action: dict | None = None,
) -> dict:
    return {
        "evaluation_run_id": "eval-1",
        "agent_version_id": "agent-1",
        "task_id": task_id or registry.loaded_pack.task_ids[0],
        "session_id": session_id,
        "call_id": call_id,
        "idempotency_key": idempotency_key,
        "pack_sha256": registry.pack_sha256,
        "tool_contract_version": TOOL_CONTRACT_VERSION,
        "turn": turn,
        "action": action or {"name": "inspect_cart", "args": {}},
    }


@pytest.mark.parametrize("proxy_url", ["http://proxy:80", "http://127.0.0.1:80"])
def test_default_simulator_uses_the_miner_funded_proxy(
    loaded_pack: LoadedPack,
    proxy_url: str,
    tmp_path,
) -> None:
    stats_file = str(tmp_path / "simulator-inference.jsonl")
    with SessionRegistry(
        loaded_pack,
        inference_access_token="miner-token",
        inference_stats_file=stats_file,
        simulator_proxy_url=proxy_url,
    ) as registry:
        _start(registry)
        state = registry._sessions["session-1"]
        simulator = registry._default_simulator(state)

        assert simulator.model == state.session.model_roles["user_simulator"]
        assert simulator._completion._client.api_key == "miner-token"
        assert simulator._completion._client.proxy_url == proxy_url
        assert simulator._completion._client.inference_stats._problem_id == "session-1"
        assert simulator._completion._client.inference_stats._stats_file == stats_file


def test_default_simulator_evidence_is_private_and_persisted(
    loaded_pack: LoadedPack,
) -> None:
    with SessionRegistry(
        loaded_pack,
        inference_access_token="miner-token",
    ) as registry:
        _start(registry)
        # SimulatorCompletion switched to ``post_verbose`` (ORO-2191);
        # patch the new method with a ``PostResult(data=..., error=None)``
        # equivalent of the old success shape.
        from src.agent.proxy_client import PostResult

        state = registry._sessions["session-1"]
        state.simulator = registry._default_simulator(state)
        state.simulator._completion._client.post_verbose = MagicMock(
            return_value=PostResult(
                data={
                    "choices": [
                        {
                            "message": {
                                "content": '{"action":"clarify","content":"which size?"}'
                            },
                            "finish_reason": "stop",
                        }
                    ]
                },
                error=None,
            )
        )
        response = registry.call(
            _call_envelope(
                registry,
                action={"name": "message", "args": {"content": "Any preference?"}},
            )
        )
        registry.call(
            _call_envelope(
                registry,
                call_id="call-2",
                idempotency_key="idem-2",
                turn=2,
            )
        )
        traces = registry.finalized_results()[0]["call_trace"]

    assert response["user_message"] == {"content": "my budget is at most 200.00 USD."}
    assert "simulator" not in response
    evidence = traces[0]["simulator"]
    assert evidence["latency_ms"] >= 0
    assert len(evidence["exchanges"]) == 1
    exchange = evidence["exchanges"][0]
    assert exchange["model"]
    assert [message["role"] for message in exchange["messages"]] == [
        "system",
        "user",
    ]
    assert exchange["configuration"] == {"max_tokens": 400, "temperature": 0.0}
    assert exchange["response"]["finish_reason"] == "stop"
    assert (
        exchange["response"]["text"] == '{"action":"clarify","content":"which size?"}'
    )
    assert exchange["error"] is None
    assert "miner-token" not in json.dumps(traces)
    assert traces[1]["simulator"] is None


def test_default_simulator_failure_is_an_environment_error(
    loaded_pack: LoadedPack,
) -> None:
    private_detail = "credential-bearing-provider-detail"
    with SessionRegistry(
        loaded_pack,
        inference_access_token="miner-token",
    ) as registry:
        _start(registry)
        # SimulatorCompletion now calls ``post_verbose``; a bug/misuse in the
        # completion client that raises an arbitrary ``RuntimeError`` (as
        # opposed to returning a ``PostResult`` with an error dict) must NOT
        # leak the exception's private message into the ledger. Only the
        # class name is safe to surface — trusted upstream bodies flow
        # through ``InferenceProviderError`` instead (ORO-2191).
        state = registry._sessions["session-1"]
        state.simulator = registry._default_simulator(state)
        state.simulator._completion._client.post_verbose = MagicMock(
            side_effect=RuntimeError(private_detail)
        )
        with pytest.raises(
            HarnessExecutionError, match="user simulator failed: RuntimeError"
        ):
            registry.call(
                _call_envelope(
                    registry,
                    action={
                        "name": "message",
                        "args": {"content": "Any preference?"},
                    },
                )
            )
        result = registry.finalized_results()[0]

    assert result["outcome"] == "environment_error"
    assert result["environment_error"] is True
    assert result["error_detail"] == "user simulator failed: RuntimeError"
    trace = result["call_trace"][0]
    assert trace["error"]["type"] == "HarnessExecutionError"
    assert trace["simulator"]["exchanges"][0]["error"] == {"type": "RuntimeError"}
    assert private_detail not in json.dumps(result)


def test_miner_key_exhaustion_is_agent_error_and_stops_run(
    loaded_pack: LoadedPack,
) -> None:
    from src.agent.proxy_client import PostResult

    with SessionRegistry(loaded_pack, inference_access_token="miner-token") as registry:
        _start(registry)
        state = registry._sessions["session-1"]
        state.simulator = registry._default_simulator(state)
        post = MagicMock(
            return_value=PostResult(
                data=None,
                error={
                    "kind": "upstream",
                    "status": 403,
                    "body": '{"error":{"message":"Key limit exceeded (total limit)"}}',
                },
            )
        )
        state.simulator._completion._client.post_verbose = post
        with pytest.raises(HarnessExecutionError, match="miner inference key exhausted"):
            registry.call(
                _call_envelope(
                    registry,
                    action={"name": "message", "args": {"content": "Any preference?"}},
                )
            )
        _start(registry, session_id="session-2")
        runtime = SessionRuntime()
        runtime.install(registry)
        response = TestClient(create_session_app(runtime)).post(
            "/v1/session/call",
            json=_call_envelope(registry, session_id="session-2"),
        )
        assert response.status_code == 402
        assert response.json()["detail"]["environment_error"] is False
        result = registry.finalized_results()[0]
        assert registry.key_exhausted.is_set()

    assert post.call_count == 1
    assert result["outcome"] == "agent_error"
    assert result["environment_error"] is False
    assert result["error_detail"] == "user simulator failed: miner inference key exhausted"
    assert result["call_trace"][0]["error"]["type"] == "AgentInferenceBudgetError"


def test_default_simulator_rejects_missing_miner_credentials(
    loaded_pack: LoadedPack,
) -> None:
    with SessionRegistry(loaded_pack) as registry:
        _start(registry)
        state = registry._sessions["session-1"]

        with pytest.raises(RuntimeError, match="miner inference credentials"):
            registry._default_simulator(state)


def test_fresh_sessions_are_isolated_and_hide_private_truth(
    registry: SessionRegistry,
) -> None:
    first = _start(registry)
    second = _start(registry, session_id="session-2")
    policy = json.dumps(first["policy_view"])

    assert set(first) == {"session_id", "policy_view"}
    assert set(first["policy_view"]) == {
        "query",
        "max_steps",
        "tool_contract_version",
        "tools",
        "max_calls_per_turn",
    }
    assert first["session_id"] != second["session_id"]
    assert "gold_set" not in policy
    assert "family_payload" not in policy

    first_state_hash = registry._sessions["session-1"].session.env.state_hash_now()
    second_state_hash = registry._sessions["session-2"].session.env.state_hash_now()
    assert first_state_hash == second_state_hash
    task = registry.loaded_pack.task_specs[0]
    candidate = task.gold_set[0]
    changed = registry.call(
        _call_envelope(
            registry,
            action={
                "name": "add_to_cart",
                "args": {
                    "product_id": candidate.product_id,
                    "sku": candidate.sku,
                },
            },
        )
    )
    assert "state_hash" not in changed
    assert (
        registry._sessions["session-1"].session.env.state_hash_now() != first_state_hash
    )
    assert registry._sessions["session-2"].session.env.ledger.entries() == []


def test_session_ids_cannot_be_rebound(registry: SessionRegistry) -> None:
    _start(registry)
    with pytest.raises(InvalidSessionError, match="already active"):
        _start(registry)


def test_forged_session_token_is_rejected(registry: SessionRegistry) -> None:
    _start(registry)
    with pytest.raises(InvalidSessionError, match="unknown session_id"):
        registry.call(_call_envelope(registry, session_id="forged-token"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluation_run_id", "other-eval"),
        ("agent_version_id", "other-agent"),
        ("task_id", "other-task"),
        ("pack_sha256", "0" * 64),
        ("tool_contract_version", "other-tools"),
    ],
)
def test_call_must_match_full_session_binding(
    registry: SessionRegistry, field: str, value: str
) -> None:
    _start(registry)
    envelope = {**_call_envelope(registry), field: value}
    with pytest.raises(InvalidSessionError, match=field):
        registry.call(envelope)


def test_idempotency_is_serialized_and_call_ids_are_bound(
    registry: SessionRegistry,
) -> None:
    _start(registry)
    envelope = _call_envelope(registry)
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(registry.call, [envelope, envelope]))

    assert sorted(response["replayed"] for response in responses) == [False, True]
    assert len(registry._sessions["session-1"].session.env.ledger.entries()) == 2

    # Callers never receive references into the authoritative replay cache.
    for response in responses:
        response["observation"]["observation"]["cart"].append({"forged": True})
    replayed = registry.call(envelope)
    assert replayed["replayed"] is True
    assert replayed["observation"]["observation"]["cart"] == []

    with pytest.raises(InvalidSessionError, match="different call or action"):
        registry.call({**envelope, "call_id": "call-2"})
    with pytest.raises(InvalidSessionError, match="already belongs"):
        registry.call(
            {
                **envelope,
                "idempotency_key": "idem-2",
                "action": {"name": "search", "args": {"query": "x"}},
            }
        )
    assert len(registry._sessions["session-1"].session.env.ledger.entries()) == 2


def test_forged_result_and_ledger_fields_cannot_change_authoritative_state(
    registry: SessionRegistry,
) -> None:
    _start(registry)
    registry._sessions["session-1"].session.max_steps = 1
    canary = "ORO_FORGED_RESULT_CANARY_1868"
    envelope = {
        **_call_envelope(registry),
        "verdict": {"correct": True, "paid_reward": 1.0, "canary": canary},
        "reward": 1.0,
        "ledger": [{"kind": "forged", "canary": canary}],
        "entries": [{"kind": "forged", "canary": canary}],
        "state_hash": canary,
        "terminal_state_hash": canary,
    }

    response = registry.call(envelope)
    result = registry.verdict(envelope)

    assert not {
        "verdict",
        "reward",
        "ledger",
        "entries",
        "state_hash",
        "terminal_state_hash",
    } & set(response)
    assert result["verdict"]["correct"] is False
    assert result["verdict"].get("paid_reward") != 1.0
    assert canary not in json.dumps(result["ledger"])
    # The attempted forgery remains available as untrusted validator-owned
    # attack evidence without becoming environment or verifier authority.
    assert canary in json.dumps(result["call_trace"][0]["request"])


def test_turns_are_strictly_monotonic(registry: SessionRegistry) -> None:
    _start(registry)
    with pytest.raises(InvalidSessionError, match="expected 1, got 2"):
        registry.call(_call_envelope(registry, turn=2))

    registry.call(_call_envelope(registry))
    with pytest.raises(InvalidSessionError, match="expected 2, got 1"):
        registry.call(
            _call_envelope(
                registry,
                call_id="call-2",
                idempotency_key="idem-2",
            )
        )


def test_grouped_calls_keep_order_ids_and_simulator_reply(
    loaded_pack: LoadedPack,
) -> None:
    signals: list[dict | None] = []

    class Simulator:
        async def respond(self, transcript, signal):
            signals.append(signal)
            if signal is not None:
                return {"action": "no_op", "content": "", "reason": "chose_no_op"}
            assert [call["name"] for call in transcript[-1]["tool_calls"]] == [
                "inspect_cart",
                "message",
            ]
            return {
                "action": "clarify",
                "content": "please choose the lowest-priced valid option",
                "reason": "spoke",
            }

        def ensure_react(self, _decision, signal):
            return {
                "action": "react_to_event",
                "content": (
                    f"price changed to {signal['new_price']:.2f} {signal['currency']}"
                ),
                "reason": "forced_event_notice",
            }

    with SessionRegistry(
        loaded_pack,
        simulator_factory=lambda _session: Simulator(),
    ) as registry:
        _start(registry)
        envelope = {
            **_call_envelope(registry),
            "call_id": "solver-turn-1",
            "idempotency_key": "solver-turn-1",
            "calls": [
                {
                    "call_id": "tool-1",
                    "action": {"name": "inspect_cart", "args": {}},
                },
                {
                    "call_id": "tool-2",
                    "action": {
                        "name": "message",
                        "args": {"content": "Should I choose the lowest price?"},
                    },
                },
            ],
        }
        envelope.pop("action")

        response = registry.call(envelope)

        assert [call["call_id"] for call in response["calls"]] == [
            "tool-1",
            "tool-2",
        ]
        timing = registry._sessions["session-1"].call_trace[0]["timing_ms"]
        assert timing["total"] >= timing["tool"] >= 0
        assert timing["total"] >= timing["simulator"] >= 0
        assert response["solver_turn_count"] == 1
        assert response["action_count"] == 2
        assert response["user_message"]["content"].startswith("please choose")
        assert registry.call(envelope)["replayed"] is True
        entries = registry._sessions["session-1"].session.env.ledger.entries()
        assert [entry.kind for entry in entries].count("user_message") == 1

        repeated_inner_id = {
            **envelope,
            "call_id": "solver-turn-2",
            "idempotency_key": "solver-turn-2",
            "turn": 2,
            "calls": [
                {
                    "call_id": "tool-1",
                    "action": {"name": "inspect_cart", "args": {}},
                }
            ],
        }
        with pytest.raises(InvalidSessionError, match="already belongs"):
            registry.call(repeated_inner_id)

        state = registry._sessions["session-1"]
        state.session.env.applied_events.append(
            Event(
                kind="price_change",
                target=state.session.task.gold_set[0],
                old_price=10.0,
                new_price=12.0,
                currency=state.session.task.hard.currency,
            )
        )
        state.event_fired_turn = 1
        event_response = registry.call(
            _call_envelope(
                registry,
                call_id="solver-turn-2",
                idempotency_key="event-turn",
                turn=2,
            )
        )
        expected_signal = {
            "kind": "price_change",
            "old_price": 10.0,
            "new_price": 12.0,
            "currency": state.session.task.hard.currency,
        }
        assert signals[-1] == expected_signal
        assert event_response["user_message"] == {
            "content": f"price changed to 12.00 {state.session.task.hard.currency}"
        }
        assert state.session.env.ledger.last().payload["env_signal"] == expected_signal


def test_shopper_reply_is_delivered_for_every_retained_family(
    loaded_pack: LoadedPack,
) -> None:
    class Simulator:
        async def respond(self, _transcript, _signal):  # noqa: ANN001, ANN201
            return {"action": "clarify", "content": "shopper reply", "reason": "spoke"}

    task_by_family = {
        task.family: task_id
        for task_id, task in zip(
            loaded_pack.task_ids,
            loaded_pack.task_specs,
            strict=True,
        )
    }
    expected_families = {
        family
        for family, count in loaded_pack.manifest["epoch"]["family_counts"].items()
        if count
    }
    assert set(task_by_family) == expected_families

    with SessionRegistry(
        loaded_pack,
        simulator_factory=lambda _session: Simulator(),
    ) as registry:
        for index, (family, task_id) in enumerate(sorted(task_by_family.items())):
            session_id = f"session-{family}"
            _start(registry, session_id=session_id, task_id=task_id)
            response = registry.call(
                _call_envelope(
                    registry,
                    session_id=session_id,
                    task_id=task_id,
                    call_id=f"call-{index}",
                    idempotency_key=f"idem-{index}",
                    action={
                        "name": "message",
                        "args": {"content": "What matters most?"},
                    },
                )
            )

            assert response["user_message"] == {"content": "shopper reply"}
            ledger = registry._sessions[session_id].session.env.ledger.entries()
            assert [entry.kind for entry in ledger].count("user_message") == 1


def test_simulator_timeout_is_independent_of_the_tool_timeout(
    loaded_pack: LoadedPack,
) -> None:
    class SlowSimulator:
        async def respond(self, _transcript, _signal):  # noqa: ANN001, ANN201
            await asyncio.sleep(0.2)
            return {"action": "clarify", "content": "late", "reason": "spoke"}

    def message_envelope(registry: SessionRegistry) -> dict:
        return {
            **_call_envelope(registry),
            "action": {
                "name": "message",
                "args": {"content": "Which option should I choose?"},
            },
        }

    # The provider is slower than the local tool budget and still answers.
    with SessionRegistry(
        loaded_pack,
        tool_timeout_s=0.05,
        simulator_timeout_s=5.0,
        simulator_factory=lambda _session: SlowSimulator(),
    ) as registry:
        _start(registry)
        assert registry.call(message_envelope(registry))["user_message"]["content"]

    # Its own budget still quarantines a provider that never answers.
    with SessionRegistry(
        loaded_pack,
        tool_timeout_s=5.0,
        simulator_timeout_s=0.05,
        simulator_factory=lambda _session: SlowSimulator(),
    ) as registry:
        _start(registry)
        envelope = message_envelope(registry)
        with pytest.raises(HarnessTimeoutError, match="simulator call exceeded"):
            registry.call(envelope)
        timing = registry.finalized_results()[0]["call_trace"][0]["timing_ms"]
        assert timing["tool"] is not None
        assert timing["simulator"] >= 50
        assert timing["total"] >= timing["simulator"]
        with pytest.raises(InvalidSessionError, match="quarantined"):
            registry.verdict(envelope)


def test_grouped_calls_reject_more_than_the_declared_limit(
    loaded_pack: LoadedPack,
) -> None:
    with SessionRegistry(loaded_pack, max_calls_per_turn=2) as registry:
        bootstrap = _start(registry)
        assert bootstrap["policy_view"]["max_calls_per_turn"] == 2
        envelope = _call_envelope(registry)
        envelope.pop("action")
        envelope["calls"] = [
            {
                "call_id": f"tool-{index}",
                "action": {"name": "inspect_cart", "args": {}},
            }
            for index in range(3)
        ]

        with pytest.raises(ValueError, match="at most 2"):
            registry.call(envelope)
        assert registry._sessions["session-1"].session.solver_turn_count == 0


def test_terminal_call_allows_replay_but_rejects_new_turn(
    registry: SessionRegistry,
) -> None:
    _start(registry)
    candidate = registry.loaded_pack.task_specs[0].gold_set[0]
    registry.call(
        _call_envelope(
            registry,
            action={
                "name": "add_to_cart",
                "args": {
                    "product_id": candidate.product_id,
                    "sku": candidate.sku,
                },
            },
        )
    )
    terminal = _call_envelope(
        registry,
        call_id="call-2",
        idempotency_key="idem-2",
        turn=2,
    )
    terminal.pop("action")
    terminal["calls"] = [
        {
            "call_id": "tool-2",
            "action": {"name": "place_test_order", "args": {}},
        },
        {"call_id": "tool-3", "action": {"name": "inspect_cart", "args": {}}},
    ]
    first = registry.call(terminal)
    assert first["calls"][0]["observation"]["done"] is True
    assert first["calls"][1]["observation"]["error"] == "skipped_after_terminal"
    assert registry.call(terminal)["replayed"] is True
    terminal_result = registry.verdict(terminal)
    assert "state_hash" not in first
    assert terminal_result["step_count"] == 2
    assert terminal_result["solver_turn_count"] == 2
    assert terminal_result["action_count"] == 2
    assert terminal_result["family"] == registry.loaded_pack.task_specs[0].family
    assert terminal_result["terminal_reason"] == "environment_done"
    assert (
        terminal_result["ledger"][-1]["state_hash"]
        == terminal_result["terminal_state_hash"]
    )
    assert terminal_result["bootstrap"]["session_id"] == "session-1"
    assert [row["request"]["call_id"] for row in terminal_result["call_trace"]] == [
        "call-1",
        "call-2",
    ]
    assert terminal_result["provenance"]["pack_sha256"] == registry.pack_sha256

    session = registry.loaded_pack.open_session(terminal_result["task_id"])
    replay = replay_ledger(
        ledger=[LedgerEntry.model_validate(row) for row in terminal_result["ledger"]],
        task=session.task,
        catalog=session.catalog,
        expected_terminal_hash=terminal_result["terminal_state_hash"],
        expected_verdict=terminal_result["verdict"],
        render_budget=terminal_result["render_budget"],
    )
    assert replay.terminal_hash_matches is True
    assert replay.verdict_matches is True

    with pytest.raises(InvalidSessionError, match="terminal observation"):
        registry.call(
            _call_envelope(
                registry,
                call_id="call-3",
                idempotency_key="idem-3",
                turn=3,
            )
        )

    # Sealing changes admission only, not the successful receipt or verifier.
    assert json.dumps(registry.finalized_results()[0], sort_keys=True) == json.dumps(
        terminal_result, sort_keys=True
    )


def test_verdict_requires_terminal_state(registry: SessionRegistry) -> None:
    _start(registry)
    with pytest.raises(InvalidSessionError, match="not reached terminal"):
        registry.verdict(_call_envelope(registry))


def test_terminal_results_snapshot_does_not_finalize_registry(
    registry: SessionRegistry,
) -> None:
    _start(registry)
    candidate = registry.loaded_pack.task_specs[0].gold_set[0]
    registry.call(
        _call_envelope(
            registry,
            action={
                "name": "add_to_cart",
                "args": {"product_id": candidate.product_id, "sku": candidate.sku},
            },
        )
    )
    registry.call(
        _call_envelope(
            registry,
            call_id="call-2",
            idempotency_key="idem-2",
            turn=2,
            action={"name": "place_test_order", "args": {}},
        )
    )

    snapshot = registry.terminal_results()

    assert [result["session_id"] for result in snapshot] == ["session-1"]
    snapshot[0]["outcome"] = "tampered"
    assert registry.terminal_results()[0]["outcome"] == "completed"
    _start(registry, session_id="session-2")
    assert [result["session_id"] for result in registry.terminal_results()] == [
        "session-1"
    ]


def test_finalization_rejects_new_sessions_calls_and_cached_replays(
    registry: SessionRegistry,
) -> None:
    _start(registry)
    envelope = _call_envelope(registry)
    registry.call(envelope)
    before = json.dumps(registry.finalized_results(), sort_keys=True)

    with pytest.raises(RuntimeError, match="finalized"):
        _start(registry, session_id="late-session")
    for request in (
        envelope,
        _call_envelope(registry, call_id="late", idempotency_key="late", turn=2),
    ):
        with pytest.raises(InvalidSessionError, match="finalized"):
            registry.call(request)
    assert json.dumps(registry.finalized_results(), sort_keys=True) == before


def test_finalization_drains_inflight_call_and_rejects_queued_replay(
    registry: SessionRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _start(registry)
    entered = threading.Event()
    release = threading.Event()
    queued = threading.Event()
    validate = registry._validate_binding
    state_lookup = registry._state

    def hold_inflight(envelope: dict, state: object) -> None:
        entered.set()
        assert release.wait(5), "test must release the active call"
        validate(envelope, state)

    def lookup(envelope: dict) -> tuple:
        result = state_lookup(envelope)
        if entered.is_set():
            queued.set()
        return result

    monkeypatch.setattr(registry, "_validate_binding", hold_inflight)
    monkeypatch.setattr(registry, "_state", lookup)
    envelope = _call_envelope(registry)
    with ThreadPoolExecutor(max_workers=3) as executor:
        active = executor.submit(registry.call, envelope)
        try:
            assert entered.wait(5)
            replay = executor.submit(registry.call, envelope)
            assert queued.wait(5)
            finalization = executor.submit(registry.finalized_results)
            deadline = time.monotonic() + 5
            while not registry._finalized and time.monotonic() < deadline:
                time.sleep(0.001)
            assert registry._finalized
            assert not finalization.done(), "receipt must wait for active work"
        finally:
            release.set()
        assert active.result(timeout=5)["replayed"] is False
        with pytest.raises(InvalidSessionError, match="finalized"):
            replay.result(timeout=5)
        results = finalization.result(timeout=5)
    assert len(results[0]["call_trace"]) == 1
    assert results[0]["step_count"] == 1
    assert registry.finalized_results() == results


def test_finalization_classifies_unfinished_sessions_as_agent_errors(
    registry: SessionRegistry,
) -> None:
    _start(registry)
    _start(registry, session_id="session-2")
    registry.call(_call_envelope(registry, session_id="session-2"))

    results = {result["session_id"]: result for result in registry.finalized_results()}

    assert results["session-1"]["outcome"] == "agent_error"
    assert results["session-1"]["error_detail"] == "session was not attempted"
    assert results["session-2"]["outcome"] == "agent_error"
    assert (
        results["session-2"]["error_detail"] == "agent did not reach a terminal state"
    )
    assert results["session-2"]["call_trace"][0]["response"]["replayed"] is False


def test_step_limit_freezes_a_serializable_partial_session(
    registry: SessionRegistry,
) -> None:
    _start(registry)
    registry._sessions["session-1"].session.max_steps = 1

    response = registry.call(_call_envelope(registry))
    result = registry.verdict(_call_envelope(registry))

    assert response["observation"]["done"] is False
    assert result["terminal_reason"] == "step_limit"
    assert result["step_count"] == 1
    assert result["ledger"]


def test_timeout_quarantines_session_and_blocks_verdict(
    loaded_pack: LoadedPack, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SessionRegistry(loaded_pack, tool_timeout_s=0.001) as registry:
        _start(registry)
        session = registry._sessions["session-1"].session
        original = session.step

        def slow_step(action: dict) -> dict:
            time.sleep(0.03)
            return original(action)

        monkeypatch.setattr(session, "step", slow_step)
        envelope = _call_envelope(registry)
        with pytest.raises(HarnessTimeoutError, match="session quarantined"):
            registry.call(envelope)
        with pytest.raises(InvalidSessionError, match="quarantined"):
            registry.verdict(envelope)
        result = registry.finalized_results()[0]
        assert result["outcome"] == "environment_error"
        assert result["call_trace"][0]["error"]["type"] == "HarnessTimeoutError"
        timing = result["call_trace"][0]["timing_ms"]
        assert timing["tool"] >= 1
        assert timing["simulator"] is None
        assert result["call_trace"][0]["latency_ms"] == timing["total"]


def test_search_retry_trace_is_private_and_persisted(registry: SessionRegistry) -> None:
    _start(registry)
    session = registry._sessions["session-1"].session
    retry_trace = [
        {
            "attempt": 1,
            "method": "GET",
            "outcome": "retry",
            "error_type": "HTTPError",
            "status_code": 503,
            "backoff_seconds": 0.1,
        },
        {
            "attempt": 2,
            "method": "GET",
            "outcome": "recovered",
            "error_type": None,
            "status_code": 200,
            "backoff_seconds": 0.0,
        },
    ]

    session.search.capture_retry_trace = MagicMock(
        return_value=nullcontext(retry_trace)
    )

    response = registry.call(_call_envelope(registry))
    trace = registry._sessions["session-1"].call_trace[0]

    assert "search_retries" not in response
    assert trace["search_retries"] == retry_trace
    session.search.capture_retry_trace.assert_called_once_with()


def test_timeout_trace_keeps_completed_search_retries(
    loaded_pack: LoadedPack, monkeypatch: pytest.MonkeyPatch
) -> None:
    retry_trace = []
    entered = threading.Event()
    release = threading.Event()
    with SessionRegistry(loaded_pack, tool_timeout_s=0.01) as registry:
        _start(registry)
        session = registry._sessions["session-1"].session
        session.search.capture_retry_trace = MagicMock(
            return_value=nullcontext(retry_trace)
        )

        def blocked_step(_action: dict) -> None:
            retry_trace.append({"attempt": 1, "outcome": "retry"})
            entered.set()
            assert release.wait(5), "test must release the timed-out worker"

        monkeypatch.setattr(session, "step", blocked_step)
        try:
            with pytest.raises(HarnessTimeoutError, match="session quarantined"):
                registry.call(_call_envelope(registry))
            assert entered.is_set()
            trace = registry.finalized_results()[0]["call_trace"][0]
            assert trace["search_retries"] == [{"attempt": 1, "outcome": "retry"}]
        finally:
            release.set()


def test_timeout_finalization_and_close_do_not_wait_for_tool_worker(
    loaded_pack: LoadedPack, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    registry = SessionRegistry(loaded_pack, tool_timeout_s=0.01)
    try:
        _start(registry)
        session = registry._sessions["session-1"].session
        original = session.step

        def blocked_step(action: dict) -> dict:
            entered.set()
            assert release.wait(5), "test must release the timed-out worker"
            try:
                return original(action)
            finally:
                finished.set()

        monkeypatch.setattr(session, "step", blocked_step)
        with pytest.raises(HarnessTimeoutError, match="session quarantined"):
            registry.call(_call_envelope(registry))
        assert entered.is_set()

        with ThreadPoolExecutor(max_workers=1) as executor:
            finalization = executor.submit(registry.finalized_results)
            result = finalization.result(timeout=0.5)[0]

        assert result["outcome"] == "environment_error"
        assert result["step_count"] == 0
        assert result["ledger"] == []
        assert result["terminal_state_hash"] is None
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(registry.close).result(timeout=0.5)
    finally:
        release.set()
        assert finished.wait(5)
        registry.close()


def test_finalized_result_records_stable_monotonic_wall_seconds(
    registry: SessionRegistry,
) -> None:
    _start(registry)
    state = registry._sessions["session-1"]
    state.started_at = time.perf_counter() - 2.5

    first = registry.finalized_results()[0]
    second = registry.finalized_results()[0]

    assert 2.4 < first["wall_seconds"] < 2.6
    assert second["wall_seconds"] == first["wall_seconds"]


def test_http_bridge_exposes_calls_but_not_private_operations(
    registry: SessionRegistry,
) -> None:
    runtime = SessionRuntime()
    client = TestClient(create_session_app(runtime))
    assert client.get("/health").json() == {"status": "ok", "ready": False}
    assert client.post("/v1/session/call", json={}).status_code == 503

    runtime.install(registry)
    _start(registry)
    response = client.post("/v1/session/call", json=_call_envelope(registry))
    assert response.status_code == 200
    assert response.json()["observation"]["observation"] == {"cart": []}
    assert client.post("/v1/session/start", json={}).status_code == 404
    assert client.post("/v1/session/verdict", json={}).status_code == 404


def test_runtime_replacement_closes_the_previous_pack_generation() -> None:
    runtime = SessionRuntime()
    first = MagicMock(spec=SessionRegistry)
    second = MagicMock(spec=SessionRegistry)

    runtime.install(first)
    runtime.install(second)
    runtime.clear(second)

    first.close.assert_called_once_with()
    second.close.assert_called_once_with()
