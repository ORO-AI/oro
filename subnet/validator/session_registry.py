"""Validator-owned session isolation for sealed environment tasks."""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NoReturn
from uuid import uuid4

from oro_env_runtime import loop
from oro_env_runtime.loop import record_user_message
from oro_env_runtime.observations import event_observed_or_signaled
from oro_env_runtime.runtime import TOOL_CONTRACT_VERSION, TaskSession
from oro_env_runtime.user_sim import UserSim

from .env_pack_loader import LoadedPack
from .session_errors import (
    AgentInferenceBudgetError,
    HarnessError,
    HarnessExecutionError,
    HarnessTimeoutError,
    InvalidSessionError,
)
from .simulator_completion import InferenceProviderError, SimulatorCompletion

DEFAULT_TOOL_TIMEOUT_S = 10.0
MAX_CALLS_PER_TURN = 16
# The simulator is a provider call, not a local tool. env.openrouter allows 60s
# per attempt, so a shorter budget would quarantine sessions the provider would
# still answer.
DEFAULT_SIMULATOR_TIMEOUT_S = 60.0
_PACK_PROVENANCE_FIELDS = (
    "contract_version",
    "runtime_version",
    "tool_contract_version",
    "verifier_version",
    "result_schema_version",
    "catalog_epoch",
    "catalog_sha256",
    "search_index_epoch",
    "search_index_sha256",
)

# The sandbox receives a deliberately smaller contract than the validator keeps
# for replay and scoring.  Treat this as an allowlist: fields added by the
# runtime stay private until they are explicitly reviewed for agent exposure.
_PUBLIC_POLICY_FIELDS = (
    "query",
    "max_steps",
    "tool_contract_version",
    "tools",
)
_PUBLIC_STEP_FIELDS = ("observation", "done", "error")


def _state_hash(session: TaskSession) -> str:
    return session.env.state_hash_now()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _public_policy_view(
    session: TaskSession, *, max_calls_per_turn: int
) -> dict[str, Any]:
    runtime_view = session.policy_view()
    return {
        **{
            field: copy.deepcopy(runtime_view[field]) for field in _PUBLIC_POLICY_FIELDS
        },
        "max_calls_per_turn": max_calls_per_turn,
    }


def _public_step_result(result: dict[str, Any]) -> dict[str, Any]:
    """Project a runtime result without its validator-only ledger entries."""

    if not isinstance(result, dict):
        raise TypeError("environment step result must be an object")
    observation = result.get("observation")
    if not isinstance(observation, dict):
        raise TypeError("environment observation must be an object")
    return {field: copy.deepcopy(result.get(field)) for field in _PUBLIC_STEP_FIELDS}


def _deliver(
    session: TaskSession,
    transcript: list[dict[str, Any]],
    decisions: list[tuple[dict[str, Any], dict[str, Any] | None]],
    *,
    step: int,
) -> None:
    """The runtime loop's boundary tail: record each shopper decision, reveal the
    facet it disclosed, then say the lines fired timeline transitions queued."""

    for decision, signal in decisions:
        content = str(decision.get("content") or "").strip()
        record_user_message(
            session, step=step, decision={**decision, "content": content}, signal=signal
        )
        if decision.get("disclosed"):
            session.env.disclose_facet(decision["disclosed"])
        transcript.append({"role": "user", "content": content})
    loop.say_utterances(session, transcript, step=step)


def _strip_undeclared_arguments(
    action: dict[str, Any], tool_parameters: dict[str, frozenset[str]]
) -> dict[str, Any]:
    args = action.get("args")
    allowed = tool_parameters.get(action.get("name"))
    if allowed is not None and isinstance(args, dict):
        action["args"] = {key: value for key, value in args.items() if key in allowed}
    return action


@dataclass
class _CachedResponse:
    call_id: str
    action_hash: str
    response: dict[str, Any]


@dataclass
class _SessionState:
    session_id: str
    evaluation_run_id: str
    agent_version_id: str
    task_id: str
    session: TaskSession
    simulator: Any | None
    transcript: list[dict[str, Any]]
    bootstrap: dict[str, Any]
    tool_parameters: dict[str, frozenset[str]]
    started_at: float
    finished_at: float | None = None
    call_trace: list[dict[str, Any]] = field(default_factory=list)
    event_fired_turns: dict[int, int] = field(default_factory=dict)
    surfaced_events: set[int] = field(default_factory=set)
    terminal_reason: str | None = None
    quarantined_reason: str | None = None
    quarantined_outcome: str = "environment_error"
    final_result: dict[str, Any] | None = None
    responses: dict[str, _CachedResponse] = field(default_factory=dict)
    call_ids: dict[str, str] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass
class _CallTrace:
    request: dict[str, Any]
    state_hash_before: str
    started_at: float = field(default_factory=time.perf_counter)
    tool_latency_ms: float | None = None
    simulator_latency_ms: float | None = None
    search_retries: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        state: _SessionState,
        *,
        state_hash_after: str | None = None,
        response: dict[str, Any] | None = None,
        error_type: str | None = None,
        error_detail: str | None = None,
        simulator_exchanges: list[dict[str, Any]] | None = None,
    ) -> None:
        timing = {
            "tool": self.tool_latency_ms,
            "simulator": self.simulator_latency_ms,
            "total": _elapsed_ms(self.started_at),
        }
        state.call_trace.append(
            {
                "request": self.request,
                "response": copy.deepcopy(response),
                "state_hash_before": self.state_hash_before,
                "state_hash_after": state_hash_after or _state_hash(state.session),
                "latency_ms": timing["total"],
                "timing_ms": timing,
                "search_retries": copy.deepcopy(self.search_retries),
                "simulator": (
                    {
                        "latency_ms": self.simulator_latency_ms,
                        "exchanges": simulator_exchanges or [],
                    }
                    if self.simulator_latency_ms is not None
                    else None
                ),
                "error": (
                    {"type": error_type, "detail": error_detail}
                    if error_type is not None
                    else None
                ),
            }
        )


class SessionRegistry:
    """Fresh, bound, idempotent sessions over one validated sealed pack.

    A registry is validator-local and owns its :class:`LoadedPack`. Calls for
    one session are serialized; different sessions may run concurrently
    through the bounded worker pool.
    """

    def __init__(
        self,
        loaded_pack: LoadedPack,
        *,
        tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
        simulator_timeout_s: float = DEFAULT_SIMULATOR_TIMEOUT_S,
        max_calls_per_turn: int = MAX_CALLS_PER_TURN,
        max_workers: int = 8,
        inference_access_token: str | None = None,
        inference_stats_file: str | None = None,
        simulator_proxy_url: str = "http://proxy:80",
        simulator_factory: Callable[[TaskSession], Any] | None = None,
    ) -> None:
        if tool_timeout_s <= 0:
            raise ValueError("tool_timeout_s must be positive")
        if simulator_timeout_s <= 0:
            raise ValueError("simulator_timeout_s must be positive")
        if max_calls_per_turn <= 0:
            raise ValueError("max_calls_per_turn must be positive")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.loaded_pack = loaded_pack
        self.pack_sha256 = loaded_pack.pack_sha256
        self.tool_timeout_s = float(tool_timeout_s)
        self.simulator_timeout_s = float(simulator_timeout_s)
        self.max_calls_per_turn = int(max_calls_per_turn)
        self._inference_access_token = inference_access_token
        self._inference_stats_file = inference_stats_file
        self._simulator_proxy_url = simulator_proxy_url
        self._simulator_factory = simulator_factory
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._sessions: dict[str, _SessionState] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._finalized = False
        self.key_exhausted = threading.Event()

    def _default_simulator(self, state: _SessionState) -> UserSim:
        if self._inference_access_token is None:
            raise RuntimeError(
                "miner inference credentials are required for user simulation"
            )
        completion = SimulatorCompletion(
            self._inference_access_token,
            proxy_url=self._simulator_proxy_url,
            inference_stats_file=self._inference_stats_file,
            episode_id=state.session_id,
        )
        session = state.session
        return UserSim(
            session.task,
            model=session.model_roles["user_simulator"],
            surface_events=not session.state_blind,
            completion=completion,
            fired=lambda: session.env.fired_transitions,
        )

    def _simulator(self, state: _SessionState) -> Any:
        if state.simulator is None:
            state.simulator = (
                self._simulator_factory(state.session)
                if self._simulator_factory is not None
                else self._default_simulator(state)
            )
        return state.simulator

    def _simulator_response(
        self, state: _SessionState, signal: dict[str, Any] | None
    ) -> dict[str, Any]:
        self._simulator(state)
        decision = asyncio.run(state.simulator.respond(state.transcript, signal))
        if not isinstance(decision, dict):
            raise TypeError("simulator response must be an object")
        if signal is not None:
            decision = state.simulator.ensure_react(decision, signal)
        return decision

    @staticmethod
    def _simulator_exchanges(simulator: Any | None) -> list[dict[str, Any]]:
        exporter = getattr(simulator, "exchange_trace", None)
        if not callable(exporter):
            return []
        trace = exporter()
        if not isinstance(trace, list):
            raise TypeError("simulator exchange trace must be a list")
        return copy.deepcopy(trace)

    def _shopper_turn_decisions(
        self,
        state: _SessionState,
        signal: dict[str, Any] | None,
        turn: int,
        message_sent: bool,
    ) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
        """Shared runtime scheduling: the shopper replies to a message or an event notice
        unless a timeline line speaks for it this turn; queued lines are reworded."""
        decisions = []
        lines = state.session.env.utterances
        if signal is not None or (message_sent and not lines):
            decisions.append((self._simulator_response(state, signal), signal))
        if lines:
            asyncio.run(loop.reword_utterances(self._simulator(state), state.session))
        return decisions

    def start(
        self,
        *,
        evaluation_run_id: str,
        agent_version_id: str,
        task_id: str,
        session_id: str | None = None,
        state_blind: bool = False,
    ) -> dict[str, Any]:
        """Create one fresh task state and bind it to an evaluation and agent."""

        if any(
            not isinstance(value, str) or not value
            for value in (evaluation_run_id, agent_version_id, task_id)
        ):
            raise ValueError(
                "evaluation_run_id, agent_version_id, and task_id must be non-empty strings"
            )
        if session_id is None:
            session_id = uuid4().hex
        elif not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")

        with self._lock:
            if self._closed:
                raise RuntimeError("session registry is closed")
            if self._finalized:
                raise RuntimeError("session registry is finalized")
            if session_id in self._sessions:
                raise InvalidSessionError(
                    f"session_id {session_id!r} is already active"
                )
            session = self.loaded_pack.open_session(task_id, state_blind=state_blind)
            bootstrap = {
                "session_id": session_id,
                "policy_view": _public_policy_view(
                    session,
                    max_calls_per_turn=self.max_calls_per_turn,
                ),
            }
            tool_parameters = {}
            for spec in bootstrap["policy_view"]["tools"]:
                function = spec["function"]
                tool_parameters[function["name"]] = frozenset(
                    function["parameters"].get("properties", {})
                )
            state = _SessionState(
                session_id=session_id,
                evaluation_run_id=evaluation_run_id,
                agent_version_id=agent_version_id,
                task_id=task_id,
                session=session,
                simulator=None,
                transcript=[{"role": "user", "content": session.task.goal_text}],
                bootstrap=bootstrap,
                tool_parameters=tool_parameters,
                started_at=time.perf_counter(),
            )
            self._sessions[session_id] = state

        return copy.deepcopy(bootstrap)

    def _state(self, envelope: dict[str, Any]) -> tuple[str, _SessionState]:
        session_id = str(envelope.get("session_id") or "")
        with self._lock:
            state = self._sessions.get(session_id)
        if state is None:
            raise InvalidSessionError("unknown session_id")
        return session_id, state

    def _validate_binding(self, envelope: dict[str, Any], state: _SessionState) -> None:
        # session_id is the opaque bearer binding.  Older clients may still
        # send the private identifiers used by the first protocol revision;
        # reject forged values when present without requiring or disclosing
        # them in new sandbox bootstraps.
        optional_expected = {
            "evaluation_run_id": state.evaluation_run_id,
            "agent_version_id": state.agent_version_id,
            "task_id": state.task_id,
            "pack_sha256": self.pack_sha256,
        }
        for key, value in optional_expected.items():
            if key in envelope and envelope[key] != value:
                raise InvalidSessionError(f"{key} does not match the active session")
        if envelope.get("tool_contract_version") != TOOL_CONTRACT_VERSION:
            raise InvalidSessionError(
                "tool_contract_version does not match the active session"
            )
        if state.quarantined_reason is not None:
            raise InvalidSessionError("session is quarantined")
        if self.key_exhausted.is_set():
            raise AgentInferenceBudgetError("miner inference key exhausted")

    @staticmethod
    def _required_string(envelope: dict[str, Any], key: str) -> str:
        value = envelope.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a non-empty string")
        return value

    def _call_group(
        self, envelope: dict[str, Any], call_id: str
    ) -> tuple[list[dict[str, Any]], bool]:
        action = envelope.get("action")
        calls = envelope.get("calls")
        if action is not None:
            if calls is not None:
                raise ValueError("set action or calls, not both")
            if not isinstance(action, dict):
                raise ValueError("action must be an object")
            return [{"call_id": call_id, "action": copy.deepcopy(action)}], False
        if not isinstance(calls, list) or not calls:
            raise ValueError("calls must be a non-empty list")
        if len(calls) > self.max_calls_per_turn:
            raise ValueError(
                f"calls must contain at most {self.max_calls_per_turn} items"
            )

        call_group = []
        call_ids: set[str] = set()
        for item in calls:
            call_group.append(self._grouped_call(item, call_ids))
        return call_group, True

    @staticmethod
    def _grouped_call(item: Any, call_ids: set[str]) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError("each call must be an object")
        call_id = item.get("call_id")
        action = item.get("action")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("each call_id must be a non-empty string")
        if call_id in call_ids:
            raise ValueError("call ids within one solver turn must be unique")
        if not isinstance(action, dict):
            raise ValueError("each call action must be an object")
        call_ids.add(call_id)
        return {"call_id": call_id, "action": copy.deepcopy(action)}

    def _replay_response(
        self,
        envelope: dict[str, Any],
        state: _SessionState,
        idempotency_key: str,
        call_id: str,
        action_hash: str,
    ) -> dict[str, Any] | None:
        if self._finalized:
            raise InvalidSessionError("session registry is finalized")
        self._validate_binding(envelope, state)
        cached = state.responses.get(idempotency_key)
        if cached is None:
            return None
        if cached.call_id != call_id or cached.action_hash != action_hash:
            raise InvalidSessionError(
                "idempotency key reused with a different call or action"
            )
        replayed = copy.deepcopy(cached.response)
        replayed["replayed"] = True
        return replayed

    @staticmethod
    def _validate_new_call(
        state: _SessionState, call_ids: list[str], turn: int
    ) -> None:
        if state.terminal_reason is not None:
            raise InvalidSessionError("session already reached a terminal observation")
        for call_id in call_ids:
            previous_key = state.call_ids.get(call_id)
            if previous_key is not None:
                raise InvalidSessionError(
                    f"call_id already belongs to idempotency key {previous_key!r}"
                )
        expected_turn = state.session.solver_turn_count + 1
        if turn != expected_turn:
            raise InvalidSessionError(
                "turn does not match session sequence: "
                f"expected {expected_turn}, got {turn}"
            )

    @staticmethod
    def _pending_event(state: _SessionState, turn: int) -> int | None:
        """The first announced event still owed a shopper notice, as loop.run picks it."""
        session = state.session
        if session.state_blind:
            return None
        hard_band = session.task.difficulty_band == "hard"
        if hard_band:
            entries = session.env.ledger.entries()
            state.surfaced_events.update(
                index
                for index in state.event_fired_turns
                if event_observed_or_signaled(
                    entries, None, index=index, budget=session.render_budget
                )
            )
        notice_delay = 2 if hard_band else 1
        return next(
            (
                index
                for index in sorted(state.event_fired_turns)
                if index not in state.surfaced_events
                and turn >= state.event_fired_turns[index] + notice_delay
            ),
            None,
        )

    def _simulator_error(
        self, state: _SessionState, exc: Exception
    ) -> tuple[type[HarnessExecutionError], str]:
        if isinstance(exc, InferenceProviderError) and exc.key_exhausted:
            self.key_exhausted.set()
            state.quarantined_outcome = "agent_error"
            summary = "miner inference key exhausted"
        elif isinstance(exc, InferenceProviderError):
            summary = f"upstream status={exc.status} body={exc.body!r}"
        else:
            summary = type(exc).__name__
        error_type = (
            AgentInferenceBudgetError
            if state.quarantined_outcome == "agent_error"
            else HarnessExecutionError
        )
        return error_type, f"user simulator failed: {summary}"

    def _raise_timeout(
        self,
        session_id: str,
        state: _SessionState,
        trace: _CallTrace,
        *,
        reason: str,
        snapshot: dict[str, Any],
        cause: concurrent.futures.TimeoutError,
    ) -> NoReturn:
        state.quarantined_reason = reason
        trace.record(
            state,
            state_hash_after=snapshot["state_hash"],
            error_type="HarnessTimeoutError",
            error_detail=state.quarantined_reason,
        )
        state.final_result = self._result(
            session_id,
            state,
            outcome="environment_error",
            verdict=None,
            error_detail=state.quarantined_reason,
            snapshot=snapshot,
        )
        raise HarnessTimeoutError(
            f"{state.quarantined_reason}; session quarantined"
        ) from cause

    def _execute_actions(
        self,
        session_id: str,
        state: _SessionState,
        actions: list[dict[str, Any]],
        *,
        is_group: bool,
        trace: _CallTrace,
    ) -> list[dict[str, Any]]:
        started = time.perf_counter()
        snapshot = self._session_snapshot(state)
        step = state.session.step_parallel if is_group else state.session.step
        step_arg = actions if is_group else actions[0]

        def execute() -> Any:
            capture = getattr(state.session.search, "capture_retry_trace", None)
            if not callable(capture):
                return step(step_arg)
            with capture() as retries:
                trace.search_retries = retries
                return step(step_arg)

        future = self._executor.submit(execute)
        try:
            result = future.result(timeout=self.tool_timeout_s)
        except concurrent.futures.TimeoutError as exc:
            trace.tool_latency_ms = _elapsed_ms(started)
            future.cancel()
            self._raise_timeout(
                session_id,
                state,
                trace,
                reason=f"tool call exceeded {self.tool_timeout_s:.3f}s",
                snapshot=snapshot,
                cause=exc,
            )
        except Exception as exc:
            trace.tool_latency_ms = _elapsed_ms(started)
            state.quarantined_reason = f"tool call failed: {type(exc).__name__}"
            trace.record(
                state,
                error_type="HarnessExecutionError",
                error_detail=state.quarantined_reason,
            )
            raise HarnessExecutionError(
                f"{state.quarantined_reason}; session quarantined"
            ) from exc
        trace.tool_latency_ms = _elapsed_ms(started)
        return result if is_group else [result]

    def _in_simulator(
        self,
        session_id: str,
        state: _SessionState,
        trace: _CallTrace,
        work: Callable[..., Any],
        *args: Any,
    ) -> Any:
        """Run simulator work under its timeout; a failure quarantines the session."""
        started = time.perf_counter()
        snapshot = self._session_snapshot(state)
        future = self._executor.submit(work, *args)
        try:
            result = future.result(timeout=self.simulator_timeout_s)
        except concurrent.futures.TimeoutError as exc:
            trace.simulator_latency_ms = _elapsed_ms(started)
            future.cancel()
            self._raise_timeout(
                session_id,
                state,
                trace,
                reason=f"simulator call exceeded {self.simulator_timeout_s:.3f}s",
                snapshot=snapshot,
                cause=exc,
            )
        except Exception as exc:
            trace.simulator_latency_ms = _elapsed_ms(started)
            # Only trusted provider failures may surface status and body;
            # arbitrary simulator messages can contain private task material.
            error_type, state.quarantined_reason = self._simulator_error(state, exc)
            trace.record(
                state,
                error_type=error_type.__name__,
                error_detail=state.quarantined_reason,
                simulator_exchanges=self._simulator_exchanges(state.simulator),
            )
            # The detail stays in the validator's trace and result; the agent
            # gets no provider status or body.
            public = (
                state.quarantined_reason
                if error_type is AgentInferenceBudgetError
                else "user simulator failed"
            )
            raise error_type(f"{public}; session quarantined") from exc
        trace.simulator_latency_ms = (trace.simulator_latency_ms or 0.0) + _elapsed_ms(
            started
        )
        return result

    def _simulate_user_turn(
        self,
        session_id: str,
        state: _SessionState,
        turn: int,
        call_group: list[dict[str, Any]],
        trace: _CallTrace,
    ) -> dict[str, str] | None:
        env = state.session.env
        if env.done():
            return None
        said = len(state.transcript)
        message_sent = any(
            item["action"].get("name") == "message" for item in call_group
        )
        # Preserve an unaided policy turn (two on the hard band) after each
        # public event. The agent must react to the observation before the
        # shopper simulator can reinforce it; a terminal action in that turn
        # gets no rescue. Each announced event gets its own notice, in order.
        pending_event = self._pending_event(state, turn)
        if message_sent or pending_event is not None or env.utterances:
            signal = None
            if pending_event is not None:
                signal = loop._event_signal(env.applied_events[pending_event])
            decisions = self._in_simulator(
                session_id,
                state,
                trace,
                self._shopper_turn_decisions,
                state,
                signal,
                turn,
                message_sent,
            )
            _deliver(state.session, state.transcript, decisions, step=turn)
            if pending_event is not None:
                state.surfaced_events.add(pending_event)
        # The next turn's boundary fires now, so its lines are heard before the agent
        # acts on that turn, as loop.run says them.
        if turn < state.session.max_steps:
            env.begin_solver_turn(turn + 1)
            if env.utterances:
                self._in_simulator(
                    session_id,
                    state,
                    trace,
                    lambda: asyncio.run(
                        loop.reword_utterances(self._simulator(state), state.session)
                    ),
                )
                loop.say_utterances(state.session, state.transcript, step=turn + 1)
        if len(state.transcript) == said:
            return None
        return {
            "content": "\n".join(
                message["content"]
                for message in state.transcript[said:]
                if message["content"]
            )
        }

    def call(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Execute or replay one strictly ordered solver-turn envelope."""

        if not isinstance(envelope, dict):
            raise ValueError("call envelope must be an object")
        session_id, state = self._state(envelope)
        call_id = self._required_string(envelope, "call_id")
        idempotency_key = self._required_string(envelope, "idempotency_key")
        turn = envelope.get("turn")
        if not isinstance(turn, int) or isinstance(turn, bool) or turn < 1:
            raise ValueError("turn must be a positive integer")

        call_group, is_group = self._call_group(envelope, call_id)

        call_ids_to_bind = [call_id]
        call_ids_to_bind.extend(
            item["call_id"] for item in call_group if item["call_id"] != call_id
        )
        action_hash = hashlib.sha256(
            json.dumps(
                call_group,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

        with state.lock:
            replayed = self._replay_response(
                envelope, state, idempotency_key, call_id, action_hash
            )
            if replayed is not None:
                return replayed
            self._validate_new_call(state, call_ids_to_bind, turn)

            # Ignore undeclared top-level arguments for compatibility with
            # agents that relied on the runtime's previously wider surface.
            actions = [
                _strip_undeclared_arguments(item["action"], state.tool_parameters)
                for item in call_group
            ]

            trace = _CallTrace(
                request=copy.deepcopy(envelope),
                state_hash_before=_state_hash(state.session),
            )
            observations = self._execute_actions(
                session_id,
                state,
                actions,
                is_group=is_group,
                trace=trace,
            )

            public_observations = [
                _public_step_result(observation) for observation in observations
            ]
            call_results = [
                {
                    "call_id": item["call_id"],
                    "observation": observation,
                }
                for item, observation in zip(
                    call_group, public_observations, strict=True
                )
            ]
            # A silent market change is left for the agent to see; only an
            # announced one gets a shopper notice.
            for index in state.session.env.announced_events():
                state.event_fired_turns.setdefault(index, turn)
            state.transcript.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item["call_id"],
                            "name": item["action"].get("name"),
                            "args": item["action"].get("args") or {},
                        }
                        for item in call_group
                    ],
                }
            )

            user_message = self._simulate_user_turn(
                session_id,
                state,
                turn,
                call_group,
                trace,
            )

            response = {
                "session_id": session_id,
                "call_id": call_id,
                "turn": turn,
                "solver_turn_count": state.session.solver_turn_count,
                "action_count": state.session.step_count,
                "tool_contract_version": TOOL_CONTRACT_VERSION,
                "calls": call_results,
                "user_message": user_message,
                "provider_status": "complete",
                "environment_error": False,
                "replayed": False,
            }
            if not is_group:
                response["observation"] = public_observations[0]
            state.responses[idempotency_key] = _CachedResponse(
                call_id=call_id,
                action_hash=action_hash,
                response=copy.deepcopy(response),
            )
            for grouped_call_id in call_ids_to_bind:
                state.call_ids[grouped_call_id] = idempotency_key
            if any(observation.get("done") is True for observation in observations):
                state.terminal_reason = "environment_done"
            elif state.session.solver_turn_count >= state.session.max_steps:
                state.terminal_reason = "step_limit"
            trace.record(
                state,
                response=response,
                simulator_exchanges=(
                    self._simulator_exchanges(state.simulator)
                    if trace.simulator_latency_ms is not None
                    else None
                ),
            )
            return response

    def verdict(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Return validator-only deterministic truth for a healthy session."""

        if not isinstance(envelope, dict):
            raise ValueError("verdict envelope must be an object")
        session_id, state = self._state(envelope)
        with state.lock:
            self._validate_binding(envelope, state)
            if state.terminal_reason is None:
                raise InvalidSessionError("session has not reached terminal state")
            result = state.session.verdict()
            return self._result(
                session_id,
                state,
                outcome="completed",
                verdict=result.model_dump(mode="json"),
            )

    def _provenance(self) -> dict[str, Any]:
        return {
            "pack_sha256": self.pack_sha256,
            **{
                key: self.loaded_pack.metadata.get(key)
                for key in _PACK_PROVENANCE_FIELDS
            },
        }

    def _result(
        self,
        session_id: str,
        state: _SessionState,
        *,
        outcome: str,
        verdict: dict[str, Any] | None,
        error_detail: str | None = None,
        snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot = snapshot or self._session_snapshot(state)
        if state.finished_at is None:
            state.finished_at = time.perf_counter()
        has_terminal_state = outcome in {"completed", "partial", "leakage", "exploit"}
        return {
            "evaluation_run_id": state.evaluation_run_id,
            "agent_version_id": state.agent_version_id,
            "task_id": state.task_id,
            "session_id": session_id,
            "pack_sha256": self.pack_sha256,
            "family": snapshot["family"],
            "outcome": outcome,
            "terminal_reason": snapshot["terminal_reason"],
            "error_detail": error_detail,
            "verdict": verdict,
            "terminal_state_hash": (
                snapshot["state_hash"] if has_terminal_state else None
            ),
            "step_count": snapshot["step_count"],
            "wall_seconds": round(max(0.0, state.finished_at - state.started_at), 6),
            "solver_turn_count": snapshot["solver_turn_count"],
            "action_count": snapshot["step_count"],
            "render_budget": snapshot["render_budget"],
            "bootstrap": copy.deepcopy(state.bootstrap),
            "call_trace": copy.deepcopy(state.call_trace),
            "ledger": copy.deepcopy(snapshot["ledger"]),
            "provenance": self._provenance(),
            "environment_error": outcome == "environment_error",
            "verifier_error": outcome == "verifier_error",
        }

    @staticmethod
    def _session_snapshot(state: _SessionState) -> dict[str, Any]:
        """Capture receipt fields before work that may outlive its timeout."""

        return {
            "family": state.session.task.family,
            "terminal_reason": state.terminal_reason,
            "state_hash": _state_hash(state.session),
            "step_count": state.session.step_count,
            "solver_turn_count": state.session.solver_turn_count,
            "render_budget": state.session.render_budget,
            "ledger": [
                entry.model_dump(mode="json")
                for entry in state.session.env.ledger.entries()
            ],
        }

    def _results(
        self,
        states: list[tuple[str, _SessionState]],
        *,
        include_unfinished: bool,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for session_id, state in states:
            with state.lock:
                if state.final_result is not None:
                    results.append(copy.deepcopy(state.final_result))
                    continue
                if state.quarantined_reason is not None:
                    outcome = state.quarantined_outcome
                    verdict = None
                    error_detail = state.quarantined_reason
                elif state.terminal_reason is None:
                    if not include_unfinished:
                        continue
                    outcome = "agent_error"
                    verdict = None
                    error_detail = (
                        "session was not attempted"
                        if not state.call_trace
                        else "agent did not reach a terminal state"
                    )
                else:
                    try:
                        verdict = state.session.verdict().model_dump(mode="json")
                    except Exception as exc:
                        outcome = "verifier_error"
                        verdict = None
                        error_detail = type(exc).__name__
                    else:
                        outcome = "completed"
                        error_detail = None
                state.final_result = self._result(
                    session_id,
                    state,
                    outcome=outcome,
                    verdict=verdict,
                    error_detail=error_detail,
                )
                results.append(copy.deepcopy(state.final_result))
        return results

    def terminal_results(self) -> list[dict[str, Any]]:
        """Snapshot completed or quarantined sessions without sealing the registry."""

        with self._lock:
            states = list(self._sessions.items())
        return self._results(states, include_unfinished=False)

    def finalized_results(self) -> list[dict[str, Any]]:
        """Seal new work, drain active calls, and finalize every session."""

        with self._lock:
            self._finalized = True
            states = list(self._sessions.items())
        return self._results(states, include_unfinished=True)

    def close(self) -> None:
        """Reject new work, quarantine active sessions, and release workers."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            states = list(self._sessions.values())
            self._sessions.clear()
        for state in states:
            with state.lock:
                state.quarantined_reason = "registry closed"
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.loaded_pack.close()

    def __enter__(self) -> SessionRegistry:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = [
    "AgentInferenceBudgetError",
    "DEFAULT_SIMULATOR_TIMEOUT_S",
    "DEFAULT_TOOL_TIMEOUT_S",
    "MAX_CALLS_PER_TURN",
    "HarnessError",
    "HarnessExecutionError",
    "HarnessTimeoutError",
    "InvalidSessionError",
    "SessionRegistry",
]
