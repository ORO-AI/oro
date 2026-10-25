"""Validator-owned world sessions: one runtime ``Session`` per task, bound and idempotent."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

import requests
from oro_env_runtime import models, session as runtime_session
from oro_env_runtime.pack import Pack
from oro_env_runtime.session import Session

from src.agent.proxy_client import ProxyClient

from .session_errors import (
    VALIDATOR_CALLER_HEADER,
    VALIDATOR_CALLER_SECRET,
    AgentInferenceBudgetError,
    HarnessError,
    HarnessExecutionError,
    HarnessTimeoutError,
    InvalidSessionError,
)

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_SERVER_URL = "http://search-server:5632"
# One envelope may carry several tool calls; each runs as its own step.
MAX_ACTIONS_PER_CALL = 16
# Arguments nested deeper than this would overflow recursive JSON handling in the
# runtime and void the episode; they run as a non-object, an invalid-arguments step.
MAX_ARG_DEPTH = 64
TOO_DEEP = "arguments nested too deeply"
# A string with a lone surrogate cannot be canonically encoded and would void the
# episode: such arguments run as an invalid-arguments step, such a name as an unknown tool.
NOT_UTF8 = "arguments are not valid UTF-8"
NOT_UTF8_TOOL = "tool name is not valid UTF-8"
# A verdict for a fault of the validator's machine (wall clock, memory, worker crash),
# never a score: the episode re-runs from the start with the same nonce, and a fault
# that persists is the episode's infrastructure failure.
ENVIRONMENT_FAULT = "environment fault"
MAX_ENVIRONMENT_RERUNS = 2


def _too_deep(value: object) -> bool:
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, (dict, list)):
            if depth > MAX_ARG_DEPTH:
                return True
            stack.extend((child, depth + 1) for child in (item.values() if isinstance(item, dict) else item))
    return False


def _not_utf8(value: object) -> bool:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str):
            try:
                item.encode()
            except UnicodeEncodeError:
                return True
    return False


def _checked_args(args: object) -> object:
    if _too_deep(args):
        return TOO_DEEP
    return NOT_UTF8 if _not_utf8(args) else args


class InferenceProviderError(RuntimeError):
    """A validator model call through the proxy returned no usable result."""

    def __init__(self, status: int | None, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"validator inference failed (status={status} body={body!r})")

    @property
    def key_exhausted(self) -> bool:
        """The run key's limit (403) or the miner account's credit (402) is spent.

        The in-flight reservation 402 clears as other calls finish, so it doesn't count.
        """
        body = self.body.lower()
        if self.status == 403:
            return "key limit exceeded" in body
        return self.status == 402 and (
            "insufficient credits" in body or "requires more credits" in body
        )


class ProxyModelClient:
    """The runtime's model client over the validator proxy, on the miner-funded run key.

    One attempt per call: the runtime's ``live_reader`` owns timeouts and retries,
    and turns a raised failure into an ``"error"`` reading.
    """

    def __init__(
        self,
        access_token: str,
        *,
        proxy_url: str = "http://proxy:80",
        timeout: float = 60.0,
        inference_stats_file: str | None = None,
        key_exhausted_event: threading.Event | None = None,
    ) -> None:
        if not access_token:
            raise ValueError("inference access token is required")
        self._access_token = access_token
        self._key_exhausted_event = key_exhausted_event
        self._client = ProxyClient(
            proxy_url=proxy_url,
            api_key=access_token,
            timeout=timeout,
            max_retries=1,
            inference_stats_file=inference_stats_file,
            inference_stats_problem_id="validator",
        )
        self._client.headers = {VALIDATOR_CALLER_HEADER: VALIDATOR_CALLER_SECRET}

    def complete(self, model: str, prompt: str, params: dict, timeout: float) -> str:
        data = self._post(
            "/inference/chat/completions",
            {**params, "model": model, "messages": [{"role": "user", "content": prompt}], "stream": False},
        )
        choices = data.get("choices")
        message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            raise InferenceProviderError(200, "completion has no text")
        return text.replace(self._access_token, "[REDACTED]")

    def decide(self, model: str, state: object, questions: object, params: dict, timeout: float) -> object:
        # ``questions`` is pack-defined (OpenRouter's keyed dict for Jev): forward it unchanged.
        return self._post(
            "/inference/alpha/decisions",
            {**params, "model": model, "state": state, "questions": questions},
        )

    def _post(self, path: str, body: dict) -> dict:
        result = self._client.post_verbose(path, json_data=body)
        if isinstance(result.data, dict):
            return result.data
        error = result.error or {}
        failure = InferenceProviderError(
            error.get("status"), str(error.get("body") or "").replace(self._access_token, "[REDACTED]")
        )
        if failure.key_exhausted and self._key_exhausted_event is not None:
            self._key_exhausted_event.set()
        raise failure


class SearchServerClient:
    """The runtime's search client over the search server's world routes."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._indexes: dict[str, dict] = {}

    def ensure(self, index: dict) -> None:
        """Upload one verified index archive (``pack.indexes[name]``) under its sha256."""
        with open(index["archive"], "rb") as archive:
            response = requests.put(
                f"{self.base_url}/internal/worlds/{index['id']}",
                data=archive,
                headers={"X-ORO-World-Settings": json.dumps(index["settings"], sort_keys=True, separators=(",", ":"))},
                timeout=self.timeout,
            )
        response.raise_for_status()
        self._indexes[index["id"]] = index

    def search(self, index_id: str, query: str, params: dict, timeout: float) -> list[tuple[str, float]]:
        response = self._post_search(index_id, query, params, timeout)
        if response.status_code == 404 and index_id in self._indexes:  # the server restarted
            self.ensure(self._indexes[index_id])
            response = self._post_search(index_id, query, params, timeout)
        response.raise_for_status()
        return [(hit["id"], float(hit["score"])) for hit in response.json()["hits"]]

    def _post_search(self, index_id: str, query: str, params: dict, timeout: float) -> requests.Response:
        return requests.post(
            f"{self.base_url}/internal/worlds/{index_id}/search",
            json={"query": query, "params": params},
            timeout=timeout,
        )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


@dataclass
class _SessionState:
    session_id: str
    task_id: str
    episode_nonce: str
    session: Session
    call_trace: list[dict[str, Any]] = field(default_factory=list)
    responses: dict[int, tuple[str, dict[str, Any]]] = field(default_factory=dict)
    quarantined_reason: str | None = None
    record: dict[str, Any] | None = None
    retired: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class SessionRegistry:
    """Fresh, bound, idempotent sessions over one verified pack.

    Calls for one session are serialized; different sessions may run
    concurrently on the caller's threads.
    """

    def __init__(
        self,
        pack: Pack,
        *,
        episode_nonces: dict[str, str],
        model_client: models.ModelClient | None = None,
        search_client: models.SearchClient | None = None,
        timeout: float = 60.0,
        retries: int = 2,
        inference_access_token: str | None = None,
        proxy_url: str = "http://proxy:80",
        inference_stats_file: str | None = None,
    ) -> None:
        self.pack = pack
        self.episode_nonces = dict(episode_nonces)
        self.key_exhausted = threading.Event()
        if model_client is None:
            if not inference_access_token:
                raise ValueError("model_client or inference_access_token is required")
            # Resolved from the module at construction so harnesses can wrap it.
            model_client = ProxyModelClient(
                inference_access_token,
                proxy_url=proxy_url,
                timeout=timeout,
                inference_stats_file=inference_stats_file,
                key_exhausted_event=self.key_exhausted,
            )
        if search_client is None:
            search_client = SearchServerClient(
                os.environ.get("SEARCH_SERVER_URL", DEFAULT_SEARCH_SERVER_URL), timeout
            )
        for index in pack.indexes.values():
            search_client.ensure(index)
        self.read = models.live_reader(model_client, search_client, timeout, retries)
        self._sessions: dict[str, _SessionState] = {}
        self._reruns: dict[str, int] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._finalized = False

    def start(self, task_id: str, *, session_id: str | None = None) -> dict[str, Any]:
        """Open one fresh episode for ``task_id`` with the claim's nonce."""

        if task_id not in self.pack.tasks:
            raise KeyError(f"unknown task_id {task_id!r}")
        nonce = self.episode_nonces.get(task_id)
        if not isinstance(nonce, str) or not nonce:
            raise ValueError(f"no episode nonce for task_id {task_id!r}")
        session_id = session_id or uuid4().hex
        with self._lock:
            if self._closed or self._finalized:
                raise RuntimeError("session registry is closed")
            if session_id in self._sessions:
                raise InvalidSessionError(f"session_id {session_id!r} is already active")
            session = Session(self.pack, task_id, nonce, self.read)
            self._sessions[session_id] = _SessionState(session_id, task_id, nonce, session)
        return {
            "session_id": session_id,
            "binding": {"session_id": session_id},
            "policy_view": {
                "query": session.request_text,
                "opening": copy.deepcopy(session.opening),
                "tools": [{"type": "function", "function": tool} for tool in session.tools()],
                "max_steps": self.pack.limits["max_steps"],
                "finished": session.finished,
            },
        }

    def _state(self, envelope: dict[str, Any]) -> _SessionState:
        with self._lock:
            state = self._sessions.get(str(envelope.get("session_id") or ""))
        if state is None:
            raise InvalidSessionError("unknown session_id")
        if "task_id" in envelope and envelope["task_id"] != state.task_id:
            raise InvalidSessionError("task_id does not match the active session")
        return state

    @staticmethod
    def _actions(envelope: dict[str, Any]) -> list[dict[str, Any]]:
        action, actions = envelope.get("action"), envelope.get("actions")
        if (action is None) == (actions is None):
            raise ValueError("set exactly one of action or actions")
        if action is not None:
            actions = [action]
        if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_ACTIONS_PER_CALL:
            raise ValueError(f"actions must be a list of 1 to {MAX_ACTIONS_PER_CALL} items")
        for item in actions:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise ValueError("each action must be an object with a string name")
        return [
            {
                "name": NOT_UTF8_TOOL if _not_utf8(item["name"]) else item["name"],
                "args": _checked_args(item.get("args", {})),
            }
            for item in actions
        ]

    def call(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Run (or replay) one envelope; grouped actions run as consecutive steps."""

        if not isinstance(envelope, dict):
            raise ValueError("call envelope must be an object")
        state = self._state(envelope)
        step = envelope.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise ValueError("step must be a positive integer")
        actions = self._actions(envelope)
        action_hash = hashlib.sha256(
            json.dumps(actions, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        with state.lock:
            if self._finalized:
                raise InvalidSessionError("session registry is finalized")
            if state.quarantined_reason is not None:
                raise InvalidSessionError("session is quarantined")
            if self.key_exhausted.is_set():
                raise AgentInferenceBudgetError("miner inference key exhausted")
            cached = state.responses.get(step)
            if cached is not None:
                if cached[0] != action_hash:
                    raise InvalidSessionError("step reused with a different action")
                return {**copy.deepcopy(cached[1]), "replayed": True}
            session = state.session
            if session.finished:
                raise InvalidSessionError("the episode has finished")
            if step != session.steps + 1:
                raise InvalidSessionError(
                    f"step does not match session sequence: expected {session.steps + 1}, got {step}"
                )
            observations = []
            for action in actions:
                started = time.perf_counter()
                try:
                    result = session.step(action["name"], action["args"])
                except Exception as exc:
                    state.quarantined_reason = f"runtime step failed: {type(exc).__name__}"
                    state.call_trace.append({"tool": action["name"], "step": session.steps,
                                             "elapsed_ms": _elapsed_ms(started), "error": state.quarantined_reason})
                    raise HarnessExecutionError(f"{state.quarantined_reason}; session quarantined") from exc
                state.call_trace.append({"tool": action["name"], "step": session.steps, "elapsed_ms": _elapsed_ms(started)})
                observations.append(result.observation)
                if result.finished:
                    break
            response: dict[str, Any] = {"observation": observations[-1], "finished": session.finished, "step": session.steps}
            if "actions" in envelope:
                response["observations"] = observations
            state.responses[step] = (action_hash, copy.deepcopy(response))
            return {**response, "replayed": False}

    def finish_all(self) -> None:
        """Stop every unfinished, healthy episode (a runtime-forced finish) and seal new work."""

        with self._lock:
            self._finalized = True
            states = list(self._sessions.values())
        for state in states:
            with state.lock:
                if not state.session.finished and state.quarantined_reason is None:
                    state.session.finish()

    def _record(self, state: _SessionState) -> dict[str, Any]:
        """Grade a finished episode once; a quarantined one gets no verdict."""

        if state.record is None:
            ledger = state.session.ledger()
            verdict = None
            if state.quarantined_reason is None:
                verdict = runtime_session.grade_episode(self.pack, ledger, self.read)
            state.record = {
                "task_id": state.task_id,
                "session_id": state.session_id,
                "episode_nonce": state.episode_nonce,
                "ledger": ledger,
                "verdict": verdict,
                "error": state.quarantined_reason,
                "call_trace": state.call_trace,
                "steps": state.session.steps,
            }
        return copy.deepcopy(state.record)

    def _terminal(self) -> list[tuple[_SessionState, dict[str, Any]]]:
        with self._lock:
            states = [state for state in self._sessions.values() if not state.retired]
        terminal = []
        for state in states:
            with state.lock:
                if state.session.finished or state.quarantined_reason is not None:
                    terminal.append((state, self._record(state)))
        return terminal

    def _may_rerun(self, record: dict[str, Any]) -> bool:
        verdict = record["verdict"]
        return (
            verdict is not None
            and verdict["category"] == ENVIRONMENT_FAULT
            and self._reruns.get(record["task_id"], 0) < MAX_ENVIRONMENT_RERUNS
        )

    def terminal_results(self, *, final: bool = False) -> list[dict[str, Any]]:
        """Records of every finished or quarantined episode, graded on first request.

        An environment fault awaiting its re-run is held back until ``final``; a
        fault that persisted, or that ends the run without its re-run, is unscored:
        no verdict and ``unscored`` "environment".
        """

        results = []
        for _state, record in self._terminal():
            verdict = record["verdict"]
            if verdict is not None and verdict["category"] == ENVIRONMENT_FAULT:
                if self._may_rerun(record) and not final:
                    continue
                reruns = self._reruns.get(record["task_id"], 0)
                record.update(verdict=None, unscored="environment",
                              error=f"environment fault persisted after {reruns} re-runs")
            results.append(record)
        return results

    def environment_faults(self) -> list[str]:
        """Tasks whose latest episode ended in an environment fault and may re-run."""

        return [record["task_id"] for _state, record in self._terminal() if self._may_rerun(record)]

    def restart(self, task_id: str) -> dict[str, Any]:
        """Retire the task's faulted episode and open a fresh one with the same nonce."""

        faulted = [state for state, record in self._terminal() if record["task_id"] == task_id and self._may_rerun(record)]
        if not faulted:
            raise InvalidSessionError(f"task {task_id!r} has no environment fault to re-run")
        with self._lock:
            faulted[0].retired = True
            self._reruns[task_id] = self._reruns.get(task_id, 0) + 1
            self._finalized = False
        return self.start(task_id)

    def rerun_environment_faults(self, run: Callable[[list[dict[str, Any]]], object]) -> None:
        """Re-run every environment-faulted episode with the agent until none may re-run.

        ``run`` drives the agent over the new sessions' bootstraps.
        """

        while faulted := self.environment_faults():
            logger.warning("Re-running %d episodes after an environment fault", len(faulted))
            run([self.restart(task_id) for task_id in faulted])
            self.finish_all()

    def close(self) -> None:
        """Reject new work and stop every episode's sandbox worker."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            states = list(self._sessions.values())
            self._sessions.clear()
        for state in states:
            with state.lock:
                if state.session.box is not None:
                    state.session.box.close()

    def __enter__(self) -> SessionRegistry:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = [
    "AgentInferenceBudgetError",
    "HarnessError",
    "HarnessExecutionError",
    "HarnessTimeoutError",
    "InferenceProviderError",
    "InvalidSessionError",
    "MAX_ACTIONS_PER_CALL",
    "ProxyModelClient",
    "SearchServerClient",
    "SessionRegistry",
    "VALIDATOR_CALLER_HEADER",
    "VALIDATOR_CALLER_SECRET",
]
