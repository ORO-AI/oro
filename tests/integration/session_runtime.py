"""Real SessionServer process for sandbox-to-proxy integration coverage."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from typing import Any

from subnet.validator.session_errors import InvalidSessionError
from subnet.validator.session_service import SessionRuntime, SessionServer


class _IntegrationRegistry:
    # Deliberately present in private runtime state. Hostile integration tests
    # assert it never appears in successful or error responses.
    private_canary = "ORO_PRIVATE_RUNTIME_CANARY_1866"

    def __init__(self) -> None:
        self._responses: dict[str, tuple[str, str, dict[str, Any]]] = {}
        self._call_ids: dict[str, str] = {}

    def call(self, envelope: dict[str, Any]) -> dict[str, Any]:
        if envelope.get("session_id") != "integration-session":
            raise InvalidSessionError("unknown session_id")
        if envelope.get("tool_contract_version") != "oro_task_tools_v2":
            raise InvalidSessionError(
                "tool_contract_version does not match the active session"
            )
        call_id = envelope.get("call_id")
        idempotency_key = envelope.get("idempotency_key")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("call_id must be a non-empty string")
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("idempotency_key must be a non-empty string")
        calls = envelope.get("calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError("calls must be a non-empty list")
        action_hash = hashlib.sha256(
            json.dumps(
                calls,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        cached = self._responses.get(idempotency_key)
        if cached is not None:
            cached_call_id, cached_action_hash, cached_response = cached
            if cached_call_id != call_id or cached_action_hash != action_hash:
                raise InvalidSessionError(
                    "idempotency key reused with a different call or action"
                )
            replayed = copy.deepcopy(cached_response)
            replayed["replayed"] = True
            return replayed

        bound_call_ids = [call_id, *[str(call.get("call_id") or "") for call in calls]]
        if any(not value for value in bound_call_ids):
            raise ValueError("each call_id must be a non-empty string")
        if len(bound_call_ids) != len(set(bound_call_ids)):
            raise ValueError("call ids within one solver turn must be unique")
        for bound_call_id in bound_call_ids:
            previous_key = self._call_ids.get(bound_call_id)
            if previous_key is not None:
                raise InvalidSessionError(
                    f"call_id already belongs to idempotency key {previous_key!r}"
                )
        public_calls = [
            {
                "call_id": str(call.get("call_id") or ""),
                "observation": {
                    "observation": {"ok": True},
                    "done": False,
                    "error": None,
                },
            }
            for call in calls
        ]
        response = {
            "session_id": "integration-session",
            "call_id": call_id,
            "turn": int(envelope.get("turn") or 1),
            "solver_turn_count": int(envelope.get("turn") or 1),
            "action_count": len(calls),
            "tool_contract_version": "oro_task_tools_v2",
            "calls": public_calls,
            "user_message": {"content": "continue with the selected option"},
            "provider_status": "complete",
            "environment_error": False,
            "replayed": False,
        }
        self._responses[idempotency_key] = (
            call_id,
            action_hash,
            copy.deepcopy(response),
        )
        for bound_call_id in bound_call_ids:
            self._call_ids[bound_call_id] = idempotency_key
        return response

    def close(self) -> None:
        return None


def main() -> None:
    runtime = SessionRuntime()
    runtime.install(_IntegrationRegistry())  # type: ignore[arg-type]
    server = SessionServer(runtime, host="0.0.0.0", port=9101)
    server.start()
    threading.Event().wait()


if __name__ == "__main__":
    main()
