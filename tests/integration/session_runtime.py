"""Private session server for sandbox-to-proxy integration coverage."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import uvicorn

from subnet.validator.session_errors import InvalidSessionError
from subnet.validator.session_service import SessionRuntime, create_session_app


class _IntegrationRegistry:
    # Deliberately present in private runtime state. Hostile integration tests
    # assert it never appears in successful or error responses.
    private_canary = "ORO_PRIVATE_RUNTIME_CANARY_1866"

    def __init__(self) -> None:
        # Per session: {step: (action hash, response)}, the last step and whether it finished. Each test opens
        # its own "integration-..." session; any other id is unknown.
        self._sessions: dict[str, dict[str, Any]] = {}

    def call(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """The contract-8 call shape and rules of ``SessionRegistry.call``: one action (or a list) per step,
        steps in sequence, a step replayed only with the same action, and an order or decline finishing."""
        if not isinstance(envelope, dict):
            raise ValueError("call envelope must be an object")
        session_id = envelope.get("session_id")
        if not isinstance(session_id, str) or not session_id.startswith("integration-"):
            raise InvalidSessionError("unknown session_id")
        state = self._sessions.setdefault(session_id, {"responses": {}, "steps": 0, "finished": False})
        step = envelope.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise ValueError("step must be a positive integer")
        action, actions = envelope.get("action"), envelope.get("actions")
        if (action is None) == (actions is None):
            raise ValueError("set exactly one of action or actions")
        actions = [action] if action is not None else actions
        if not isinstance(actions, list) or not actions or not all(
            isinstance(a, dict) and isinstance(a.get("name"), str) for a in actions
        ):
            raise ValueError("each action must be an object with a string name")
        action_hash = hashlib.sha256(
            json.dumps(actions, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        cached = state["responses"].get(step)
        if cached is not None:
            if cached[0] != action_hash:
                raise InvalidSessionError("step reused with a different action")
            return {**copy.deepcopy(cached[1]), "replayed": True}
        if state["finished"]:
            raise InvalidSessionError("the episode has finished")
        if step != state["steps"] + 1:
            raise InvalidSessionError(f"step does not match session sequence: expected {state['steps'] + 1}")
        observations = []
        for item in actions:
            state["steps"] += 1
            observations.append({"ok": True, "tool": item["name"]})
            if item["name"] in ("order", "decline"):
                state["finished"] = True
                break
        response: dict[str, Any] = {"observation": observations[-1], "finished": state["finished"], "step": state["steps"]}
        if envelope.get("actions") is not None:
            response["observations"] = observations
        state["responses"][step] = (action_hash, copy.deepcopy(response))
        return {**response, "replayed": False}

    def close(self) -> None:
        return None


def main() -> None:
    runtime = SessionRuntime()
    runtime.install(_IntegrationRegistry())  # type: ignore[arg-type]
    app = create_session_app(runtime)

    @app.put("/_test/inference-grant")
    def set_grant(grant: dict[str, Any]) -> None:
        if grant:
            runtime.set_inference_grant(
                str(grant["run_id"]), str(grant["token"]), float(grant["expires_at"])
            )
        else:
            runtime.clear_inference_grant()

    uvicorn.run(app, host="0.0.0.0", port=9101, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
