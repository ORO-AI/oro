"""``/v1/session/call`` must map every exception runtime.call() can raise to
a structured HTTPException response — never let one fall through FastAPI's
default handler as an unstructured, undistinguishable 500.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from validator.session_errors import (
    HarnessExecutionError,
    HarnessResponseError,
    HarnessTimeoutError,
    InvalidSessionError,
)
from validator.session_service import SessionRuntime, create_session_app


class _RaisingRegistry:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def call(self, envelope: dict[str, Any]) -> dict[str, Any]:
        raise self._exc


def _client_for(exc: Exception) -> TestClient:
    runtime = SessionRuntime()
    runtime.install(_RaisingRegistry(exc))
    app = create_session_app(runtime)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("exc", "status", "environment_error"),
    [
        (HarnessTimeoutError("tool call exceeded 10s"), 504, True),
        (HarnessExecutionError("tool call failed: RuntimeError"), 500, True),
        (InvalidSessionError("unknown session_id"), 409, None),
        (ValueError("turn must be a positive integer"), 422, None),
        (RuntimeError("no environment pack is active"), 503, None),
        # session_registry's own response-shape guards (_public_step_result,
        # _simulator_response, _simulator_exchanges) raise this dedicated
        # exception on a malformed runtime/simulator result. Previously
        # uncaught here (a bare TypeError, unhandled), this fell through
        # FastAPI's default handler as a bare, unstructured 500 -- unlike
        # every other harness failure, which the sandbox client can't
        # distinguish from an unrelated crash. A bare TypeError is
        # deliberately NOT mapped here: an unrelated internal TypeError from
        # other registry logic must not be reported as environment_error.
        (HarnessResponseError("environment step result must be an object"), 500, True),
    ],
)
def test_call_endpoint_maps_every_exception_to_structured_response(
    exc: Exception, status: int, environment_error: bool | None
) -> None:
    client = _client_for(exc)
    response = client.post("/v1/session/call", json={"session_id": "x"})

    assert response.status_code == status
    detail = response.json()["detail"]
    assert detail["error"] == str(exc)
    if environment_error is not None:
        assert detail["environment_error"] is environment_error
    else:
        assert "environment_error" not in detail


def test_unrelated_type_error_is_not_reported_as_environment_error() -> None:
    # A bare TypeError from somewhere else in registry.call() -- not one of
    # the named response-shape guards -- is an internal validator defect,
    # not an environment failure. It must NOT be caught and relabeled
    # environment_error=True; that would misattribute the bug to the
    # environment. Confirms the fix is scoped to HarnessResponseError only.
    client = _client_for(TypeError("unrelated internal bug, not a shape guard"))
    response = client.post("/v1/session/call", json={"session_id": "x"})

    assert response.status_code == 500
    assert response.text == "Internal Server Error"
