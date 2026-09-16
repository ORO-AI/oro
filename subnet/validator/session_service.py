"""Private HTTP bridge from the sandbox proxy to validator-owned sessions."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, HTTPException

from .session_errors import (
    HarnessExecutionError,
    HarnessResponseError,
    HarnessTimeoutError,
    InvalidSessionError,
)

if TYPE_CHECKING:
    from .session_registry import SessionRegistry


class SessionRuntime:
    """Thread-safe holder for the pack registry active in the validator."""

    def __init__(self) -> None:
        self._registry: SessionRegistry | None = None
        self._lock = threading.Lock()

    def install(self, registry: SessionRegistry) -> None:
        """Atomically activate a registry and close the previous generation."""

        with self._lock:
            previous = self._registry
            self._registry = registry
        if previous is not None and previous is not registry:
            previous.close()

    def clear(self, registry: SessionRegistry | None = None) -> None:
        """Deactivate the current registry, optionally only if it still matches."""

        with self._lock:
            if registry is not None and self._registry is not registry:
                return
            previous = self._registry
            self._registry = None
        if previous is not None:
            previous.close()

    def call(self, envelope: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            registry = self._registry
        if registry is None:
            raise RuntimeError("no environment pack is active")
        return registry.call(envelope)

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._registry is not None


def create_session_app(runtime: SessionRuntime) -> FastAPI:
    """Build the internal app; session creation and verdict are not routed."""

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "ready": runtime.ready}

    @app.post("/v1/session/call")
    def call(envelope: dict[str, Any]) -> dict[str, Any]:
        try:
            return runtime.call(envelope)
        except HarnessTimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail={"error": str(exc), "environment_error": True},
            ) from exc
        except HarnessExecutionError as exc:
            raise HTTPException(
                status_code=500,
                detail={"error": str(exc), "environment_error": True},
            ) from exc
        except HarnessResponseError as exc:
            # session_registry's own named response-shape guards (e.g.
            # _public_step_result, _simulator_response, _simulator_exchanges)
            # raise this when the runtime/simulator returns a malformed
            # result -- an environment-side failure, not caller input. Left
            # uncaught (this used to be a bare TypeError, unhandled here),
            # this fell through FastAPI's default handler as a bare 500 with
            # no body, unlike every other harness failure here. Deliberately
            # narrower than "except TypeError": an unrelated internal
            # TypeError from other registry logic should NOT be reported as
            # environment_error, since that would misattribute a validator
            # defect as an environment failure.
            raise HTTPException(
                status_code=500,
                detail={"error": str(exc), "environment_error": True},
            ) from exc
        except InvalidSessionError as exc:
            raise HTTPException(status_code=409, detail={"error": str(exc)}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"error": str(exc)}) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail={"error": str(exc)}) from exc

    return app


class SessionServer:
    """Run the internal FastAPI bridge beside the synchronous validator loop."""

    def __init__(self, runtime: SessionRuntime, *, host: str, port: int) -> None:
        config = uvicorn.Config(
            create_session_app(runtime),
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            name="session-runtime-http",
            daemon=True,
        )

    def start(self, *, timeout: float = 5.0) -> None:
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started and self._thread.is_alive():
            if time.monotonic() >= deadline:
                raise RuntimeError("session runtime HTTP server did not start")
            time.sleep(0.01)
        if not self._server.started:
            raise RuntimeError("session runtime HTTP server failed to start")

    def stop(self) -> None:
        self._server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)


__all__ = ["SessionRuntime", "SessionServer", "create_session_app"]
