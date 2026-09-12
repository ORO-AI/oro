"""Background batching for generated-environment episode results."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from bittensor.utils.btlogging import logging

from .session_registry import SessionRegistry

REPORT_INTERVAL_SECONDS = 10.0
POLL_INTERVAL_SECONDS = 1.0


class GeneratedProgressReporter:
    """Emit each terminal task once, with a final blocking flush."""

    def __init__(
        self,
        registry: SessionRegistry,
        emit_batch: Callable[[list[dict[str, Any]]], None],
        *,
        report_interval: float = REPORT_INTERVAL_SECONDS,
        poll_interval: float = POLL_INTERVAL_SECONDS,
    ) -> None:
        self._registry = registry
        self._emit_batch = emit_batch
        self._report_interval = report_interval
        self._poll_interval = poll_interval
        self._acknowledged: set[str] = set()
        self._last_attempt = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        self._thread = thread

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def flush(self, results: list[dict[str, Any]]) -> None:
        pending = [
            result
            for result in results
            if str(result["task_id"]) not in self._acknowledged
        ]
        if not pending:
            return
        self._emit_batch(pending)
        self._acknowledged.update(str(result["task_id"]) for result in pending)

    def _run(self) -> None:
        while not self._stop_event.wait(self._poll_interval):
            now = time.monotonic()
            if now - self._last_attempt < self._report_interval:
                continue
            results = self._registry.terminal_results()
            if not results:
                continue
            self._last_attempt = now
            try:
                self.flush(results)
            except Exception as exc:
                logging.warning(
                    "Generated progress batch failed; will retry: "
                    f"{type(exc).__name__}: {exc}"
                )


__all__ = ["GeneratedProgressReporter"]
