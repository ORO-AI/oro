"""Background thread for maintaining evaluation lease via heartbeats."""

import threading
import logging
from typing import Optional
from uuid import UUID

from .backend_client import BackendClient, BackendError


class HeartbeatManager:
    """Manages periodic heartbeats to maintain evaluation lease.

    Runs heartbeat calls in a background thread while the main thread
    executes the sandbox. Continues attempting heartbeats even on
    transient failures.
    """

    def __init__(
        self,
        backend_client: BackendClient,
        eval_run_id: UUID,
        interval_seconds: int = 30,
        service_versions: dict[str, str] | None = None,
    ):
        self.backend_client = backend_client
        self.eval_run_id = eval_run_id
        self.interval_seconds = interval_seconds
        self.service_versions = service_versions

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[Exception] = None
        self._healthy = True
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the heartbeat background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the heartbeat thread and wait for it to finish.

        Uses a timeout of interval_seconds + 35 to account for:
        - HTTP request timeout (up to 30s)
        - Buffer time (5s)
        """
        self._stop_event.set()
        if self._thread is not None:
            # Wait for thread to finish, with generous timeout for in-flight requests
            join_timeout = self.interval_seconds + 35
            self._thread.join(timeout=join_timeout)
            if self._thread.is_alive():
                logging.warning(
                    f"Heartbeat thread for {self.eval_run_id} did not stop within {join_timeout}s"
                )

    def is_healthy(self) -> bool:
        """Check if heartbeats are succeeding."""
        with self._lock:
            return self._healthy

    def get_last_error(self) -> Optional[Exception]:
        """Get the last error encountered, if any."""
        with self._lock:
            return self._last_error

    def _run(self) -> None:
        """Background thread main loop."""
        while not self._stop_event.is_set():
            try:
                response = self.backend_client.heartbeat(
                    self.eval_run_id, service_versions=self.service_versions
                )
                with self._lock:
                    self._healthy = True
                    self._last_error = None
                logging.debug(
                    f"Heartbeat success for {self.eval_run_id}, "
                    f"lease expires at {response.lease_expires_at}"
                )
            except BackendError as e:
                with self._lock:
                    self._healthy = False
                    self._last_error = e

                # Permanent errors - stop heartbeating
                if e.is_lease_expired:
                    logging.error(
                        f"Lease expired for {self.eval_run_id}, stopping heartbeat"
                    )
                    return
                if e.is_not_run_owner:
                    logging.error(
                        f"Not run owner for {self.eval_run_id}, stopping heartbeat"
                    )
                    return
                if e.is_eval_run_not_found:
                    logging.error(
                        f"Eval run {self.eval_run_id} not found, stopping heartbeat"
                    )
                    return

                # Transient errors - log and continue
                logging.warning(f"Heartbeat failed for {self.eval_run_id}: {e}")
            except Exception as e:
                with self._lock:
                    self._healthy = False
                    self._last_error = e
                logging.warning(f"Heartbeat failed for {self.eval_run_id}: {e}")

            self._stop_event.wait(self.interval_seconds)
