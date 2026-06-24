"""In-process liveness watchdog for the validator main loop (ORO-1414, Follow-up 2).

The validator container has ``restart: unless-stopped``, which only fires when
the container *exits*. In the 2026-06-23 incident the process wedged (host disk
filled) without exiting, so the container stayed ``Up`` (0B) and never
restarted. This watchdog turns a stalled main loop into a process exit: the
loop calls :meth:`beat` each iteration, and a background thread aborts the
process if no beat lands within ``timeout_seconds`` so the restart policy can
bring a fresh validator back up.

Known limitation (accepted): if the whole interpreter is hard-hung (e.g. GIL
deadlock), the watchdog thread itself is starved and cannot fire.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable

# The loop beats once per iteration, so the default must exceed the longest
# legitimate single iteration. Worst case is a hung update check (~900s) plus a
# full eval (sandbox_timeout 1800s + scoring/reasoning ~900s+) in the same pass,
# ~3600-3900s; 5400s leaves margin so a healthy busy validator never trips it,
# while a truly wedged loop is still recycled within ~90 min. Tighter recovery
# would require beating from the heartbeat thread (fires every 30s during an
# eval) — see ORO-1414 follow-up.
_DEFAULT_TIMEOUT_SECONDS = 5400.0
_DEFAULT_CHECK_INTERVAL = 30.0


class ProgressWatchdog:
    """Abort the process if the main loop stops beating within ``timeout_seconds``."""

    def __init__(
        self,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        *,
        check_interval: float = _DEFAULT_CHECK_INTERVAL,
        now: Callable[[], float] = time.monotonic,
        on_stall: Callable[[], None] | None = None,
    ) -> None:
        self._timeout = timeout_seconds
        self._check_interval = check_interval
        self._now = now
        self._on_stall = on_stall or self._abort
        self._last_beat = now()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def beat(self) -> None:
        """Record forward progress; resets the stall timer."""
        self._last_beat = self._now()

    def stalled(self) -> bool:
        return (self._now() - self._last_beat) > self._timeout

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="validator-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._check_interval + 1.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._check_interval):
            if self.stalled():
                self._on_stall()
                return

    def _abort(self) -> None:
        elapsed = self._now() - self._last_beat
        # WATCHDOG_ABORT is a stable token so a CloudWatch Logs metric filter can
        # alarm on validator self-recoveries (otherwise a restart loop is
        # invisible unless someone reads container logs). A Prometheus counter
        # would not work here: the process exits immediately, before the next
        # scrape, and the counter resets to 0 on restart (ORO-1414).
        logging.error(
            "WATCHDOG_ABORT: validator made no progress for %.0fs (timeout %.0fs); "
            "aborting so the container restart policy recovers the validator (ORO-1414).",
            elapsed,
            self._timeout,
        )
        # os._exit, not sys.exit: bypass cleanup/atexit so a wedged process can't
        # swallow the exit; the container exits and `restart: unless-stopped` fires.
        os._exit(1)
