"""Drain-mode sentinel-file hook for the validator main loop (ORO-1150).

Touch ``DRAIN_FILE`` to make the validator stop claiming new work but keep
finishing in-flight evals. Remove it to resume. Path is overridable via
``ORO_DRAIN_FILE``. The orchestrator (AWS ASG drain script, k8s preStop
hook, manual touch) writes the file; the validator polls it.

Detection latency upper bound: ``DRAIN_CACHE_TTL_SECONDS`` (cache age) +
the main loop's ``poll_interval`` (next claim opportunity). In the worst
case a long claim_work response can extend this further. Sub-second
detection isn't a goal — race-scale-down drain tolerates ~10-15s.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Protocol

DRAIN_FILE = os.environ.get("ORO_DRAIN_FILE", "/var/run/oro-validator/drain")
DRAIN_CACHE_TTL_SECONDS = 10


def drain_mode_active(
    cache: dict, *, now: float | None = None, drain_file: str = DRAIN_FILE
) -> bool:
    """True iff ``drain_file`` exists, OR the path is unreadable.

    Fail-CLOSED on EACCES / ENOTDIR / missing-mount because the alternative
    — silently keep claiming work while the orchestrator thinks we're
    draining — is the worse failure mode (the exact ABANDON scenario this
    feature exists to prevent). Operators see a WARNING in container logs
    on the first unreadable check, so a misconfigured control path is
    observable rather than silent.

    ``cache`` is a dict the caller persists across calls; ``os.stat`` runs
    at most once per ``DRAIN_CACHE_TTL_SECONDS``. ``now`` is injectable
    for tests.
    """
    t = time.time() if now is None else now
    if t - cache.get("checked_at", 0.0) < DRAIN_CACHE_TTL_SECONDS:
        return bool(cache.get("active", False))
    cache["checked_at"] = t
    try:
        os.stat(drain_file)
        active = True
    except FileNotFoundError:
        active = False
    except OSError as e:
        # Wrong ownership/mode on the bind-mount dir, mount missing,
        # parent path replaced by a file, etc. Treat as draining and
        # surface the misconfiguration.
        logging.warning(
            f"Drain sentinel path unreadable ({type(e).__name__}: {e}) — "
            "fail-CLOSED, treating as draining. Check the host bind mount "
            "and ownership on the parent directory."
        )
        active = True
    cache["active"] = active
    return active


class _RetryQueue(Protocol):
    def get_pending_count(self) -> int: ...
    def process_pending(self, *, count_attempts: bool = ...) -> None: ...


def handle_drain_tick(
    state: dict,
    retry_queue: _RetryQueue,
    poll_interval: float,
    *,
    metric_counter=None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now: float | None = None,
    drain_file: str = DRAIN_FILE,
) -> bool:
    """Inspect the drain sentinel and decide whether the main-loop tick
    should short-circuit (ORO-1150).

    Returns True iff drain mode is active — caller should `continue`,
    skipping claim_work, auto-update, and in-flight bookkeeping. False
    lets the rest of the loop body run normally.

    ``state`` is a dict the caller persists across calls — holds both the
    ``drain_mode_active`` cache and the ``logged`` flag for log-once
    transitions. ``metric_counter`` is the dedicated drain-tick counter
    (kept separate from CLAIM_WORK_TOTAL so claim-success dashboards
    aren't skewed by drain windows). ``sleep_fn`` and ``now`` are
    injectable for tests.

    Retry-queue flush uses ``count_attempts=False`` so a multi-minute
    drain plus a coincident backend transient can't permanently drop
    pending completion/score reports before instance termination.
    """
    if drain_mode_active(state, now=now, drain_file=drain_file):
        if not state.get("logged"):
            logging.info(
                "Drain sentinel present — skipping new claim_work; "
                "in-flight evals finish normally, auto-update suppressed"
            )
            state["logged"] = True
        if metric_counter is not None:
            metric_counter.inc()
        if retry_queue.get_pending_count() > 0:
            retry_queue.process_pending(count_attempts=False)
        sleep_fn(poll_interval)
        return True
    if state.get("logged"):
        logging.info("Drain sentinel cleared — resuming claim_work")
        state["logged"] = False
    return False
