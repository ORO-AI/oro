"""Drain-mode sentinel-file hook for the validator main loop (ORO-1150).

When the operator wants this validator to stop claiming NEW work but finish
any in-flight evals (e.g. before stopping the container, before an
autoscaling group terminates the host, before a planned maintenance window),
they create the sentinel file. The main loop checks for it once per claim
attempt and, when present, skips the claim and sleeps instead.

The mechanism is intentionally generic — any orchestrator (systemd timer,
k8s preStop hook, AWS ASG lifecycle drain script, manual `touch`) can flip
it. The orchestrator clears the file (or removes it / lets the container
restart) to take the validator back out of drain mode.

Path is overridable via ``ORO_DRAIN_FILE`` env var for non-default container
layouts.
"""

from __future__ import annotations

import os
import time
from typing import Optional

DRAIN_FILE = os.environ.get("ORO_DRAIN_FILE", "/var/run/oro-validator/drain")
# Cache drain-file existence check briefly so the main loop doesn't stat() on
# every iteration when poll_interval is short.
DRAIN_CACHE_TTL_SECONDS = 10


def drain_mode_active(
    cache: dict[str, float],
    *,
    now: Optional[float] = None,
    drain_file: str = DRAIN_FILE,
) -> bool:
    """Return True iff the validator should stop claiming new work.

    ``cache`` is a single-entry dict the caller persists across calls — keeps
    ``os.path.exists`` cost off the hot path. Mutated in place.

    ``now`` and ``drain_file`` are injectable for tests.
    """
    t = time.time() if now is None else now
    if t - cache.get("checked_at", 0.0) < DRAIN_CACHE_TTL_SECONDS:
        return bool(cache.get("active", False))
    active = os.path.exists(drain_file)
    cache["checked_at"] = t
    cache["active"] = active
    return active
