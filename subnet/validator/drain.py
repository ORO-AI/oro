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

import os
import time

DRAIN_FILE = os.environ.get("ORO_DRAIN_FILE", "/var/run/oro-validator/drain")
DRAIN_CACHE_TTL_SECONDS = 10


def drain_mode_active(
    cache: dict, *, now: float | None = None, drain_file: str = DRAIN_FILE
) -> bool:
    """True iff ``drain_file`` exists. ``cache`` is a dict the caller persists
    across calls; ``os.path.exists`` runs at most once per
    ``DRAIN_CACHE_TTL_SECONDS``. ``now`` is injectable for tests.
    """
    t = time.time() if now is None else now
    if t - cache.get("checked_at", 0.0) < DRAIN_CACHE_TTL_SECONDS:
        return bool(cache.get("active", False))
    cache["checked_at"] = t
    cache["active"] = active = os.path.exists(drain_file)
    return active
