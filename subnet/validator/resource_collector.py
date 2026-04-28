"""Snapshot the validator host's resource utilisation.

Returned as a small dict with cpu_pct, ram_pct, disk_pct, and
docker_container_count, all optional. The dict is attached to
HeartbeatRequest payloads so the Backend can persist the latest sample on
the Validator row.

Failures collecting any individual metric are logged at DEBUG and that
field is omitted — the heartbeat path must never fail because metrics
are unavailable.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import TypedDict

import psutil

logger = logging.getLogger(__name__)

# Disk volume to sample. Validators run sandbox workdirs under SANDBOX_DIR
# when set, otherwise the root filesystem is the safest fallback (the same
# volume holds the Docker image cache that bloats fastest).
_SANDBOX_DIR_ENV = "SANDBOX_DIR"


class ResourceMetrics(TypedDict, total=False):
    cpu_pct: float
    ram_pct: float
    disk_pct: float
    docker_container_count: int


def collect_resource_metrics() -> ResourceMetrics:
    metrics: ResourceMetrics = {}

    try:
        # interval=None returns the value since the last call (non-blocking
        # after the first invocation). Heartbeats fire every 30s so the gap
        # between calls is always plenty.
        metrics["cpu_pct"] = float(psutil.cpu_percent(interval=None))
    except Exception:
        logger.debug("Failed to sample cpu_percent", exc_info=True)

    try:
        metrics["ram_pct"] = float(psutil.virtual_memory().percent)
    except Exception:
        logger.debug("Failed to sample virtual_memory", exc_info=True)

    try:
        path = os.environ.get(_SANDBOX_DIR_ENV) or "/"
        metrics["disk_pct"] = float(psutil.disk_usage(path).percent)
    except Exception:
        logger.debug("Failed to sample disk_usage", exc_info=True)

    try:
        result = subprocess.run(
            ["docker", "ps", "-q"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            metrics["docker_container_count"] = sum(
                1 for line in result.stdout.splitlines() if line.strip()
            )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        logger.debug("Failed to count docker containers", exc_info=True)

    return metrics
