"""Snapshot the validator host's resource utilisation.

Each individual metric is collected independently — a failure on one
returns a partial dict instead of raising, because the heartbeat path
must never fail just because a metric is unavailable.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Callable, TypedDict

import psutil

logger = logging.getLogger(__name__)

# Sandbox workdirs live under SANDBOX_DIR; fall back to root when unset
# since that's the volume holding the Docker image cache anyway.
_SANDBOX_DIR_ENV = "SANDBOX_DIR"


class ResourceMetrics(TypedDict, total=False):
    cpu_pct: float
    ram_pct: float
    disk_pct: float
    docker_container_count: int


def _docker_container_count() -> int:
    result = subprocess.run(
        ["docker", "ps", "-q"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        raise OSError(f"docker ps -q exited {result.returncode}")
    return sum(1 for line in result.stdout.splitlines() if line.strip())


# psutil.cpu_percent(interval=None) returns the value since the last call.
# Heartbeats fire every 30s so the inter-call gap is always plenty.
_COLLECTORS: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("cpu_pct", lambda: float(psutil.cpu_percent(interval=None))),
    ("ram_pct", lambda: float(psutil.virtual_memory().percent)),
    (
        "disk_pct",
        lambda: float(psutil.disk_usage(os.getenv(_SANDBOX_DIR_ENV) or "/").percent),
    ),
    ("docker_container_count", _docker_container_count),
)


def collect_resource_metrics() -> ResourceMetrics:
    metrics: ResourceMetrics = {}
    for name, collector in _COLLECTORS:
        try:
            metrics[name] = collector()  # type: ignore[literal-required]
        except Exception:
            logger.debug("Failed to sample %s", name, exc_info=True)
    return metrics
