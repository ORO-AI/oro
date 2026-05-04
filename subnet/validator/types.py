"""Validator-internal data types shared across modules.

Agent-level types live in :mod:`src.agent.types`; this module is for
state types that only validator components produce or consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from oro_sdk.models import ProblemStatus

from src.agent.types import ScoreDict


class ResourceMetrics(TypedDict, total=False):
    """Host resource utilisation snapshot reported on the heartbeat path."""
    cpu_pct: float
    ram_pct: float
    disk_pct: float
    docker_container_count: int


@dataclass
class EnvelopeMeta:
    """Per-problem metadata captured from the sandbox envelope line.

    Held under ``ProgressReporter._lock`` and read by both dispatch (terminal
    branch) and the scoring worker thread.
    """

    inference_failure_count: int
    inference_total: int
    execution_time: float


@dataclass
class ProblemResult:
    """Single source of truth for one problem's scoring outcome."""

    problem_id: str
    category: str
    status: ProblemStatus
    score: float
    score_dict: ScoreDict = field(default_factory=dict)
    inference_failures: int = 0
    inference_total: int = 0
    reasoning_score: float | None = None
    reasoning_explanation: str = ""
    reasoning_model: str = ""
    reasoning_inf_failed: int = 0
    reasoning_inf_total: int = 0
    execution_time: float | None = None
