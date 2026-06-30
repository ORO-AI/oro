"""Validator-internal data types shared across modules.

Agent-level types live in :mod:`src.agent.types`; this module is for
state types that only validator components produce or consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict

from oro_sdk.models import ProblemStatus

from src.agent.types import ScoreDict


class ProblemFailureReason(str, Enum):
    """Why a problem was not scored as SUCCESS.

    Distinct from ``ProblemStatus`` so we can keep the wire-level status
    coarse (SUCCESS/FAILED/TIMED_OUT) while still attributing every
    non-SUCCESS outcome to a specific cause for triage. ORO-1461: before
    this enum, scoring-path exceptions silently produced no result and
    the end-of-run sweep marked the problem TIMED_OUT, masking real bugs
    like a missing voucher metadata field.
    """

    SCORING_EXCEPTION = "scoring_exception"  # scorer raised during score_problem
    SCORING_RETURNED_NONE = "scoring_returned_none"  # scorer returned no usable dict
    MISSING_METADATA = "missing_metadata"  # required problem.metadata.* field absent
    NO_SCORER_FOR_CATEGORY = "no_scorer_for_category"
    NO_DIALOGUE = "no_dialogue"  # envelope success but dialogue empty/non-list
    UNKNOWN_PROBLEM = "unknown_problem"  # dispatched id not in problem set
    JUDGE_INFERENCE_FAILED = "judge_inference_failed"


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
    reasoning_inf_402: int = 0
    execution_time: float | None = None
    # When status != SUCCESS, the most specific reason we could attribute.
    # None for SUCCESS or for genuine sandbox TIMED_OUT / FAILED records
    # where the sandbox itself already told us what happened (ORO-1461).
    failure_reason: ProblemFailureReason | None = None
