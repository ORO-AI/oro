"""Shared type definitions for the ShoppingBench agent and scoring pipeline."""

from __future__ import annotations

from typing import Any, List, TypedDict


class ScoreDict(TypedDict, total=False):
    """Per-problem scoring output from ProblemScorer."""
    gt: float
    rule: float
    format: float
    length: float
    product: float
    shop: float
    budget: float
    title: float
    price: float
    service: float


class AggregateScore(TypedDict):
    """Output of compute_aggregate()."""
    ground_truth_rate: float
    success_rate: float
    format_score: float
    field_matching: float
    total_problems: int
    successful_problems: int
    scored_problems: int


class SandboxMetadata(TypedDict):
    """Metadata from sandbox execution."""
    exit_code: int | None
    duration_seconds: float | None
    stderr_tail: str | None


class DialogueMessage(TypedDict, total=False):
    """The `message` sub-dict of a dialogue step.

    Subset of think/tool_call/response — keys are present only when the
    agent emitted that role on this step.
    """
    think: str
    tool_call: List[dict]
    response: str


class DialogueCompletion(TypedDict):
    """The `completion` sub-dict of a dialogue step."""
    reasoning_content: str
    content: str
    message: DialogueMessage


class DialogueExtraInfo(TypedDict):
    """The `extra_info` sub-dict of a dialogue step."""
    step: int
    query: str
    timestamp: int


class DialogueStep(TypedDict):
    """One step in an agent trajectory, as emitted by `create_dialogue_step`."""
    completion: DialogueCompletion
    extra_info: DialogueExtraInfo


# A complete agent trajectory: list of steps in execution order.
Dialogue = List[DialogueStep]


class ProblemDict(TypedDict, total=False):
    """A single problem record as loaded from a problem suite or fetched
    from the backend.

    `query` and `category` are always present; the rest are optional and
    depend on the source (suite vs backend) and category (product/shop/
    voucher).
    """
    query: str
    category: str
    reward: Any
    difficulty: str
    title: str
    reward_title_embeddings: Any
    problem_id: str


class JudgeResult(TypedDict):
    """Return value of `score_reasoning_quality`.

    `score` is 0.0–1.0; `model` is the Chutes model that returned the
    parseable JSON, or "" when the judge failed entirely.
    """
    score: float
    explanation: str
    model: str
    inference_failed: int
    inference_total: int


class ReasoningSummary(TypedDict):
    """Aggregate reasoning metrics returned by
    `ProgressReporter.get_reasoning_data` for run-level score components.
    """
    reasoning_quality: float
    reasoning_coefficient: float
    judge_inference_failed: int
    judge_inference_total: int


class ScoreComponentsSummary(TypedDict, total=False):
    """Per-problem reasoning detail attached to a `ProblemProgressUpdate`."""
    reasoning_explanation: str
    reasoning_model: str
