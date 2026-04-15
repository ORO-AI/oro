"""Shared type definitions for the ShoppingBench agent and scoring pipeline."""

from __future__ import annotations

from typing import TypedDict


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
