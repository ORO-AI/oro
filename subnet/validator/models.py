"""Local models for validator functionality.

This module only contains types that are specific to the validator implementation
and not provided by the oro-sdk. All Backend API response types should be imported
directly from oro_sdk.models.
"""

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from oro_sdk.models.terminal_status import TerminalStatus


@dataclass
class CompletionRequest:
    """Request payload for completing a run (used by retry queue).

    This is a local type for persisting failed completion requests to disk.
    The retry queue uses this to store and later retry failed completions.

    For successful runs, all fields are populated.
    For failed runs, score fields may be None.
    """

    eval_run_id: UUID
    status: TerminalStatus
    validator_score: Optional[float] = None
    score_components: Optional[dict] = field(default_factory=dict)
    results_s3_key: str = ""
    failure_reason: Optional[str] = None
    sandbox_metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        """Serialize to dict for JSON persistence."""
        d = {
            "eval_run_id": str(self.eval_run_id),
            "terminal_status": self.status.value,
            "validator_score": self.validator_score,
            "score_components": self.score_components,
            "results_s3_key": self.results_s3_key,
        }
        if self.failure_reason:
            d["failure_reason"] = self.failure_reason
        if self.sandbox_metadata:
            d["sandbox_metadata"] = self.sandbox_metadata
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "CompletionRequest":
        """Deserialize from dict loaded from JSON."""
        return cls(
            eval_run_id=UUID(data["eval_run_id"]),
            status=TerminalStatus(data["terminal_status"]),
            validator_score=data.get("validator_score"),
            score_components=data.get("score_components", {}),
            results_s3_key=data.get("results_s3_key", ""),
            failure_reason=data.get("failure_reason"),
            sandbox_metadata=data.get("sandbox_metadata"),
        )
