"""Tests for local models (CompletionRequest only).

All other models (ClaimWorkResponse, HeartbeatResponse, etc.) are provided by
the oro-sdk and are tested within that package.
"""

from uuid import UUID

from oro_sdk.models.terminal_status import TerminalStatus

from validator.models import CompletionRequest


class TestCompletionRequest:
    """Tests for CompletionRequest - the only local model."""

    def test_to_dict(self):
        """Test serialization to dict for JSON persistence."""
        request = CompletionRequest(
            eval_run_id=UUID("12345678-1234-1234-1234-123456789012"),
            status=TerminalStatus.SUCCESS,
            validator_score=0.85,
            score_components={"accuracy": 0.9},
            results_s3_key="logs/run-123.tar.gz",
        )
        data = request.to_dict()
        assert data["eval_run_id"] == "12345678-1234-1234-1234-123456789012"
        assert data["terminal_status"] == "SUCCESS"
        assert data["validator_score"] == 0.85
        assert data["score_components"] == {"accuracy": 0.9}
        assert data["results_s3_key"] == "logs/run-123.tar.gz"
