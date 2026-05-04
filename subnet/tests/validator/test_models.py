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

    def test_from_dict(self):
        """Test deserialization from dict loaded from JSON."""
        data = {
            "eval_run_id": "12345678-1234-1234-1234-123456789012",
            "terminal_status": "FAILED",
            "validator_score": 0.0,
            "score_components": {"error": "timeout"},
            "results_s3_key": "logs/run-456.tar.gz",
        }
        request = CompletionRequest.from_dict(data)
        assert request.eval_run_id == UUID("12345678-1234-1234-1234-123456789012")
        assert request.status == TerminalStatus.FAILED
        assert request.validator_score == 0.0
        assert request.score_components == {"error": "timeout"}
        assert request.results_s3_key == "logs/run-456.tar.gz"

    def test_roundtrip(self):
        """Test that to_dict -> from_dict preserves data."""
        original = CompletionRequest(
            eval_run_id=UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
            status=TerminalStatus.SUCCESS,
            validator_score=0.75,
            score_components={"metric_a": 0.8, "metric_b": 0.7},
            results_s3_key="logs/roundtrip-test.tar.gz",
        )
        data = original.to_dict()
        restored = CompletionRequest.from_dict(data)

        assert restored.eval_run_id == original.eval_run_id
        assert restored.status == original.status
        assert restored.validator_score == original.validator_score
        assert restored.score_components == original.score_components
        assert restored.results_s3_key == original.results_s3_key
