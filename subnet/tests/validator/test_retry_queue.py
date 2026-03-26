"""Tests for LocalRetryQueue.

Uses temp_storage_path and mock_backend_client fixtures from conftest.py.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest

# Add test-subnet to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from oro_sdk.models import ProblemProgressUpdate, ProblemStatus
from oro_sdk.models.terminal_status import TerminalStatus

from validator.backend_client import BackendError
from validator.models import CompletionRequest
from validator.retry_queue import LocalRetryQueue


@pytest.fixture
def sample_completion():
    """Sample completion request for retry queue tests."""
    return CompletionRequest(
        eval_run_id=UUID("12345678-1234-1234-1234-123456789012"),
        status=TerminalStatus.SUCCESS,
        validator_score=0.85,
        score_components={"accuracy": 0.9},
        results_s3_key="logs/run-123.tar.gz",
    )


class TestLocalRetryQueue:
    def test_add_persists_to_file(
        self, temp_storage_path, mock_backend_client, sample_completion
    ):
        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        queue.add(sample_completion)

        with open(temp_storage_path) as f:
            data = json.load(f)

        assert len(data["pending"]) == 1
        assert (
            data["pending"][0]["eval_run_id"] == "12345678-1234-1234-1234-123456789012"
        )

    def test_get_pending_count(
        self, temp_storage_path, mock_backend_client, sample_completion
    ):
        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        assert queue.get_pending_count() == 0

        queue.add(sample_completion)
        assert queue.get_pending_count() == 1

    def test_process_pending_removes_on_success(
        self, temp_storage_path, mock_backend_client, sample_completion
    ):
        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        queue.add(sample_completion)

        mock_backend_client.complete_run.return_value = MagicMock()

        queue.process_pending()

        assert queue.get_pending_count() == 0
        mock_backend_client.complete_run.assert_called_once()

    def test_process_pending_keeps_on_failure(
        self, temp_storage_path, mock_backend_client, sample_completion
    ):
        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        queue.add(sample_completion)

        mock_backend_client.complete_run.side_effect = BackendError(
            "Server unavailable"
        )

        queue.process_pending()

        assert queue.get_pending_count() == 1

    def test_retry_count_increments(
        self, temp_storage_path, mock_backend_client, sample_completion
    ):
        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        queue.add(sample_completion)

        mock_backend_client.complete_run.side_effect = BackendError(
            "Server unavailable"
        )

        queue.process_pending()
        queue.process_pending()

        with open(temp_storage_path) as f:
            data = json.load(f)

        assert data["pending"][0]["retry_count"] == 2

    def test_loads_existing_queue_on_init(self, temp_storage_path, mock_backend_client):
        existing_data = {
            "pending": [
                {
                    "eval_run_id": "12345678-1234-1234-1234-123456789012",
                    "terminal_status": "SUCCESS",
                    "validator_score": 0.7,
                    "score_components": {},
                    "results_s3_key": "logs/old.tar.gz",
                    "added_at": "2025-01-13T10:00:00",
                    "retry_count": 1,
                }
            ]
        }
        with open(temp_storage_path, "w") as f:
            json.dump(existing_data, f)

        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        assert queue.get_pending_count() == 1


class TestRetryQueueProgress:
    """Tests for progress entry support (ORO-340)."""

    @pytest.fixture
    def sample_progress(self):
        return ProblemProgressUpdate(
            problem_id=UUID("11111111-1111-1111-1111-111111111111"),
            status=ProblemStatus.SUCCESS,
            score=0.85,
        )

    def test_add_progress_persists_to_file(
        self, temp_storage_path, mock_backend_client, sample_progress
    ):
        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        eval_run_id = UUID("12345678-1234-1234-1234-123456789012")
        queue.add_progress(eval_run_id, sample_progress)

        with open(temp_storage_path) as f:
            data = json.load(f)

        assert len(data["pending"]) == 1
        entry = data["pending"][0]
        assert entry["type"] == "progress"
        assert entry["eval_run_id"] == str(eval_run_id)
        assert entry["problem_id"] == "11111111-1111-1111-1111-111111111111"
        assert entry["status"] == "SUCCESS"
        assert entry["score"] == 0.85
        assert "retry_count" in entry
        assert "added_at" in entry

    def test_add_progress_handles_unset_fields(
        self, temp_storage_path, mock_backend_client
    ):
        """Progress with no score or logs_s3_key should omit those fields."""
        progress = ProblemProgressUpdate(
            problem_id=UUID("11111111-1111-1111-1111-111111111111"),
            status=ProblemStatus.RUNNING,
        )
        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        queue.add_progress(UUID("12345678-1234-1234-1234-123456789012"), progress)

        with open(temp_storage_path) as f:
            data = json.load(f)

        entry = data["pending"][0]
        assert "score" not in entry
        assert "logs_s3_key" not in entry

    def test_process_pending_retries_progress(
        self, temp_storage_path, mock_backend_client, sample_progress
    ):
        """Successful progress retry removes entry from queue."""
        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        queue.add_progress(UUID("12345678-1234-1234-1234-123456789012"), sample_progress)

        queue.process_pending()

        assert queue.get_pending_count() == 0
        mock_backend_client.report_progress.assert_called_once()

    def test_process_pending_keeps_progress_on_transient(
        self, temp_storage_path, mock_backend_client, sample_progress
    ):
        """Transient error keeps progress entry for retry."""
        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        queue.add_progress(UUID("12345678-1234-1234-1234-123456789012"), sample_progress)

        mock_backend_client.report_progress.side_effect = BackendError(
            "Server error", status_code=500
        )

        queue.process_pending()

        assert queue.get_pending_count() == 1

    def test_process_pending_drops_progress_on_permanent(
        self, temp_storage_path, mock_backend_client, sample_progress
    ):
        """Permanent error drops progress entry."""
        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        queue.add_progress(UUID("12345678-1234-1234-1234-123456789012"), sample_progress)

        mock_backend_client.report_progress.side_effect = BackendError(
            "Bad request", status_code=400
        )

        queue.process_pending()

        assert queue.get_pending_count() == 0

    def test_process_pending_handles_mixed_entries(
        self, temp_storage_path, mock_backend_client, sample_completion, sample_progress
    ):
        """Queue with both completion and progress entries processes both."""
        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        queue.add(sample_completion)
        queue.add_progress(UUID("12345678-1234-1234-1234-123456789012"), sample_progress)

        mock_backend_client.complete_run.return_value = MagicMock()

        queue.process_pending()

        assert queue.get_pending_count() == 0
        mock_backend_client.complete_run.assert_called_once()
        mock_backend_client.report_progress.assert_called_once()

    def test_legacy_entries_without_type_treated_as_completion(
        self, temp_storage_path, mock_backend_client
    ):
        """Entries without 'type' field are treated as completions (backwards compat)."""
        legacy_data = {
            "pending": [
                {
                    "eval_run_id": "12345678-1234-1234-1234-123456789012",
                    "terminal_status": "SUCCESS",
                    "validator_score": 0.7,
                    "score_components": {},
                    "results_s3_key": "logs/old.tar.gz",
                    "added_at": "2025-01-13T10:00:00",
                    "retry_count": 0,
                }
            ]
        }
        with open(temp_storage_path, "w") as f:
            json.dump(legacy_data, f)

        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        mock_backend_client.complete_run.return_value = MagicMock()

        queue.process_pending()

        assert queue.get_pending_count() == 0
        mock_backend_client.complete_run.assert_called_once()
