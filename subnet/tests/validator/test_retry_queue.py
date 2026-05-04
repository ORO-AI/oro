"""Tests for LocalRetryQueue.

Uses temp_storage_path and mock_backend_client fixtures from conftest.py.
"""

import json
from unittest.mock import MagicMock
from uuid import UUID

import pytest

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


