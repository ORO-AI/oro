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
    def test_rate_limited_completion_survives_restart_and_retries(
        self, temp_storage_path, mock_backend_client, sample_completion
    ):
        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        queue.add(sample_completion)
        mock_backend_client.complete_run.side_effect = BackendError(
            "rate limited", status_code=429
        )

        queue.process_pending()

        restarted = LocalRetryQueue(mock_backend_client, temp_storage_path)
        assert restarted.get_pending_count() == 1
        with open(temp_storage_path) as f:
            assert json.load(f)["pending"][0]["retry_count"] == 1

        mock_backend_client.complete_run.side_effect = None
        restarted.process_pending()
        assert restarted.get_pending_count() == 0

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


@pytest.fixture
def sample_progress_update():
    """Sample progress update for retry queue progress tests."""
    from oro_sdk.models import ProblemProgressUpdate
    from oro_sdk.models.problem_status import ProblemStatus

    return ProblemProgressUpdate(
        problem_id=UUID("87654321-4321-4321-4321-210987654321"),
        status=ProblemStatus.SUCCESS,
        logs_s3_key="logs/run-123/p-87654321.jsonl.gz",
    )


class TestLocalRetryQueueProgress:
    """Regression coverage for add_progress()/process_pending() on 'progress' entries.

    main.py calls retry_queue.add_progress() when report_progress() fails
    while reporting logs_s3_key (see _upload_logs), but add_progress() and
    the 'progress' branch of process_pending() previously did not exist —
    the call raised AttributeError, which the caller's broad except turned
    into a misleading "Failed to upload logs" error even when every upload
    had actually succeeded, and the retry was silently lost.
    """

    def test_add_progress_persists_to_file(
        self, temp_storage_path, mock_backend_client, sample_progress_update
    ):
        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        eval_run_id = UUID("12345678-1234-1234-1234-123456789012")

        queue.add_progress(eval_run_id, sample_progress_update)

        with open(temp_storage_path) as f:
            data = json.load(f)

        assert len(data["pending"]) == 1
        assert data["pending"][0]["type"] == "progress"
        assert data["pending"][0]["eval_run_id"] == str(eval_run_id)
        assert queue.get_pending_count() == 1

    def test_process_pending_removes_progress_on_success(
        self, temp_storage_path, mock_backend_client, sample_progress_update
    ):
        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        queue.add_progress(
            UUID("12345678-1234-1234-1234-123456789012"), sample_progress_update
        )

        mock_backend_client.report_progress.return_value = None

        queue.process_pending()

        assert queue.get_pending_count() == 0
        mock_backend_client.report_progress.assert_called_once()

        # The value that motivated add_progress()/_process_progress() in the
        # first place (logs_s3_key) must survive the JSON to_dict/from_dict
        # round trip intact, not just "some update" being replayed.
        call_args = mock_backend_client.report_progress.call_args
        eval_run_id_arg, updates_arg = call_args.args
        assert eval_run_id_arg == UUID("12345678-1234-1234-1234-123456789012")
        assert len(updates_arg) == 1
        assert updates_arg[0].logs_s3_key == sample_progress_update.logs_s3_key
        assert updates_arg[0].problem_id == sample_progress_update.problem_id

    def test_process_pending_keeps_progress_on_transient_failure(
        self, temp_storage_path, mock_backend_client, sample_progress_update
    ):
        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        queue.add_progress(
            UUID("12345678-1234-1234-1234-123456789012"), sample_progress_update
        )

        # No status_code/sdk_error set -> is_transient defaults to True,
        # same convention test_process_pending_keeps_on_failure relies on above.
        mock_backend_client.report_progress.side_effect = BackendError(
            "Server unavailable"
        )

        queue.process_pending()

        assert queue.get_pending_count() == 1

    def test_process_pending_retries_unwrapped_transport_error(
        self, temp_storage_path, mock_backend_client, sample_progress_update
    ):
        # A transport-level error _call_api didn't convert to BackendError
        # (e.g. a raw ConnectionError) must still be retried, not dropped.
        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        queue.add_progress(
            UUID("12345678-1234-1234-1234-123456789012"), sample_progress_update
        )

        mock_backend_client.report_progress.side_effect = ConnectionError("reset")

        queue.process_pending()

        assert queue.get_pending_count() == 1
        with open(temp_storage_path) as f:
            data = json.load(f)
        assert data["pending"][0]["retry_count"] == 1

    def test_malformed_progress_entry_does_not_abort_other_entries(
        self, temp_storage_path, mock_backend_client, sample_progress_update
    ):
        # A malformed persisted entry (bad UUID) must not raise past
        # _process_progress and abort process_pending()'s whole loop --
        # every other pending entry (here, a second, well-formed progress
        # update) still has to be attempted in the same pass.
        queue = LocalRetryQueue(mock_backend_client, temp_storage_path)
        queue.add_progress(
            UUID("12345678-1234-1234-1234-123456789012"), sample_progress_update
        )
        with open(temp_storage_path) as f:
            data = json.load(f)
        data["pending"].insert(0, {**data["pending"][0], "eval_run_id": "not-a-uuid"})
        with open(temp_storage_path, "w") as f:
            json.dump(data, f)

        mock_backend_client.report_progress.return_value = None

        queue.process_pending()

        # The malformed entry is retried (bounded by max_retries) rather
        # than crashing the pass; the well-formed one still got processed.
        mock_backend_client.report_progress.assert_called_once()
        assert queue.get_pending_count() == 1
        with open(temp_storage_path) as f:
            data = json.load(f)
        assert data["pending"][0]["eval_run_id"] == "not-a-uuid"
        assert data["pending"][0]["retry_count"] == 1
