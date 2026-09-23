"""Rate-limited terminal reports must enter the durable retry queue."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from oro_sdk.models.terminal_status import TerminalStatus

from validator.backend_client import BackendClient, BackendError
from validator.main import Validator
from validator.retry_queue import LocalRetryQueue


@pytest.mark.parametrize("failed", [False, True])
def test_rate_limited_terminal_report_is_queued_and_retried(tmp_path, failed):
    backend = MagicMock(spec=BackendClient)
    backend.complete_run.side_effect = BackendError("Rate limited", status_code=429)
    validator = Validator.__new__(Validator)
    validator.backend_client = backend
    storage_path = tmp_path / "retry_queue.json"
    validator.retry_queue = LocalRetryQueue(backend, storage_path)
    run_id = uuid4()

    if failed:
        validator._complete_with_failure(
            run_id, TerminalStatus.FAILED, "Test execution failure"
        )
    else:
        validator._complete_run(run_id, TerminalStatus.SUCCESS, 0.75)

    assert validator.retry_queue.get_pending_count() == 1
    # A fresh queue instance must recover the same terminal report after restart.
    restored = LocalRetryQueue(backend, storage_path)
    assert restored.get_pending_count() == 1
    backend.complete_run.side_effect = None
    restored.process_pending()
    assert restored.get_pending_count() == 0
    assert backend.complete_run.call_args.kwargs["eval_run_id"] == run_id
    if failed:
        assert backend.complete_run.call_args.kwargs["failure_reason"] == "Test execution failure"
    else:
        assert backend.complete_run.call_args.kwargs["score"] == 0.75
