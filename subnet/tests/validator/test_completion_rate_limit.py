"""Terminal reports: rate limits enter the durable retry queue, rejections fail the run."""

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


def test_rejected_success_report_fails_the_run():
    backend = MagicMock(spec=BackendClient)
    rejection = BackendError("complete_run: episode set is incomplete", status_code=400)
    backend.complete_run.side_effect = [rejection, MagicMock()]
    validator = Validator.__new__(Validator)
    validator.backend_client = backend
    run_id = uuid4()

    validator._complete_run(run_id, TerminalStatus.SUCCESS, 0.75)

    report = backend.complete_run.call_args.kwargs
    assert report["eval_run_id"] == run_id
    assert report["status"] == TerminalStatus.FAILED
    assert report["failure_reason"] == "Completion rejected by Backend (HTTP 400)"


def test_failure_report_drops_unsendable_sandbox_metadata():
    backend = MagicMock(spec=BackendClient)
    unsendable = ValueError("Out of range float values are not JSON compliant")
    backend.complete_run.side_effect = [unsendable, MagicMock()]
    validator = Validator.__new__(Validator)
    validator.backend_client = backend
    run_id = uuid4()

    validator._complete_with_failure(
        run_id,
        TerminalStatus.FAILED,
        "metadata overflow",
        sandbox_metadata={"cost": float("inf")},
    )

    report = backend.complete_run.call_args.kwargs
    assert report["eval_run_id"] == run_id
    assert report["failure_reason"] == "metadata overflow"
    assert report["sandbox_metadata"] is None
