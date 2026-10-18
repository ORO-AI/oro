"""Regression tests for _upload_logs post-run progress reporting (ORO-2315).

ORO-702 removed the progress retry queue ("no disk retry queue for progress
reports"), but an orphaned ``self.retry_queue.add_progress(...)`` call was left
on the post-run logs_s3_key report path. It raised AttributeError whenever the
report failed — dead since April, masked by the surrounding try/except.

Option 1 (chosen) deletes the call: a failed logs_s3_key report is dropped with
the real error logged, and the uploaded S3 key is still returned.
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4

from validator.backend_client import BackendClient, BackendError
from validator.main import Validator
from validator.retry_queue import LocalRetryQueue


def _validator(backend_client):
    """A bare Validator with only the attribute _upload_logs uses."""
    v = Validator.__new__(Validator)  # bypass __init__ — isolate _upload_logs
    v.backend_client = backend_client
    return v


def test_report_progress_failure_does_not_raise_and_returns_key(tmp_path):
    """A failed post-run logs_s3_key report must not raise (previously an
    AttributeError from the orphaned add_progress call) and must still return
    the uploaded S3 key."""
    pid = uuid4()

    presign = MagicMock()
    presign.upload_url = "http://s3.local/put"
    presign.results_s3_key = "logs/run/pid.json.gz"

    bc = MagicMock(spec=BackendClient)
    bc.get_presigned_upload_url.return_value = presign
    bc.upload_to_s3.return_value = None
    bc.report_progress.side_effect = BackendError("backend blip", status_code=503)

    progress_reporter = MagicMock()
    progress_reporter.get_problem_status.return_value = "COMPLETED"

    output_file = tmp_path / "output.jsonl"
    output_file.write_text("[]\n")

    with patch(
        "validator.main.split_output_by_problem", return_value={str(pid): b"[]"}
    ):
        result = _validator(bc)._upload_logs(
            eval_run_id=uuid4(),
            output_file=output_file,
            problem_ids=[pid],
            progress_reporter=progress_reporter,
        )

    # Upload succeeded; the report failure is swallowed with a warning, and the
    # key is still returned (no AttributeError falling through to the outer catch).
    assert result == "logs/run/pid.json.gz"
    bc.report_progress.assert_called_once()


def test_retry_queue_has_no_add_progress():
    """Guard against reintroducing the orphaned progress-retry path (ORO-702)."""
    assert not hasattr(LocalRetryQueue, "add_progress")
