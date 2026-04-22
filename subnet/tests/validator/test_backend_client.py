"""Tests for backend_client using oro-sdk."""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from bittensor_wallet import Keypair, Wallet

# Add test-subnet to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from oro_sdk.models import ProblemProgressUpdate, ProblemStatus

from validator.backend_client import BackendClient, BackendError


@pytest.fixture
def mock_wallet():
    """Create a mock wallet with a real keypair for signing."""
    keypair = Keypair.create_from_uri("//TestValidator")
    wallet = MagicMock(spec=Wallet)
    wallet.hotkey = keypair
    return wallet


def _create_response(status_code: int, parsed=None):
    """Create a mock Response object."""
    response = MagicMock()
    response.status_code = status_code
    response.parsed = parsed
    response.content = b""
    response.headers = {}
    return response


class TestBackendClientClaimWork:
    def test_claim_work_success(self, mock_wallet):
        from oro_sdk.models.claim_work_response import ClaimWorkResponse

        client = BackendClient("https://api.example.com", mock_wallet)

        sdk_response = ClaimWorkResponse(
            eval_run_id=UUID("12345678-1234-1234-1234-123456789012"),
            agent_version_id=UUID("87654321-4321-4321-4321-210987654321"),
            suite_id=789,
            lease_expires_at=datetime(2025, 1, 13, 12, 0, 0),
            code_download_url="https://example.com/code.py",
        )

        mock_response = _create_response(200, sdk_response)

        with patch(
            "validator.backend_client.claim_work.sync_detailed",
            return_value=mock_response,
        ):
            result = client.claim_work()

        assert result is not None
        assert result.eval_run_id == UUID("12345678-1234-1234-1234-123456789012")
        assert result.code_download_url == "https://example.com/code.py"

    def test_claim_work_with_service_versions(self, mock_wallet):
        from oro_sdk.models.claim_work_response import ClaimWorkResponse

        client = BackendClient("https://api.example.com", mock_wallet)

        sdk_response = ClaimWorkResponse(
            eval_run_id=UUID("12345678-1234-1234-1234-123456789012"),
            agent_version_id=UUID("87654321-4321-4321-4321-210987654321"),
            suite_id=789,
            lease_expires_at=datetime(2025, 1, 13, 12, 0, 0),
            code_download_url="https://example.com/code.py",
        )

        mock_response = _create_response(200, sdk_response)

        versions = {"validator": "sha256:abc123def4", "sandbox": "sha256:def456abc7"}

        with patch(
            "validator.backend_client.claim_work.sync_detailed",
            return_value=mock_response,
        ) as mock_call:
            result = client.claim_work(service_versions=versions)

        assert result is not None
        assert result.eval_run_id == UUID("12345678-1234-1234-1234-123456789012")

        # Verify body was passed with service_versions
        call_kwargs = mock_call.call_args
        assert "body" in call_kwargs.kwargs
        body = call_kwargs.kwargs["body"]
        assert body.service_versions == versions

    def test_claim_work_no_work_available(self, mock_wallet):
        client = BackendClient("https://api.example.com", mock_wallet)

        mock_response = _create_response(204, None)

        with patch(
            "validator.backend_client.claim_work.sync_detailed",
            return_value=mock_response,
        ):
            result = client.claim_work()

        assert result is None


class TestBackendClientHeartbeat:
    def test_heartbeat_success(self, mock_wallet):
        from oro_sdk.models.heartbeat_response import HeartbeatResponse

        client = BackendClient("https://api.example.com", mock_wallet)

        sdk_response = HeartbeatResponse(
            lease_expires_at=datetime(2025, 1, 13, 12, 30, 0),
        )

        mock_response = _create_response(200, sdk_response)

        with patch(
            "validator.backend_client.heartbeat.sync_detailed",
            return_value=mock_response,
        ):
            result = client.heartbeat("12345678-1234-1234-1234-123456789012")

        assert result.lease_expires_at is not None

    def test_heartbeat_lease_expired(self, mock_wallet):
        client = BackendClient("https://api.example.com", mock_wallet)

        # Create a mock error response with detail attribute
        mock_error = MagicMock()
        mock_error.detail = "Lease expired"
        mock_response = _create_response(409, mock_error)

        with patch(
            "validator.backend_client.heartbeat.sync_detailed",
            return_value=mock_response,
        ):
            with pytest.raises(BackendError) as exc_info:
                client.heartbeat("12345678-1234-1234-1234-123456789012")

            assert exc_info.value.is_conflict
            assert exc_info.value.status_code == 409


class TestBackendClientProgress:
    def test_report_progress_success(self, mock_wallet):
        from oro_sdk.models.progress_update_response import ProgressUpdateResponse

        client = BackendClient("https://api.example.com", mock_wallet)

        sdk_response = ProgressUpdateResponse(accepted=True)
        mock_response = _create_response(200, sdk_response)

        problems = [
            ProblemProgressUpdate(
                problem_id=UUID("11111111-1111-1111-1111-111111111111"),
                status=ProblemStatus.SUCCESS,
            ),
            ProblemProgressUpdate(
                problem_id=UUID("22222222-2222-2222-2222-222222222222"),
                status=ProblemStatus.RUNNING,
            ),
        ]

        with patch(
            "validator.backend_client.update_progress.sync_detailed",
            return_value=mock_response,
        ):
            # Should not raise
            client.report_progress("12345678-1234-1234-1234-123456789012", problems)


class TestBackendClientComplete:
    def test_complete_run_success(self, mock_wallet):
        from oro_sdk.models.complete_run_response import CompleteRunResponse
        from oro_sdk.models.work_item_status import WorkItemStatus

        client = BackendClient("https://api.example.com", mock_wallet)

        work_item = WorkItemStatus(
            included_success_count=1,
            required_successes=3,
            is_closed=False,
        )

        sdk_response = CompleteRunResponse(
            eval_run_id=UUID("12345678-1234-1234-1234-123456789012"),
            status="SUCCESS",
            work_item=work_item,
            agent_version_became_eligible=True,
            final_score=0.85,
        )

        mock_response = _create_response(200, sdk_response)

        with patch(
            "validator.backend_client.complete_run.sync_detailed",
            return_value=mock_response,
        ):
            result = client.complete_run(
                eval_run_id="12345678-1234-1234-1234-123456789012",
                status="SUCCESS",
                score=0.85,
                score_components={"accuracy": 0.9},
                results_s3_key="logs/run-123.tar.gz",
            )

        assert result.eval_run_id == UUID("12345678-1234-1234-1234-123456789012")
        assert result.agent_version_became_eligible is True


class TestBackendClientPresign:
    def test_get_presigned_upload_url(self, mock_wallet):
        from oro_sdk.models.presign_upload_response import PresignUploadResponse

        client = BackendClient("https://api.example.com", mock_wallet)

        sdk_response = PresignUploadResponse(
            upload_url="https://s3.example.com/presigned",
            results_s3_key="logs/run-123.tar.gz",
            expires_at=datetime(2025, 1, 13, 12, 15, 0),
            method="PUT",
        )

        mock_response = _create_response(200, sdk_response)

        with patch(
            "validator.backend_client.presign_upload.sync_detailed",
            return_value=mock_response,
        ):
            result = client.get_presigned_upload_url(
                eval_run_id=UUID("12345678-1234-1234-1234-123456789012"),
                problem_id=UUID("00000000-0000-0000-0000-000000000000"),
                content_type="application/gzip",
                content_length=1024,
            )

        assert result.upload_url == "https://s3.example.com/presigned"
        assert result.results_s3_key == "logs/run-123.tar.gz"


class TestBackendClientTopMiner:
    def test_get_top_miner(self, mock_wallet):
        from oro_sdk.models.top_agent_response import TopAgentResponse

        client = BackendClient("https://api.example.com", mock_wallet)

        sdk_response = TopAgentResponse(
            suite_id=789,
            computed_at=datetime(2025, 1, 13, 12, 0, 0),
            top_agent_version_id=UUID("87654321-4321-4321-4321-210987654321"),
            top_miner_hotkey="5GrwvaEF...",
            top_score=0.92,
        )

        mock_response = _create_response(200, sdk_response)

        with patch(
            "validator.backend_client.get_top_agent.sync_detailed",
            return_value=mock_response,
        ):
            result = client.get_top_miner()

        assert result.top_miner_hotkey == "5GrwvaEF..."
        assert result.top_score == 0.92


class TestBackendClientUploadToS3:
    def test_upload_to_s3_success(self, mock_wallet):
        from oro_sdk.models import PresignUploadResponse

        client = BackendClient("https://api.example.com", mock_wallet)
        presign = PresignUploadResponse(
            upload_url="https://s3.example.com/presigned",
            method="PUT",
            results_s3_key="logs/run-123.tar.gz",
            expires_at=datetime.now(),
        )

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch(
            "validator.backend_client.requests.request", return_value=mock_response
        ):
            client.upload_to_s3(presign, b"test data")  # Should not raise

    def test_upload_to_s3_failure(self, mock_wallet):
        from oro_sdk.models import PresignUploadResponse

        client = BackendClient("https://api.example.com", mock_wallet)
        presign = PresignUploadResponse(
            upload_url="https://s3.example.com/presigned",
            method="PUT",
            results_s3_key="logs/run-123.tar.gz",
            expires_at=datetime.now(),
        )

        mock_response = MagicMock()
        mock_response.status_code = 403

        with patch(
            "validator.backend_client.requests.request", return_value=mock_response
        ):
            with pytest.raises(BackendError):
                client.upload_to_s3(presign, b"test data")


class TestBackendClientErrorHandling:
    def test_server_error_is_transient(self, mock_wallet):
        """500 errors are transient and should be retried."""
        client = BackendClient("https://api.example.com", mock_wallet)

        mock_response = _create_response(500, None)

        with patch(
            "validator.backend_client.claim_work.sync_detailed",
            return_value=mock_response,
        ):
            with pytest.raises(BackendError) as exc_info:
                client.claim_work()

            assert exc_info.value.is_transient
            assert exc_info.value.status_code == 500

    def test_timeout_is_transient(self, mock_wallet):
        """Timeout errors are transient and should be retried."""
        import httpx

        client = BackendClient("https://api.example.com", mock_wallet)

        with patch(
            "validator.backend_client.claim_work.sync_detailed",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            with pytest.raises(BackendError) as exc_info:
                client.claim_work()

            assert exc_info.value.is_transient
            assert exc_info.value.status_code is None

    def test_connection_error_is_transient(self, mock_wallet):
        """Connection errors are transient and should be retried."""
        import httpx

        client = BackendClient("https://api.example.com", mock_wallet)

        with patch(
            "validator.backend_client.claim_work.sync_detailed",
            side_effect=httpx.ConnectError("connection refused"),
        ):
            with pytest.raises(BackendError) as exc_info:
                client.claim_work()

            assert exc_info.value.is_transient
            assert exc_info.value.status_code is None

    def test_sdk_parse_error_surfaces_actionable_message(self, mock_wallet):
        """SDK response parsing failures (e.g. KeyError('loc')) raise a
        descriptive BackendError instead of bubbling up an opaque KeyError.
        """
        client = BackendClient("https://api.example.com", mock_wallet)

        with patch(
            "validator.backend_client.claim_work.sync_detailed",
            side_effect=KeyError("loc"),
        ):
            with pytest.raises(BackendError) as exc_info:
                client.claim_work()

            msg = str(exc_info.value)
            assert "Response parsing failed" in msg
            assert "KeyError" in msg
            assert "loc" in msg

    def test_auth_error_detection(self, mock_wallet):
        """401/403 errors are detected as auth errors."""
        client = BackendClient("https://api.example.com", mock_wallet)

        mock_response = _create_response(401, None)

        with patch(
            "validator.backend_client.claim_work.sync_detailed",
            return_value=mock_response,
        ):
            with pytest.raises(BackendError) as exc_info:
                client.claim_work()

            assert exc_info.value.is_auth_error
            assert not exc_info.value.is_transient

    def test_not_found_detection(self, mock_wallet):
        """404 errors are detected via is_not_found."""
        client = BackendClient("https://api.example.com", mock_wallet)

        # Create a mock error response with detail attribute
        mock_error = MagicMock()
        mock_error.detail = "Run not found"
        mock_response = _create_response(404, mock_error)

        with patch(
            "validator.backend_client.heartbeat.sync_detailed",
            return_value=mock_response,
        ):
            with pytest.raises(BackendError) as exc_info:
                client.heartbeat("12345678-1234-1234-1234-123456789012")

            assert exc_info.value.is_not_found
            assert not exc_info.value.is_transient

    def test_error_code_from_sdk_error(self, mock_wallet):
        """error_code property extracts from typed SDK error."""

        client = BackendClient("https://api.example.com", mock_wallet)

        # Create a mock SDK error with error_code attribute
        sdk_error = MagicMock()
        sdk_error.detail = "Lease expired"
        sdk_error.error_code = "LEASE_EXPIRED"

        mock_response = _create_response(409, sdk_error)

        with patch(
            "validator.backend_client.heartbeat.sync_detailed",
            return_value=mock_response,
        ):
            with pytest.raises(BackendError) as exc_info:
                client.heartbeat("12345678-1234-1234-1234-123456789012")

            assert exc_info.value.error_code == "LEASE_EXPIRED"
            assert exc_info.value.is_conflict


class TestBackendErrorTypedErrors:
    """Tests for SDK typed error detection."""

    def test_is_lease_expired_with_typed_error(self, mock_wallet):
        """is_lease_expired returns True for LeaseExpiredError."""
        from validator.backend_client import LeaseExpiredError

        client = BackendClient("https://api.example.com", mock_wallet)
        sdk_error = LeaseExpiredError(detail="Lease has expired")
        mock_response = _create_response(409, sdk_error)

        with patch(
            "validator.backend_client.heartbeat.sync_detailed",
            return_value=mock_response,
        ):
            with pytest.raises(BackendError) as exc_info:
                client.heartbeat("12345678-1234-1234-1234-123456789012")

            assert exc_info.value.is_lease_expired
            assert exc_info.value.is_error(LeaseExpiredError)
            assert not exc_info.value.is_at_capacity

    def test_is_at_capacity_with_typed_error(self, mock_wallet):
        """is_at_capacity returns True for AtCapacityError."""
        from validator.backend_client import AtCapacityError

        client = BackendClient("https://api.example.com", mock_wallet)
        sdk_error = AtCapacityError(detail="Validator at capacity")
        mock_response = _create_response(409, sdk_error)

        with patch(
            "validator.backend_client.claim_work.sync_detailed",
            return_value=mock_response,
        ):
            with pytest.raises(BackendError) as exc_info:
                client.claim_work()

            assert exc_info.value.is_at_capacity
            assert exc_info.value.is_error(AtCapacityError)
            assert not exc_info.value.is_lease_expired

    def test_is_not_run_owner_with_typed_error(self, mock_wallet):
        """is_not_run_owner returns True for NotRunOwnerError."""
        from validator.backend_client import NotRunOwnerError

        client = BackendClient("https://api.example.com", mock_wallet)
        sdk_error = NotRunOwnerError(detail="Not run owner")
        mock_response = _create_response(403, sdk_error)

        with patch(
            "validator.backend_client.heartbeat.sync_detailed",
            return_value=mock_response,
        ):
            with pytest.raises(BackendError) as exc_info:
                client.heartbeat("12345678-1234-1234-1234-123456789012")

            assert exc_info.value.is_not_run_owner
            assert exc_info.value.is_error(NotRunOwnerError)

    def test_is_run_already_complete_with_typed_error(self, mock_wallet):
        """is_run_already_complete returns True for RunAlreadyCompleteError."""
        from validator.backend_client import RunAlreadyCompleteError

        client = BackendClient("https://api.example.com", mock_wallet)
        sdk_error = RunAlreadyCompleteError(detail="Run already complete")
        mock_response = _create_response(409, sdk_error)

        with patch(
            "validator.backend_client.complete_run.sync_detailed",
            return_value=mock_response,
        ):
            with pytest.raises(BackendError) as exc_info:
                client.complete_run(
                    eval_run_id="12345678-1234-1234-1234-123456789012",
                    status="SUCCESS",
                    score=0.85,
                    score_components={},
                )

            assert exc_info.value.is_run_already_complete
            assert exc_info.value.is_error(RunAlreadyCompleteError)

    def test_is_eval_run_not_found_with_typed_error(self, mock_wallet):
        """is_eval_run_not_found returns True for EvalRunNotFoundError."""
        from validator.backend_client import EvalRunNotFoundError

        client = BackendClient("https://api.example.com", mock_wallet)
        sdk_error = EvalRunNotFoundError(detail="Eval run not found")
        mock_response = _create_response(404, sdk_error)

        with patch(
            "validator.backend_client.heartbeat.sync_detailed",
            return_value=mock_response,
        ):
            with pytest.raises(BackendError) as exc_info:
                client.heartbeat("12345678-1234-1234-1234-123456789012")

            assert exc_info.value.is_eval_run_not_found
            assert exc_info.value.is_error(EvalRunNotFoundError)
            assert exc_info.value.is_not_found  # Also true via status code

    def test_is_error_returns_false_for_wrong_type(self, mock_wallet):
        """is_error returns False when sdk_error is different type."""
        from validator.backend_client import AtCapacityError, LeaseExpiredError

        client = BackendClient("https://api.example.com", mock_wallet)
        sdk_error = AtCapacityError(detail="At capacity")
        mock_response = _create_response(409, sdk_error)

        with patch(
            "validator.backend_client.claim_work.sync_detailed",
            return_value=mock_response,
        ):
            with pytest.raises(BackendError) as exc_info:
                client.claim_work()

            assert exc_info.value.is_error(AtCapacityError)
            assert not exc_info.value.is_error(LeaseExpiredError)

    def test_is_error_returns_false_when_no_sdk_error(self, mock_wallet):
        """is_error returns False when sdk_error is None."""
        from validator.backend_client import LeaseExpiredError

        client = BackendClient("https://api.example.com", mock_wallet)
        mock_response = _create_response(500, None)

        with patch(
            "validator.backend_client.claim_work.sync_detailed",
            return_value=mock_response,
        ):
            with pytest.raises(BackendError) as exc_info:
                client.claim_work()

            assert not exc_info.value.is_error(LeaseExpiredError)
            assert not exc_info.value.is_lease_expired
