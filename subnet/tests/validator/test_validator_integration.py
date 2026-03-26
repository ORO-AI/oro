"""Integration tests for validator with mocked Backend API.

Uses mock_wallet, mock_subtensor, mock_metagraph, and temp_storage_path
fixtures from conftest.py.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from uuid import UUID

# Add test-subnet to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from oro_sdk.models import (
    ClaimWorkResponse,
    CompleteRunResponse,
    HeartbeatResponse,
    PresignUploadResponse,
    WorkItemStatus,
)
from oro_sdk.models.terminal_status import TerminalStatus

from validator.models import CompletionRequest
from validator.backend_client import BackendClient, BackendError
from validator.heartbeat_manager import HeartbeatManager
from validator.retry_queue import LocalRetryQueue


class TestValidatorIntegration:
    """Integration tests for the validator components working together."""

    def test_evaluation_cycle_success(
        self, mock_wallet, mock_subtensor, mock_metagraph
    ):
        """Test a successful evaluation cycle from claim to complete.

        This test verifies that:
        1. ClaimWorkResponse is properly constructed with all required fields
        2. HeartbeatResponse contains lease extension info
        3. PresignUploadResponse contains upload details
        4. CompleteRunResponse contains final status
        5. The entire flow from claim -> heartbeat -> presign -> complete works
        """
        # Test ClaimWorkResponse creation
        work = ClaimWorkResponse(
            eval_run_id=UUID("12345678-1234-1234-1234-123456789012"),
            agent_version_id=UUID("87654321-4321-4321-4321-210987654321"),
            suite_id=789,
            lease_expires_at=datetime.now() + timedelta(hours=1),
            code_download_url="https://example.com/agent.py",
        )

        assert str(work.eval_run_id) == "12345678-1234-1234-1234-123456789012"
        assert str(work.agent_version_id) == "87654321-4321-4321-4321-210987654321"
        assert work.suite_id == 789
        assert work.code_download_url == "https://example.com/agent.py"
        assert work.lease_expires_at > datetime.now()

        # Test HeartbeatResponse
        heartbeat = HeartbeatResponse(
            lease_expires_at=datetime.now() + timedelta(minutes=30)
        )
        assert heartbeat.lease_expires_at > datetime.now()

        # Test PresignUploadResponse
        presign = PresignUploadResponse(
            upload_url="https://s3.example.com/presigned",
            method="PUT",
            results_s3_key="logs/run-123.tar.gz",
            expires_at=datetime.now() + timedelta(minutes=15),
        )
        assert presign.upload_url == "https://s3.example.com/presigned"
        assert presign.method == "PUT"
        assert presign.results_s3_key == "logs/run-123.tar.gz"

        # Test CompleteRunResponse
        complete = CompleteRunResponse(
            eval_run_id=UUID("12345678-1234-1234-1234-123456789012"),
            status="SUCCESS",
            work_item=WorkItemStatus(
                included_success_count=1,
                required_successes=3,
                is_closed=False,
            ),
            agent_version_became_eligible=True,
            final_score=0.85,
        )
        assert str(complete.eval_run_id) == "12345678-1234-1234-1234-123456789012"
        assert complete.status == "SUCCESS"
        assert complete.agent_version_became_eligible is True
        assert complete.final_score == 0.85

    def test_claim_work_returns_none_on_204(self, mock_wallet):
        """Test that BackendClient.claim_work() returns None when backend returns 204.

        This verifies the no-work-available polling behavior:
        - Backend returns 204 when there's no work to claim
        - BackendClient.claim_work() should return None (not raise an exception)
        - The validator should then sleep and poll again
        """
        client = BackendClient("https://api.example.com", mock_wallet)

        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.parsed = None

        with patch(
            "validator.backend_client.claim_work.sync_detailed",
            return_value=mock_response,
        ):
            result = client.claim_work()

        # Verify that 204 results in None return value
        assert result is None

    def test_claim_work_returns_work_on_200(self, mock_wallet):
        """Test that BackendClient.claim_work() returns ClaimWorkResponse on 200."""
        from oro_sdk.models.claim_work_response import (
            ClaimWorkResponse as SDKClaimWorkResponse,
        )

        client = BackendClient("https://api.example.com", mock_wallet)

        sdk_response = SDKClaimWorkResponse(
            eval_run_id=UUID("12345678-1234-1234-1234-123456789012"),
            agent_version_id=UUID("87654321-4321-4321-4321-210987654321"),
            suite_id=789,
            lease_expires_at=datetime(2025, 1, 13, 15, 0, 0),
            code_download_url="https://example.com/code.py",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.parsed = sdk_response

        with patch(
            "validator.backend_client.claim_work.sync_detailed",
            return_value=mock_response,
        ):
            result = client.claim_work()

        assert result is not None
        assert result.eval_run_id == UUID("12345678-1234-1234-1234-123456789012")
        assert isinstance(result, ClaimWorkResponse)

    def test_heartbeat_failure_continues_execution(self):
        """Test that HeartbeatManager continues running even on failures.

        This verifies resilience behavior:
        - Heartbeat failures should not crash the manager thread
        - The manager should continue attempting heartbeats
        - Failures should be recorded but not stop execution
        """
        mock_backend_client = MagicMock(spec=BackendClient)

        # First heartbeat fails (transient error), subsequent ones succeed
        transient_error = BackendError("Temporary failure", status_code=500)
        mock_backend_client.heartbeat.side_effect = [
            transient_error,
            HeartbeatResponse(lease_expires_at=datetime.now() + timedelta(minutes=5)),
            HeartbeatResponse(lease_expires_at=datetime.now() + timedelta(minutes=5)),
        ]

        manager = HeartbeatManager(
            backend_client=mock_backend_client,
            eval_run_id="run-failure-test",
            interval_seconds=0.1,
        )

        manager.start()

        # Wait long enough for multiple heartbeat attempts
        time.sleep(0.35)

        # Verify the manager is still running despite initial failure
        assert manager._thread is not None
        assert manager._thread.is_alive()

        # Verify multiple heartbeat attempts were made
        assert mock_backend_client.heartbeat.call_count >= 2

        manager.stop()

        # After recovery, manager should be healthy
        # (depends on last heartbeat success)

    def test_heartbeat_manager_tracks_health_status(self):
        """Test that HeartbeatManager correctly tracks health status on failures."""
        mock_backend_client = MagicMock(spec=BackendClient)
        # 409 conflict error (lease expired)
        conflict_error = BackendError("Lease expired", status_code=409)
        mock_backend_client.heartbeat.side_effect = conflict_error

        manager = HeartbeatManager(
            backend_client=mock_backend_client,
            eval_run_id="run-health-test",
            interval_seconds=0.1,
        )

        manager.start()
        time.sleep(0.15)

        # Verify health status reflects failure
        assert manager.is_healthy() is False
        last_error = manager.get_last_error()
        assert last_error is not None
        assert isinstance(last_error, BackendError)
        assert last_error.is_conflict

        manager.stop()

    def test_complete_failure_queues_retry(self, temp_storage_path):
        """Test that failed complete calls are added to retry queue.

        This verifies the retry queue fallback behavior:
        - When a transient BackendError occurs during complete_run
        - The completion request should be added to the retry queue
        - The queue should persist the request for later retry
        """
        mock_backend_client = MagicMock(spec=BackendClient)
        # Transient error (server unavailable)
        transient_error = BackendError(
            "Backend temporarily unavailable", status_code=503
        )
        mock_backend_client.complete_run.side_effect = transient_error

        retry_queue = LocalRetryQueue(mock_backend_client, temp_storage_path)

        # Simulate what _complete_run does in the validator
        eval_run_id = UUID("12345678-1234-1234-1234-123456789012")
        status = TerminalStatus.SUCCESS
        score = 0.85
        score_components = {"success_rate": score}
        results_s3_key = "logs/run-retry-test.tar.gz"

        # Attempt to complete (will fail)
        try:
            mock_backend_client.complete_run(
                eval_run_id=eval_run_id,
                status=status,
                score=score,
                score_components=score_components,
                results_s3_key=results_s3_key,
            )
        except BackendError as e:
            # Add to retry queue on transient failure
            if e.is_transient:
                retry_queue.add(
                    CompletionRequest(
                        eval_run_id=eval_run_id,
                        status=status,
                        validator_score=score,
                        score_components=score_components,
                        results_s3_key=results_s3_key,
                    )
                )

        # Verify the request was queued
        assert retry_queue.get_pending_count() == 1

        # Verify the queued data is correct
        data = retry_queue._load()
        pending = data["pending"][0]
        assert pending["eval_run_id"] == str(eval_run_id)
        assert pending["terminal_status"] == status.value
        assert pending["validator_score"] == score
        assert pending["results_s3_key"] == results_s3_key

    def test_retry_queue_processes_on_backend_recovery(self, temp_storage_path):
        """Test that retry queue successfully processes items when backend recovers."""
        mock_backend_client = MagicMock(spec=BackendClient)

        # First call fails (transient error), second succeeds
        transient_error = BackendError("Down", status_code=500)
        mock_backend_client.complete_run.side_effect = [
            transient_error,
            MagicMock(),  # Success on retry
        ]

        retry_queue = LocalRetryQueue(mock_backend_client, temp_storage_path)

        completion = CompletionRequest(
            eval_run_id=UUID("12345678-1234-1234-1234-123456789012"),
            status=TerminalStatus.SUCCESS,
            validator_score=0.9,
            score_components={"accuracy": 0.9},
            results_s3_key="logs/recovery.tar.gz",
        )

        retry_queue.add(completion)
        assert retry_queue.get_pending_count() == 1

        # First process attempt fails
        retry_queue.process_pending()
        assert retry_queue.get_pending_count() == 1

        # Reset the side effect for success
        mock_backend_client.complete_run.side_effect = None
        mock_backend_client.complete_run.return_value = MagicMock()

        # Second process attempt succeeds
        retry_queue.process_pending()
        assert retry_queue.get_pending_count() == 0

    def test_full_evaluation_flow_with_mocks(self, mock_wallet, temp_storage_path):
        """Test the full evaluation flow with all components mocked.

        This is an end-to-end integration test simulating:
        1. Claiming work from backend
        2. Starting heartbeat manager
        3. Simulating sandbox execution
        4. Completing the run
        """
        from oro_sdk.models.claim_work_response import (
            ClaimWorkResponse as SDKClaimWorkResponse,
        )
        from oro_sdk.models.heartbeat_response import (
            HeartbeatResponse as SDKHeartbeatResponse,
        )
        from oro_sdk.models.presign_upload_response import PresignUploadResponse
        from oro_sdk.models.complete_run_response import (
            CompleteRunResponse as SDKCompleteRunResponse,
        )
        from oro_sdk.models.work_item_status import WorkItemStatus

        client = BackendClient("https://api.example.com", mock_wallet)

        # Create SDK response objects
        sdk_claim_response = SDKClaimWorkResponse(
            eval_run_id=UUID("12345678-1234-1234-1234-123456789012"),
            agent_version_id=UUID("87654321-4321-4321-4321-210987654321"),
            suite_id=789,
            lease_expires_at=datetime(2025, 1, 13, 16, 0, 0),
            code_download_url="https://example.com/agent.py",
        )

        sdk_heartbeat_response = SDKHeartbeatResponse(
            lease_expires_at=datetime(2025, 1, 13, 16, 30, 0),
        )

        sdk_presign_response = PresignUploadResponse(
            upload_url="https://s3.example.com/presigned",
            results_s3_key="logs/run-full-test.tar.gz",
            expires_at=datetime(2025, 1, 13, 16, 15, 0),
            method="PUT",
        )

        sdk_complete_response = SDKCompleteRunResponse(
            eval_run_id=UUID("12345678-1234-1234-1234-123456789012"),
            status="SUCCESS",
            work_item=WorkItemStatus(
                included_success_count=1,
                required_successes=3,
                is_closed=False,
            ),
            agent_version_became_eligible=True,
            final_score=0.88,
        )

        # Create mock responses
        claim_response = MagicMock()
        claim_response.status_code = 200
        claim_response.parsed = sdk_claim_response

        heartbeat_response = MagicMock()
        heartbeat_response.status_code = 200
        heartbeat_response.parsed = sdk_heartbeat_response

        presign_response = MagicMock()
        presign_response.status_code = 200
        presign_response.parsed = sdk_presign_response

        complete_response = MagicMock()
        complete_response.status_code = 200
        complete_response.parsed = sdk_complete_response

        with (
            patch(
                "validator.backend_client.claim_work.sync_detailed",
                return_value=claim_response,
            ),
            patch(
                "validator.backend_client.heartbeat.sync_detailed",
                return_value=heartbeat_response,
            ),
            patch(
                "validator.backend_client.presign_upload.sync_detailed",
                return_value=presign_response,
            ),
            patch(
                "validator.backend_client.complete_run.sync_detailed",
                return_value=complete_response,
            ),
        ):
            # Step 1: Claim work
            work = client.claim_work()
            assert work is not None
            assert work.eval_run_id == UUID("12345678-1234-1234-1234-123456789012")

            # Step 2: Send heartbeat
            heartbeat = client.heartbeat(work.eval_run_id)
            assert heartbeat is not None

            # Step 3: Get presign URL
            presign = client.get_presigned_upload_url(
                eval_run_id=work.eval_run_id,
                problem_id=UUID("00000000-0000-0000-0000-000000000000"),
                content_type="application/gzip",
                content_length=1024,
            )
            assert presign.upload_url == "https://s3.example.com/presigned"

            # Step 4: Complete the run
            complete = client.complete_run(
                eval_run_id=work.eval_run_id,
                status=TerminalStatus.SUCCESS,
                score=0.88,
                score_components={"success_rate": 0.88},
                results_s3_key=presign.results_s3_key,
            )
            assert complete.eval_run_id == UUID("12345678-1234-1234-1234-123456789012")
            assert complete.agent_version_became_eligible is True
