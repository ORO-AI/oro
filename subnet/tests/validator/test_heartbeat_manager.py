import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add test-subnet to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from oro_sdk.models import HeartbeatResponse

from validator.backend_client import BackendError
from validator.heartbeat_manager import HeartbeatManager


class TestHeartbeatManager:
    """Tests for HeartbeatManager.

    Uses mock_backend_client_with_heartbeat fixture from conftest.py.
    """

    @pytest.fixture
    def mock_backend_client(self, mock_backend_client_with_heartbeat):
        """Alias to use the pre-configured heartbeat fixture."""
        return mock_backend_client_with_heartbeat

    def test_start_creates_thread(self, mock_backend_client):
        manager = HeartbeatManager(
            backend_client=mock_backend_client,
            eval_run_id="run-123",
            interval_seconds=1,
        )
        manager.start()
        assert manager._thread is not None
        assert manager._thread.is_alive()
        manager.stop()

    def test_stop_terminates_thread(self, mock_backend_client):
        manager = HeartbeatManager(
            backend_client=mock_backend_client,
            eval_run_id="run-123",
            interval_seconds=1,
        )
        manager.start()
        manager.stop()
        assert not manager._thread.is_alive()

    def test_heartbeat_called_periodically(self, mock_backend_client):
        manager = HeartbeatManager(
            backend_client=mock_backend_client,
            eval_run_id="run-123",
            interval_seconds=0.1,
        )
        manager.start()
        time.sleep(0.35)
        manager.stop()
        assert mock_backend_client.heartbeat.call_count >= 2

    def test_is_healthy_true_on_success(self, mock_backend_client):
        manager = HeartbeatManager(
            backend_client=mock_backend_client,
            eval_run_id="run-123",
            interval_seconds=0.1,
        )
        manager.start()
        time.sleep(0.15)
        assert manager.is_healthy() is True
        manager.stop()

    def test_is_healthy_false_on_error(self, mock_backend_client):
        # 409 conflict error (lease expired)
        conflict_error = BackendError("Lease expired", status_code=409)
        mock_backend_client.heartbeat.side_effect = conflict_error
        manager = HeartbeatManager(
            backend_client=mock_backend_client,
            eval_run_id="run-123",
            interval_seconds=0.1,
        )
        manager.start()
        time.sleep(0.15)
        assert manager.is_healthy() is False
        last_error = manager.get_last_error()
        assert last_error is not None
        assert isinstance(last_error, BackendError)
        manager.stop()

    def test_continues_on_transient_error(self, mock_backend_client):
        mock_backend_client.heartbeat.side_effect = [
            Exception("Network error"),
            HeartbeatResponse(lease_expires_at=datetime.now() + timedelta(minutes=5)),
            HeartbeatResponse(lease_expires_at=datetime.now() + timedelta(minutes=5)),
        ]
        manager = HeartbeatManager(
            backend_client=mock_backend_client,
            eval_run_id="run-123",
            interval_seconds=0.1,
        )
        manager.start()
        time.sleep(0.35)
        manager.stop()
        assert mock_backend_client.heartbeat.call_count >= 2

    def test_passes_service_versions_to_heartbeat(self, mock_backend_client):
        versions = {"search-server": "sha256:abc123def4", "proxy": "sha256:def456abc7"}
        manager = HeartbeatManager(
            backend_client=mock_backend_client,
            eval_run_id="run-123",
            interval_seconds=0.1,
            service_versions=versions,
        )
        manager.start()
        time.sleep(0.15)
        manager.stop()
        mock_backend_client.heartbeat.assert_called_with(
            "run-123", service_versions=versions, resource_metrics=None
        )

    def test_no_service_versions_by_default(self, mock_backend_client):
        manager = HeartbeatManager(
            backend_client=mock_backend_client,
            eval_run_id="run-123",
            interval_seconds=0.1,
        )
        manager.start()
        time.sleep(0.15)
        manager.stop()
        mock_backend_client.heartbeat.assert_called_with(
            "run-123", service_versions=None, resource_metrics=None
        )

    def test_passes_resource_metrics_to_heartbeat(self, mock_backend_client):
        metrics = {"cpu_pct": 12.5, "ram_pct": 30.0}
        manager = HeartbeatManager(
            backend_client=mock_backend_client,
            eval_run_id="run-123",
            interval_seconds=0.1,
            resource_metrics_provider=lambda: metrics,
        )
        manager.start()
        time.sleep(0.15)
        manager.stop()
        mock_backend_client.heartbeat.assert_called_with(
            "run-123", service_versions=None, resource_metrics=metrics
        )

    def test_resource_metrics_provider_failure_does_not_break_heartbeat(
        self, mock_backend_client
    ):
        def boom():
            raise RuntimeError("psutil exploded")

        manager = HeartbeatManager(
            backend_client=mock_backend_client,
            eval_run_id="run-123",
            interval_seconds=0.1,
            resource_metrics_provider=boom,
        )
        manager.start()
        time.sleep(0.15)
        manager.stop()
        mock_backend_client.heartbeat.assert_called_with(
            "run-123", service_versions=None, resource_metrics=None
        )
        assert manager.is_healthy()
