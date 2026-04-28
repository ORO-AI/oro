"""Tests for resource_collector — host metric snapshotting."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validator.resource_collector import collect_resource_metrics


class TestCollectResourceMetrics:
    def test_returns_all_metrics_when_everything_succeeds(self):
        """Happy path: all four metrics present and well-typed."""
        with (
            patch("validator.resource_collector.psutil.cpu_percent", return_value=42.5),
            patch(
                "validator.resource_collector.psutil.virtual_memory",
                return_value=MagicMock(percent=31.0),
            ),
            patch(
                "validator.resource_collector.psutil.disk_usage",
                return_value=MagicMock(percent=18.7),
            ),
            patch(
                "validator.resource_collector.subprocess.run",
                return_value=MagicMock(returncode=0, stdout="abc\ndef\n123\n"),
            ),
        ):
            metrics = collect_resource_metrics()

        assert metrics == {
            "cpu_pct": 42.5,
            "ram_pct": 31.0,
            "disk_pct": 18.7,
            "docker_container_count": 3,
        }

    def test_omits_field_when_individual_collector_fails(self):
        """If psutil raises for one metric, the dict omits it instead of failing."""
        with (
            patch("validator.resource_collector.psutil.cpu_percent", side_effect=OSError),
            patch(
                "validator.resource_collector.psutil.virtual_memory",
                return_value=MagicMock(percent=20.0),
            ),
            patch(
                "validator.resource_collector.psutil.disk_usage",
                side_effect=PermissionError,
            ),
            patch(
                "validator.resource_collector.subprocess.run",
                side_effect=FileNotFoundError,
            ),
        ):
            metrics = collect_resource_metrics()

        assert metrics == {"ram_pct": 20.0}

    def test_docker_count_handles_empty_output(self):
        """`docker ps -q` with no containers returns count = 0, not None."""
        with (
            patch("validator.resource_collector.psutil.cpu_percent", return_value=0.0),
            patch(
                "validator.resource_collector.psutil.virtual_memory",
                return_value=MagicMock(percent=0.0),
            ),
            patch(
                "validator.resource_collector.psutil.disk_usage",
                return_value=MagicMock(percent=0.0),
            ),
            patch(
                "validator.resource_collector.subprocess.run",
                return_value=MagicMock(returncode=0, stdout=""),
            ),
        ):
            metrics = collect_resource_metrics()

        assert metrics["docker_container_count"] == 0

    def test_docker_returncode_nonzero_omits_count(self):
        """Non-zero docker exit (e.g. permission denied) omits the field."""
        with (
            patch("validator.resource_collector.psutil.cpu_percent", return_value=10.0),
            patch(
                "validator.resource_collector.psutil.virtual_memory",
                return_value=MagicMock(percent=10.0),
            ),
            patch(
                "validator.resource_collector.psutil.disk_usage",
                return_value=MagicMock(percent=10.0),
            ),
            patch(
                "validator.resource_collector.subprocess.run",
                return_value=MagicMock(returncode=1, stdout=""),
            ),
        ):
            metrics = collect_resource_metrics()

        assert "docker_container_count" not in metrics

    def test_uses_sandbox_dir_env_for_disk_path(self, monkeypatch):
        """Disk usage is sampled against $SANDBOX_DIR when set."""
        monkeypatch.setenv("SANDBOX_DIR", "/var/lib/sandbox")
        with (
            patch("validator.resource_collector.psutil.cpu_percent", return_value=0.0),
            patch(
                "validator.resource_collector.psutil.virtual_memory",
                return_value=MagicMock(percent=0.0),
            ),
            patch(
                "validator.resource_collector.psutil.disk_usage",
                return_value=MagicMock(percent=55.5),
            ) as disk_usage,
            patch(
                "validator.resource_collector.subprocess.run",
                return_value=MagicMock(returncode=0, stdout=""),
            ),
        ):
            collect_resource_metrics()

        disk_usage.assert_called_once_with("/var/lib/sandbox")
