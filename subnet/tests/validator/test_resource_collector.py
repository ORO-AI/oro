"""Tests for resource_collector — host metric snapshotting."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validator.resource_collector import collect_resource_metrics


@pytest.fixture
def patched_psutil():
    """All four collectors stubbed with healthy defaults; tests override per-call."""
    with (
        patch("validator.resource_collector.psutil.cpu_percent", return_value=42.5) as cpu,
        patch(
            "validator.resource_collector.psutil.virtual_memory",
            return_value=MagicMock(percent=31.0),
        ) as mem,
        patch(
            "validator.resource_collector.psutil.disk_usage",
            return_value=MagicMock(percent=18.7),
        ) as disk,
        patch(
            "validator.resource_collector.subprocess.run",
            return_value=MagicMock(returncode=0, stdout="abc\ndef\n123\n"),
        ) as docker,
    ):
        yield {"cpu": cpu, "mem": mem, "disk": disk, "docker": docker}


class TestCollectResourceMetrics:
    def test_happy_path(self, patched_psutil):
        assert collect_resource_metrics() == {
            "cpu_pct": 42.5,
            "ram_pct": 31.0,
            "disk_pct": 18.7,
            "docker_container_count": 3,
        }

    def test_individual_failures_omit_field(self, patched_psutil):
        patched_psutil["cpu"].side_effect = OSError
        patched_psutil["disk"].side_effect = PermissionError
        patched_psutil["docker"].side_effect = FileNotFoundError
        patched_psutil["mem"].return_value = MagicMock(percent=20.0)

        assert collect_resource_metrics() == {"ram_pct": 20.0}

    def test_docker_empty_output_is_zero(self, patched_psutil):
        patched_psutil["docker"].return_value = MagicMock(returncode=0, stdout="")
        assert collect_resource_metrics()["docker_container_count"] == 0

    def test_docker_nonzero_returncode_omits_count(self, patched_psutil):
        patched_psutil["docker"].return_value = MagicMock(returncode=1, stdout="")
        assert "docker_container_count" not in collect_resource_metrics()

    def test_disk_uses_sandbox_dir_env(self, patched_psutil, monkeypatch):
        monkeypatch.setenv("SANDBOX_DIR", "/var/lib/sandbox")
        collect_resource_metrics()
        patched_psutil["disk"].assert_called_once_with("/var/lib/sandbox")
