"""Tests for version_collector module."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validator.version_collector import (
    _get_container_image_digest,
    _get_image_digest,
    _shorten_digest,
    collect_service_versions,
)


class TestShortenDigest:
    @pytest.mark.parametrize(
        "digest, expected",
        [
            ("sha256:abcdef1234567890abcdef1234567890", "sha256:abcdef1234"),
            ("sha256:short", "sha256:short"),
            ("not-a-digest", "not-a-digest"),
            ("sha256:abcdef1234", "sha256:abcdef1234"),
        ],
        ids=["long", "short", "no_prefix", "exact_length"],
    )
    def test_shorten(self, digest, expected):
        assert _shorten_digest(digest) == expected


def _mock_run(stdout="", returncode=0):
    """Create a mock subprocess.run result."""
    result = MagicMock()
    result.stdout = stdout
    result.returncode = returncode
    return result


class TestGetImageDigest:
    def test_returns_shortened_repo_digest(self):
        repo_digest = "ghcr.io/org/image@sha256:abcdef1234567890full"
        with patch(
            "validator.version_collector.subprocess.run",
            return_value=_mock_run(repo_digest),
        ):
            result = _get_image_digest("some-image:latest")
        assert result == "sha256:abcdef1234"

    def test_falls_back_to_image_id(self):
        with patch(
            "validator.version_collector.subprocess.run",
            side_effect=[
                _mock_run("", returncode=1),  # RepoDigests fails
                _mock_run("sha256:abcdef1234567890abcdef"),  # .Id succeeds
            ],
        ):
            result = _get_image_digest("some-image:latest")
        assert result == "sha256:abcdef1234"

    def test_returns_none_on_all_failures(self):
        with patch(
            "validator.version_collector.subprocess.run",
            return_value=_mock_run("", returncode=1),
        ):
            result = _get_image_digest("nonexistent:latest")
        assert result is None

    def test_returns_none_on_timeout(self):
        with patch(
            "validator.version_collector.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=10),
        ):
            result = _get_image_digest("some-image:latest")
        assert result is None


class TestGetContainerImageDigest:
    def test_resolves_via_config_image(self):
        with patch(
            "validator.version_collector.subprocess.run",
            side_effect=[
                _mock_run("ghcr.io/org/image:latest"),  # .Config.Image
                _mock_run(
                    "ghcr.io/org/image@sha256:abcdef1234567890abcdef"
                ),  # RepoDigests
            ],
        ):
            result = _get_container_image_digest("my-container")
        assert result == "sha256:abcdef1234"

    def test_falls_back_to_container_image_id(self):
        with patch(
            "validator.version_collector.subprocess.run",
            side_effect=[
                _mock_run("", returncode=1),  # .Config.Image fails
                _mock_run("sha256:abcdef1234567890abcdef"),  # .Image fallback
            ],
        ):
            result = _get_container_image_digest("my-container")
        assert result == "sha256:abcdef1234"


class TestCollectServiceVersions:
    def test_collects_all_services(self):
        digests = {
            "shoppingbench-search-server": "sha256:search1234567890",
            "shoppingbench-proxy": "sha256:proxy12345678901",
        }

        def mock_run(cmd, **kwargs):
            target = cmd[4]  # docker inspect --format <fmt> <target>
            fmt = cmd[3]
            if target in digests and "Config.Image" in fmt:
                return _mock_run(f"image-for-{target}")
            if target.startswith("image-for-") and "RepoDigests" in fmt:
                container = target.replace("image-for-", "")
                return _mock_run(f"registry/{container}@{digests[container]}")
            return _mock_run("", returncode=1)

        with patch("validator.version_collector.subprocess.run", side_effect=mock_run):
            with patch(
                "validator.version_collector.socket.gethostname", return_value=""
            ):
                versions = collect_service_versions()

        assert "search-server" in versions
        assert "proxy" in versions
        assert all(v.startswith("sha256:") for v in versions.values())

    def test_returns_empty_dict_when_docker_unavailable(self):
        with patch(
            "validator.version_collector.subprocess.run",
            side_effect=FileNotFoundError("docker not found"),
        ):
            with patch(
                "validator.version_collector.socket.gethostname", return_value=""
            ):
                versions = collect_service_versions()

        assert versions == {}

    def test_partial_failures_still_return_available(self):
        call_count = 0

        def mock_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            target = cmd[4]
            fmt = cmd[3]
            # Only proxy succeeds
            if target == "shoppingbench-proxy" and "Config.Image" in fmt:
                return _mock_run("proxy-image")
            if target == "proxy-image" and "RepoDigests" in fmt:
                return _mock_run("registry/proxy@sha256:proxydigest1234")
            return _mock_run("", returncode=1)

        with patch("validator.version_collector.subprocess.run", side_effect=mock_run):
            with patch(
                "validator.version_collector.socket.gethostname", return_value=""
            ):
                versions = collect_service_versions()

        assert "proxy" in versions
        assert "search-server" not in versions
