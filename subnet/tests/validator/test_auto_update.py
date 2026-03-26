"""Tests for _check_for_updates auto-update logic."""

from unittest.mock import MagicMock, patch

import pytest
import requests

import validator.main as main_module
from validator.main import Validator


@pytest.fixture
def validator_instance():
    with patch.object(Validator, "__init__", lambda self: None):
        yield Validator()


def _mock_get(watchtower_resp=MagicMock(ok=True, status_code=200), health_ok=True):
    """Return a side_effect function that handles both Watchtower and health URLs."""

    def side_effect(url, **kwargs):
        if "watchtower" in url or "/v1/update" in url:
            if isinstance(watchtower_resp, Exception):
                raise watchtower_resp
            return watchtower_resp
        if "/health" in url:
            resp = MagicMock(ok=health_ok)
            if not health_ok:
                raise ConnectionError("not ready")
            return resp
        return MagicMock(ok=True)

    return side_effect


class TestCheckForUpdates:
    def test_skipped_when_disabled(self, validator_instance):
        with (
            patch.object(main_module, "AUTO_UPDATE_ENABLED", False),
            patch("requests.get") as mock_get,
            patch("subprocess.run") as mock_run,
        ):
            validator_instance._check_for_updates()
            mock_get.assert_not_called()
            mock_run.assert_not_called()

    def test_triggers_watchtower_waits_for_health_and_pulls(self, validator_instance):
        with (
            patch.object(main_module, "AUTO_UPDATE_ENABLED", True),
            patch("requests.get", side_effect=_mock_get()) as mock_get,
            patch("subprocess.run", return_value=MagicMock(returncode=0)) as mock_run,
        ):
            validator_instance._check_for_updates()

            urls = [c[0][0] for c in mock_get.call_args_list]
            assert any("/v1/update" in u for u in urls)
            assert any("/health" in u for u in urls)
            pull_calls = [
                c for c in mock_run.call_args_list if c[0][0][:2] == ["docker", "pull"]
            ]
            assert len(pull_calls) > 0

    @pytest.mark.parametrize(
        "watchtower_error",
        [
            requests.exceptions.ConnectionError("refused"),
            RuntimeError("unexpected"),
        ],
    )
    def test_watchtower_errors_do_not_crash(self, validator_instance, watchtower_error):
        with (
            patch.object(main_module, "AUTO_UPDATE_ENABLED", True),
            patch(
                "requests.get", side_effect=_mock_get(watchtower_resp=watchtower_error)
            ),
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
        ):
            validator_instance._check_for_updates()

    def test_sandbox_pull_failure_does_not_crash(self, validator_instance):
        with (
            patch.object(main_module, "AUTO_UPDATE_ENABLED", True),
            patch("requests.get", side_effect=_mock_get()),
            patch(
                "subprocess.run", return_value=MagicMock(returncode=1, stderr="denied")
            ),
        ):
            validator_instance._check_for_updates()

    def test_waits_for_health_after_watchtower(self, validator_instance):
        """Health check retries until proxy is ready."""
        health_calls = 0

        def side_effect(url, **kwargs):
            nonlocal health_calls
            if "/v1/update" in url:
                return MagicMock(ok=True, status_code=200)
            if "/health" in url:
                health_calls += 1
                if health_calls < 3:
                    raise ConnectionError("not ready")
                return MagicMock(ok=True)
            return MagicMock(ok=True)

        with (
            patch.object(main_module, "AUTO_UPDATE_ENABLED", True),
            patch("requests.get", side_effect=side_effect),
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
            patch("time.sleep"),
        ):
            validator_instance._check_for_updates()

        assert health_calls == 3
