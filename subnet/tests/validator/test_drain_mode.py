"""Tests for the validator drain-mode hook (ORO-1150)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import validator.metrics  # noqa: F401 — needed so patch() can resolve the lazy import target

from validator.drain import drain_mode_active, handle_drain_tick


@pytest.fixture
def drain_file(tmp_path: Path) -> str:
    return str(tmp_path / "drain")


@pytest.mark.parametrize("present,expected", [(False, False), (True, True)])
def test_drain_mode_reads_sentinel(drain_file, present, expected):
    if present:
        Path(drain_file).touch()
    assert drain_mode_active(drain_file=drain_file) is expected


def test_drain_mode_fails_closed_on_oserror(tmp_path):
    """Mount-misconfig → fail-CLOSED + WARNING (prevents silent-claim)."""
    parent = tmp_path / "nope"
    parent.write_text("not a directory")  # NotADirectoryError on stat()
    with patch("validator.drain._log") as log:
        assert drain_mode_active(drain_file=str(parent / "drain")) is True
        assert any("fail-CLOSED" in str(c) for c in log.warning.call_args_list)


def test_handle_drain_tick_draining_flushes_no_burn(drain_file):
    """Sentinel present → True, no-burn flush, metric tick, log-once."""
    Path(drain_file).touch()
    rq = MagicMock()
    rq.get_pending_count.return_value = 3
    state: dict = {}
    with patch("validator.metrics.DRAIN_TICKS_TOTAL") as metric, patch(
        "validator.drain._log"
    ) as log:
        for _ in range(2):
            assert handle_drain_tick(state, rq, 0.0, drain_file=drain_file) is True
        assert metric.inc.call_count == 2
        assert rq.process_pending.call_count == 2
        rq.process_pending.assert_called_with(count_attempts=False)
        entry = [c for c in log.info.call_args_list if "present" in str(c)]
        assert len(entry) == 1


def test_handle_drain_tick_absent_passes_through(drain_file):
    rq = MagicMock()
    with patch("validator.metrics.DRAIN_TICKS_TOTAL") as metric:
        assert handle_drain_tick({}, rq, 0.0, drain_file=drain_file) is False
        metric.inc.assert_not_called()
        rq.process_pending.assert_not_called()


def test_handle_drain_tick_logs_resume_on_clear(drain_file):
    state = {"logged": True}
    with patch("validator.drain._log") as log:
        assert handle_drain_tick(state, MagicMock(), 0.0, drain_file=drain_file) is False
        assert state["logged"] is False
        assert any("cleared" in str(c) for c in log.info.call_args_list)
