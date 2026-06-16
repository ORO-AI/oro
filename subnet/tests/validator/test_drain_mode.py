"""Tests for the validator drain-mode hook (ORO-1150)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from subnet.validator.drain import (
    DRAIN_CACHE_TTL_SECONDS,
    drain_mode_active,
    handle_drain_tick,
)


@pytest.fixture
def drain_file(tmp_path: Path) -> str:
    return str(tmp_path / "drain")


@pytest.mark.parametrize("present,expected", [(False, False), (True, True)])
def test_drain_mode_reads_sentinel(drain_file, present, expected):
    if present:
        Path(drain_file).touch()
    assert drain_mode_active({}, drain_file=drain_file) is expected


@pytest.mark.parametrize(
    "starts,flips_to,wait_factor,expected",
    [
        (False, True, 0.5, False),  # flip-on within TTL: stale False
        (False, True, 1.5, True),   # flip-on past TTL: re-stat picks up True
        (True, False, 0.5, True),   # flip-off within TTL: stale True
        (True, False, 1.5, False),  # flip-off past TTL: re-stat picks up False
    ],
)
def test_drain_cache_ttl(drain_file, starts, flips_to, wait_factor, expected):
    if starts:
        Path(drain_file).touch()
    cache: dict = {}
    drain_mode_active(cache, now=1000.0, drain_file=drain_file)
    if flips_to and not starts:
        Path(drain_file).touch()
    elif starts and not flips_to:
        Path(drain_file).unlink()
    now = 1000.0 + DRAIN_CACHE_TTL_SECONDS * wait_factor
    assert drain_mode_active(cache, now=now, drain_file=drain_file) is expected


def test_drain_mode_fails_closed_on_oserror(tmp_path):
    """Mount-misconfig → fail-CLOSED + WARNING (prevents silent-claim)."""
    parent = tmp_path / "nope"
    parent.write_text("not a directory")  # NotADirectoryError on stat()
    with patch("subnet.validator.drain.logging") as log:
        assert drain_mode_active({}, drain_file=str(parent / "drain")) is True
        assert any("fail-CLOSED" in str(c) for c in log.warning.call_args_list)


def test_handle_drain_tick_draining_flushes_no_burn(drain_file):
    """Sentinel present → True, no-burn flush, metric tick, log-once."""
    Path(drain_file).touch()
    rq = MagicMock()
    rq.get_pending_count.return_value = 3
    state: dict = {}
    with patch("subnet.validator.drain.DRAIN_TICKS_TOTAL") as metric, patch(
        "subnet.validator.drain.logging"
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
    with patch("subnet.validator.drain.DRAIN_TICKS_TOTAL") as metric:
        assert handle_drain_tick({}, rq, 0.0, drain_file=drain_file) is False
        metric.inc.assert_not_called()
        rq.process_pending.assert_not_called()


def test_handle_drain_tick_logs_resume_on_clear(drain_file):
    state = {"logged": True}
    with patch("subnet.validator.drain.logging") as log:
        assert handle_drain_tick(state, MagicMock(), 0.0, drain_file=drain_file) is False
        assert state["logged"] is False
        assert any("cleared" in str(c) for c in log.info.call_args_list)
