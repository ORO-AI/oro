"""Tests for the validator drain-mode sentinel-file hook (ORO-1150)."""

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
def test_drain_mode_reads_sentinel(drain_file: str, present: bool, expected: bool) -> None:
    if present:
        Path(drain_file).touch()
    assert drain_mode_active({}, drain_file=drain_file) is expected


@pytest.mark.parametrize(
    "starts_present,flips_to_present,wait_factor,expected",
    [
        # File flips on AFTER cache populated False; within TTL we still see False.
        (False, True, 0.5, False),
        # Same flip-on, but past TTL: re-stat picks up True.
        (False, True, 1.5, True),
        # File flips off AFTER cache populated True; within TTL still True.
        (True, False, 0.5, True),
        # Same flip-off, past TTL: re-stat picks up False.
        (True, False, 1.5, False),
    ],
)
def test_drain_cache_ttl(
    drain_file: str,
    starts_present: bool,
    flips_to_present: bool,
    wait_factor: float,
    expected: bool,
) -> None:
    """Cache must keep stat() off the hot path within the TTL and pick up
    state changes past it. Covers both flip-on (orchestrator starts drain)
    and flip-off (orchestrator clears drain) directions."""
    if starts_present:
        Path(drain_file).touch()
    cache: dict = {}
    drain_mode_active(cache, now=1000.0, drain_file=drain_file)  # populate
    if flips_to_present and not starts_present:
        Path(drain_file).touch()
    elif starts_present and not flips_to_present:
        Path(drain_file).unlink()
    now = 1000.0 + DRAIN_CACHE_TTL_SECONDS * wait_factor
    assert drain_mode_active(cache, now=now, drain_file=drain_file) is expected


@pytest.fixture
def fake_retry_queue():
    rq = MagicMock()
    rq.get_pending_count.return_value = 3
    return rq


def test_handle_drain_tick_returns_true_and_flushes_no_burn(
    drain_file: str, fake_retry_queue
) -> None:
    """Sentinel present → return True (caller short-circuits), flush
    retry_queue with count_attempts=False so a multi-minute drain +
    transient outage doesn't drop reports, increment metric, log-once
    on entry. Second tick under same drain must NOT log again."""
    Path(drain_file).touch()
    metric = MagicMock()
    sleeps: list[float] = []
    state: dict = {}

    with patch("subnet.validator.drain.logging") as log:
        assert (
            handle_drain_tick(
                state,
                fake_retry_queue,
                poll_interval=0.0,
                metric_counter=metric,
                sleep_fn=sleeps.append,
                drain_file=drain_file,
            )
            is True
        )
        assert state["logged"] is True
        metric.inc.assert_called_once()
        fake_retry_queue.process_pending.assert_called_once_with(count_attempts=False)
        assert sleeps == [0.0]
        entry_logs = [c for c in log.info.call_args_list if "present" in str(c)]
        assert len(entry_logs) == 1

        # Second tick, same drain → still True, no second entry log.
        assert (
            handle_drain_tick(
                state,
                fake_retry_queue,
                poll_interval=0.0,
                metric_counter=metric,
                sleep_fn=sleeps.append,
                # Bypass TTL by advancing now past cache window.
                now=10_000.0,
                drain_file=drain_file,
            )
            is True
        )
        entry_logs = [c for c in log.info.call_args_list if "present" in str(c)]
        assert len(entry_logs) == 1


def test_handle_drain_tick_returns_false_when_sentinel_absent(
    drain_file: str, fake_retry_queue
) -> None:
    """No sentinel → return False so the caller proceeds to claim_work.
    No retry flush, no metric tick, no sleep."""
    metric = MagicMock()
    sleeps: list[float] = []
    assert (
        handle_drain_tick(
            {},
            fake_retry_queue,
            poll_interval=0.0,
            metric_counter=metric,
            sleep_fn=sleeps.append,
            drain_file=drain_file,
        )
        is False
    )
    metric.inc.assert_not_called()
    fake_retry_queue.process_pending.assert_not_called()
    assert sleeps == []


def test_handle_drain_tick_logs_resume_on_clear(drain_file: str, fake_retry_queue) -> None:
    """Sentinel was present (logged=True) → cleared between ticks →
    resume-log fires once and logged flips back to False."""
    state: dict = {"logged": True}
    with patch("subnet.validator.drain.logging") as log:
        assert (
            handle_drain_tick(
                state,
                fake_retry_queue,
                poll_interval=0.0,
                metric_counter=MagicMock(),
                sleep_fn=lambda _: None,
                drain_file=drain_file,
            )
            is False
        )
        assert state["logged"] is False
        cleared = [c for c in log.info.call_args_list if "cleared" in str(c)]
        assert len(cleared) == 1


def test_drain_mode_active_fails_closed_on_eacces(tmp_path: Path) -> None:
    """Mount-misconfig (OSError on stat) → fail-CLOSED (return True) +
    surface a WARNING. Prevents the silent-keep-claiming failure mode."""
    parent = tmp_path / "nope"
    parent.write_text("not a directory")  # NotADirectoryError on stat()
    bad_path = str(parent / "drain")
    with patch("subnet.validator.drain.logging") as log:
        assert drain_mode_active({}, drain_file=bad_path) is True
        assert any("fail-CLOSED" in str(c) for c in log.warning.call_args_list)
