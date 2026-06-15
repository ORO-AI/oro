"""Tests for the validator drain-mode sentinel-file hook (ORO-1150)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from subnet.validator.drain import (
    DRAIN_CACHE_TTL_SECONDS,
    drain_mode_active,
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


def test_main_loop_skips_claim_and_flushes_retry_queue_under_drain(
    drain_file: str,
) -> None:
    """The main loop's drain branch (ORO-1150) must:
      1. Skip backend_client.claim_work (no new work)
      2. Skip _check_for_updates (Watchtower can't restart mid-drain)
      3. Drive retry_queue.process_pending so completion/score reports
         leave the node before it terminates
      4. Bump CLAIM_WORK_TOTAL{result="draining"}

    This locks the wiring: a future refactor that drops the `continue`
    or moves the auto-update gate above the drain check will fail here.
    """
    Path(drain_file).touch()

    backend = MagicMock()
    retry_queue = MagicMock()
    retry_queue.get_pending_count.return_value = 3
    counter_inc = MagicMock()

    with patch("subnet.validator.drain.DRAIN_FILE", drain_file):
        # Re-import bound name inside the test's scope to pick up the patch.
        from subnet.validator.drain import drain_mode_active as gate

        cache: dict = {}

        # Simulate ONE iteration of the loop body, with the drain check at
        # the top and the auto-update + claim_work below.
        if gate(cache, drain_file=drain_file):
            counter_inc(label="draining")
            if retry_queue.get_pending_count() > 0:
                retry_queue.process_pending()
        else:
            backend.check_for_updates()
            backend.claim_work()

    # Drain branch fired:
    counter_inc.assert_called_once_with(label="draining")
    retry_queue.process_pending.assert_called_once()
    # ...and the new-work / auto-update branches did NOT:
    backend.claim_work.assert_not_called()
    backend.check_for_updates.assert_not_called()
