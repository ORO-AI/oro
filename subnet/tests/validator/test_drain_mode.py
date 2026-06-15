"""Tests for the validator drain-mode sentinel-file hook (ORO-1150)."""

from __future__ import annotations

from pathlib import Path

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
