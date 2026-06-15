"""Tests for the validator drain-mode sentinel-file hook (ORO-1150).

The hook lets any orchestrator (AWS ASG drain script, k8s preStop hook,
systemd timer, a manual `touch`) tell the validator main loop to stop
claiming new work while letting in-flight evals finish cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subnet.validator.drain import (
    DRAIN_CACHE_TTL_SECONDS,
    DRAIN_FILE,
    drain_mode_active,
)


@pytest.fixture
def drain_file(tmp_path: Path) -> str:
    return str(tmp_path / "drain")


def test_default_drain_file_constant_is_namespaced() -> None:
    """Default sentinel path lives under a validator-owned directory so
    operators don't accidentally collide with another process's flag."""
    assert DRAIN_FILE.startswith("/var/run/oro-validator/")


def test_no_drain_file_returns_false(drain_file: str) -> None:
    cache: dict[str, float] = {}
    assert drain_mode_active(cache, drain_file=drain_file) is False


def test_present_drain_file_returns_true(drain_file: str) -> None:
    Path(drain_file).touch()
    cache: dict[str, float] = {}
    assert drain_mode_active(cache, drain_file=drain_file) is True


def test_cache_skips_filesystem_check_within_ttl(drain_file: str) -> None:
    """Within the TTL the helper must return the cached value even if the
    file state changes — that's the entire point of the cache (keep the
    main-loop hot path from stat()ing on every poll)."""
    cache: dict[str, float] = {}
    # First call: file absent, populates cache as False.
    assert drain_mode_active(cache, now=1000.0, drain_file=drain_file) is False
    # File appears AFTER the cache was populated.
    Path(drain_file).touch()
    # Within TTL: still returns cached False.
    assert (
        drain_mode_active(
            cache, now=1000.0 + DRAIN_CACHE_TTL_SECONDS - 1, drain_file=drain_file
        )
        is False
    )
    # Past TTL: re-stats and picks up the new state.
    assert (
        drain_mode_active(
            cache, now=1000.0 + DRAIN_CACHE_TTL_SECONDS + 1, drain_file=drain_file
        )
        is True
    )


def test_cache_picks_up_drain_clear_after_ttl(drain_file: str) -> None:
    """The reverse — once the orchestrator removes the file, the validator
    must come back out of drain mode on the next post-TTL check."""
    Path(drain_file).touch()
    cache: dict[str, float] = {}
    assert drain_mode_active(cache, now=2000.0, drain_file=drain_file) is True
    Path(drain_file).unlink()
    # Still cached as True within TTL.
    assert (
        drain_mode_active(
            cache, now=2000.0 + DRAIN_CACHE_TTL_SECONDS - 1, drain_file=drain_file
        )
        is True
    )
    # After TTL: returns False.
    assert (
        drain_mode_active(
            cache, now=2000.0 + DRAIN_CACHE_TTL_SECONDS + 1, drain_file=drain_file
        )
        is False
    )
