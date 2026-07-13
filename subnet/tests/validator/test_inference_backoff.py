"""Tests for the inference-provider 401 backoff (ORO-1597)."""

import logging

import pytest

from validator.inference_backoff import Inference401Backoff


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def backoff(clock):
    return Inference401Backoff(
        base_backoff=30.0,
        max_backoff=300.0,
        jitter=0.0,  # deterministic
        persistent_threshold_seconds=600.0,
        clock=clock,
    )


def test_inactive_returns_zero(backoff):
    assert backoff.should_delay_claim() == 0.0


def test_single_401_activates_at_base_tier(backoff):
    backoff.record_401()
    assert backoff.should_delay_claim() == pytest.approx(30.0, abs=0.01)


def test_second_401_within_active_window_doubles_tier(backoff, clock):
    backoff.record_401()
    clock.advance(5.0)  # still active (base=30s)
    backoff.record_401()
    # tier doubled from 30 -> 60; extends _active_until to now+60
    assert backoff.should_delay_claim() == pytest.approx(60.0, abs=0.01)


def test_success_resets_to_base(backoff, clock):
    backoff.record_401()
    backoff.record_401()  # tier now 120 for next activation
    backoff.record_success()
    assert backoff.should_delay_claim() == 0.0
    backoff.record_401()
    # After reset, tier is back at base
    assert backoff.should_delay_claim() == pytest.approx(30.0, abs=0.01)


def test_backoff_clears_after_expiry(backoff, clock):
    backoff.record_401()
    assert backoff.should_delay_claim() > 0
    clock.advance(31.0)
    assert backoff.should_delay_claim() == 0.0


def test_tier_capped_at_max(clock):
    b = Inference401Backoff(
        base_backoff=30.0, max_backoff=50.0, jitter=0.0, clock=clock
    )
    for _ in range(6):
        b.record_401()
    # tier progression: 30 -> 60 -> 120 -> ... all clamped to 50
    assert b._tier == 50.0


def test_persistent_401_logs_once_per_episode(backoff, clock, caplog):
    with caplog.at_level(logging.ERROR):
        backoff.record_401()
        # Keep re-triggering to sustain the active window past threshold.
        for _ in range(40):
            clock.advance(20.0)
            backoff.record_401()
            backoff.should_delay_claim()
    persistent = [
        r for r in caplog.records if "persistent auth misconfiguration" in r.message
    ]
    assert len(persistent) == 1


def test_transient_burst_does_not_log_persistent(backoff, clock, caplog):
    with caplog.at_level(logging.ERROR):
        backoff.record_401()
        clock.advance(30.0)
        backoff.record_success()
        backoff.should_delay_claim()
    assert not any(
        "persistent auth misconfiguration" in r.message for r in caplog.records
    )


def test_persistent_flag_resets_between_episodes(backoff, clock, caplog):
    """Regression: persistent-401 ERROR must re-arm after an episode
    expires without a record_success (e.g. failures are 402/inconclusive)."""

    def cross_threshold():
        backoff.record_401()
        for _ in range(35):
            clock.advance(20.0)
            backoff.record_401()
            backoff.should_delay_claim()

    with caplog.at_level(logging.ERROR):
        cross_threshold()
        clock.advance(1000.0)
        assert backoff.should_delay_claim() == 0.0  # expire episode 1
        cross_threshold()  # episode 2 — no record_success in between
    persistent = [
        r for r in caplog.records if "persistent auth misconfiguration" in r.message
    ]
    assert len(persistent) == 2


def test_in_flight_work_never_touched(backoff):
    """Shape check: no cancellation surface. A future refactor adding
    one must consciously break this test."""
    forbidden = {"cancel", "kill", "abort", "interrupt_run", "stop_run"}
    assert not (set(dir(backoff)) & forbidden)


def test_rejects_inverted_backoff_range(clock):
    with pytest.raises(ValueError):
        Inference401Backoff(base_backoff=100, max_backoff=50, clock=clock)
