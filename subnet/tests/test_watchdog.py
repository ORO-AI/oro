"""Tests for the validator progress watchdog (ORO-1414, Follow-up 2).

The main loop stamps a liveness beat each iteration; if it stops progressing
(e.g. the host disk fills and the loop wedges) the watchdog aborts the process
so docker's ``restart: unless-stopped`` brings a fresh validator back up.

Pure unit (injectable clock) so it lives in subnet/tests/, not the heavy
validator conftest dir.
"""

from __future__ import annotations

import threading

from validator.watchdog import ProgressWatchdog


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_not_stalled_before_timeout():
    clk = _FakeClock()
    wd = ProgressWatchdog(timeout_seconds=100, now=clk)  # last beat at t=0
    clk.t = 99
    assert wd.stalled() is False


def test_stalled_after_timeout():
    clk = _FakeClock()
    wd = ProgressWatchdog(timeout_seconds=100, now=clk)
    clk.t = 101
    assert wd.stalled() is True


def test_beat_resets_the_timer():
    clk = _FakeClock()
    wd = ProgressWatchdog(timeout_seconds=100, now=clk)
    clk.t = 150
    assert wd.stalled() is True

    wd.beat()  # last beat now at t=150
    clk.t = 200
    assert wd.stalled() is False  # 200 - 150 = 50 < 100


def test_monitor_thread_aborts_when_stalled():
    clk = _FakeClock()
    fired = threading.Event()
    wd = ProgressWatchdog(
        timeout_seconds=100, now=clk, on_stall=fired.set, check_interval=0.01
    )
    clk.t = 1000  # already far past the timeout
    wd.start()
    try:
        assert fired.wait(2.0) is True
    finally:
        wd.stop()


def test_monitor_thread_quiet_while_healthy():
    clk = _FakeClock()
    fired = threading.Event()
    wd = ProgressWatchdog(
        timeout_seconds=100, now=clk, on_stall=fired.set, check_interval=0.01
    )
    wd.start()
    try:
        # clock never advances past the timeout, so on_stall must not fire
        assert fired.wait(0.2) is False
    finally:
        wd.stop()
