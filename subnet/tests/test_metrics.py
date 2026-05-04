"""Tests for the validator's Prometheus metric definitions."""

from prometheus_client import REGISTRY

from validator.metrics import (
    ACTIVE_RUNS,
    CLAIM_WORK_SECONDS,
    CLAIM_WORK_TOTAL,
    HEARTBEAT_TOTAL,
    SANDBOX_ACTIVE,
    SANDBOX_DURATION_SECONDS,
)


def _names() -> set[str]:
    """All metric names currently registered with the default registry."""
    return {m.name for m in REGISTRY.collect()}


class TestMetricsRegistration:
    """Each metric is registered under its expected base name.

    `prometheus_client` strips the `_total` suffix from Counter names in the
    registry — it's only re-added on exposition.
    """

    def test_all_metrics_registered(self):
        names = _names()
        assert {
            "validator_active_runs",
            "validator_heartbeat",
            "validator_claim_work_seconds",
            "validator_claim_work",
            "validator_sandbox_active",
            "validator_sandbox_duration_seconds",
        }.issubset(names)


class TestMetricBehavior:
    """Sanity-check that incrementing / observing actually updates values."""

    def test_active_runs_inc_dec(self):
        before = ACTIVE_RUNS._value.get()
        ACTIVE_RUNS.inc()
        assert ACTIVE_RUNS._value.get() == before + 1
        ACTIVE_RUNS.dec()
        assert ACTIVE_RUNS._value.get() == before

    def test_heartbeat_labels(self):
        # Both label values are valid; either should be incrementable.
        HEARTBEAT_TOTAL.labels(result="success").inc()
        HEARTBEAT_TOTAL.labels(result="failure").inc()

    def test_claim_work_labels(self):
        CLAIM_WORK_TOTAL.labels(result="success").inc()
        CLAIM_WORK_TOTAL.labels(result="empty").inc()
        CLAIM_WORK_TOTAL.labels(result="error").inc()

    def test_sandbox_active_inc_dec(self):
        before = SANDBOX_ACTIVE._value.get()
        SANDBOX_ACTIVE.inc()
        assert SANDBOX_ACTIVE._value.get() == before + 1
        SANDBOX_ACTIVE.dec()
        assert SANDBOX_ACTIVE._value.get() == before

    def test_sandbox_duration_observe(self):
        # Histogram.observe doesn't return; just ensure no exception.
        SANDBOX_DURATION_SECONDS.observe(120.0)
        SANDBOX_DURATION_SECONDS.observe(300.0)

    def test_claim_work_seconds_time(self):
        # `with histogram.time():` should record one observation.
        with CLAIM_WORK_SECONDS.time():
            pass  # immediate — observation should be near-zero
