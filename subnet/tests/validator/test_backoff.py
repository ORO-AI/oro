import sys
from pathlib import Path

# Add test-subnet to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validator.backoff import ExponentialBackoff


class TestExponentialBackoff:
    def test_initial_value(self):
        backoff = ExponentialBackoff(base_seconds=5, max_seconds=300)
        # First call returns base value
        value = backoff.next()
        assert 4 <= value <= 6  # Base with jitter

    def test_exponential_increase(self):
        backoff = ExponentialBackoff(base_seconds=5, max_seconds=300, jitter=False)
        assert backoff.next() == 5
        assert backoff.next() == 10
        assert backoff.next() == 20
        assert backoff.next() == 40

    def test_max_cap(self):
        backoff = ExponentialBackoff(base_seconds=5, max_seconds=20, jitter=False)
        backoff.next()  # 5
        backoff.next()  # 10
        backoff.next()  # 20
        assert backoff.next() == 20  # Capped at max

    def test_reset(self):
        backoff = ExponentialBackoff(base_seconds=5, max_seconds=300, jitter=False)
        backoff.next()  # 5
        backoff.next()  # 10
        backoff.reset()
        assert backoff.next() == 5  # Back to base

    def test_jitter_adds_randomness(self):
        backoff = ExponentialBackoff(base_seconds=10, max_seconds=300, jitter=True)
        values = [backoff.next() for _ in range(10)]
        backoff.reset()
        values2 = [backoff.next() for _ in range(10)]
        # With jitter, sequences should differ (very high probability)
        assert values != values2
