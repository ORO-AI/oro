"""Exponential backoff utility for retry logic."""

import random


class ExponentialBackoff:
    """Exponential backoff with optional jitter.

    Progression: base -> base*2 -> base*4 -> ... -> max
    Jitter adds +/-20% randomness to each value.
    """

    def __init__(
        self,
        base_seconds: float = 5.0,
        max_seconds: float = 300.0,
        jitter: bool = True,
    ):
        self.base_seconds = base_seconds
        self.max_seconds = max_seconds
        self.jitter = jitter
        self._current = base_seconds

    def next(self) -> float:
        """Get next backoff duration and increase for next call."""
        value = self._current

        # Apply jitter (+/-20%)
        if self.jitter:
            jitter_factor = 1.0 + random.uniform(-0.2, 0.2)
            value = value * jitter_factor

        # Increase for next call (before cap)
        self._current = min(self._current * 2, self.max_seconds)

        return value

    def reset(self) -> None:
        """Reset backoff to initial value."""
        self._current = self.base_seconds
