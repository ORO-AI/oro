"""Claim-level backoff on inference-provider 401 events (ORO-1597).

OpenRouter's per-run scoped keys 401 briefly after creation — a
propagation race. Races mint keys fleet-wide in the same window, so
the 401s arrive synchronized. The smoke-test in
``main.py::_validate_inference_token`` already retries individual 401s
with backoff + jitter, but if propagation exceeds that budget the run
is failed and the validator immediately claims another — walking into
the same storm.

A 401 that reaches this module has already survived the per-run retry
budget, so it is the filtered signal. Every one earns a backoff on
the NEXT claim; subsequent 401s in the active window double the tier;
a healthy first-try smoke-test clears the state.

Only the next claim is gated. In-flight work is never touched — this
module exposes no cancellation surface (guarded by a shape test).
Not thread-safe: the validator main loop is single-threaded.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, Optional

from prometheus_client import Counter

_401_EVENTS = Counter(
    "validator_inference_401_events_total",
    "Inference-provider 401 events observed after retry exhaustion",
)
_BACKOFF_ACTIVATIONS = Counter(
    "validator_inference_401_backoff_activations_total",
    "Transitions from inactive to active inference-401 claim-backoff",
)


class Inference401Backoff:
    def __init__(
        self,
        base_backoff: float = 30.0,
        max_backoff: float = 300.0,
        jitter: float = 0.3,
        persistent_threshold_seconds: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Args:
            base_backoff: First-tier sleep, sized above observed
                propagation tail.
            max_backoff: Per-tier cap — bounds silence during a wedged
                provider.
            jitter: +/- fractional randomness per sleep — prevents
                fleet-wide lockstep retry.
            persistent_threshold_seconds: Continuously-active duration
                past which we log ERROR once per episode (real
                misconfig signal, does not page).
            clock: Time source, overridable for tests.
        """
        if base_backoff <= 0 or max_backoff < base_backoff:
            raise ValueError("base_backoff and max_backoff must be > 0 and consistent")
        self._base = base_backoff
        self._max = max_backoff
        self._jitter = jitter
        self._persistent_after = persistent_threshold_seconds
        self._clock = clock

        self._tier = base_backoff
        self._active_until: float = 0.0
        self._active_since: Optional[float] = None
        self._persistent_logged = False

    def record_401(self) -> None:
        """Record a 401 event whose retries the smoke-test has exhausted.

        Run-level 401s that were retried away must NOT be recorded
        here — the smoke-test's per-run retry is a different signal.
        """
        _401_EVENTS.inc()
        now = self._clock()
        sleep_for = min(
            self._tier * (1 + random.uniform(-self._jitter, self._jitter)),
            self._max,
        )
        was_active = self._active_since is not None and now < self._active_until
        if not was_active:
            self._active_since = now
            self._persistent_logged = False
            _BACKOFF_ACTIVATIONS.inc()
            logging.warning(
                "Inference-401 backoff activated: sleeping %.1fs before next claim_work",
                sleep_for,
            )
        else:
            logging.info(
                "Inference-401 backoff extended: next tier sleep %.1fs",
                sleep_for,
            )
        self._active_until = max(self._active_until, now + sleep_for)
        self._tier = min(self._tier * 2, self._max)

    def record_success(self) -> None:
        """Reset the state machine on a healthy first-try smoke-test."""
        self._tier = self._base
        self._active_until = 0.0
        self._active_since = None
        self._persistent_logged = False

    def should_delay_claim(self) -> float:
        """Seconds the main loop should sleep before the next claim, 0 if none."""
        now = self._clock()
        if now >= self._active_until:
            # Clear both per-episode flags so the NEXT episode re-arms
            # persistent-401 logging even without an intervening
            # record_success (all failures could be 402/inconclusive).
            self._active_since = None
            self._persistent_logged = False
            return 0.0
        if (
            self._active_since is not None
            and not self._persistent_logged
            and (now - self._active_since) >= self._persistent_after
        ):
            self._persistent_logged = True
            logging.error(
                "Inference-401 backoff active continuously for %.0fs (>= %.0fs) — "
                "likely persistent auth misconfiguration; check provider credentials.",
                now - self._active_since,
                self._persistent_after,
            )
        return self._active_until - now
