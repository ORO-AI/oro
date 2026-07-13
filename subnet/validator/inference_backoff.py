"""Claim-level backoff on inference-provider 401 events (ORO-1597).

OpenRouter's per-run scoped keys 401 for a few seconds right after
creation — a propagation race. A race mints keys for the whole
validator fleet in the same window, so those 401s arrive as a
synchronized burst. The smoke-test in
``main.py::_validate_inference_token`` already retries individual 401s
with exponential backoff + jitter, but if the propagation lag exceeds
the per-run retry budget the run is failed and the validator
immediately claims another one — walking straight into the same storm.

A 401 that reaches this module has already survived the smoke-test's
retry budget, so it is the filtered signal (not a transient blip).
Every one earns a backoff on the *next claim*; subsequent 401s within
the active window double the tier. A single healthy first-try
smoke-test clears the state.

Not thread-safe: the validator main loop is single-threaded, and all
record/query calls originate there. If the topology ever grows workers
that share this state, wrap in a Lock.

Only the next claim is gated. In-flight work is never touched — this
module exposes no cancellation surface (guarded by a shape test).
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
            base_backoff: First-tier sleep. Sized above the observed
                propagation tail so one tier usually suffices.
            max_backoff: Cap on any single tier — never block a claim
                longer than this so a wedged provider doesn't silence
                the validator indefinitely.
            jitter: Fractional +/- randomness applied per sleep,
                preventing fleet-wide lockstep retry.
            persistent_threshold_seconds: Continuously-active duration
                past which we log ERROR + tick a counter, exactly once
                per episode. Signals a real auth misconfig without
                paging.
            clock: Time source override for tests.
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
            self._active_since = None
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
