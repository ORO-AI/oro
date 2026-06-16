"""Drain-mode sentinel-file hook (ORO-1150).

Touch DRAIN_FILE to stop new claim_work while in-flight evals finish.
Remove to resume. Path override: ORO_DRAIN_FILE.
"""

from __future__ import annotations

import logging
import os
import time

DRAIN_FILE = os.environ.get("ORO_DRAIN_FILE", "/var/run/oro-validator/drain")

# bittensor's btlogging configures a stdlib StreamHandler on the "bittensor"
# logger; the root logger has none. Use the named logger so drain messages
# actually reach docker logs (rest of the validator does the same).
_log = logging.getLogger("bittensor")


def drain_mode_active(*, drain_file: str = DRAIN_FILE) -> bool:
    """True iff drain_file exists. Fail-CLOSED on unreadable path
    (EACCES / ENOTDIR / missing mount) — silently claiming work while
    the orchestrator thinks we're draining is the worse failure mode.
    """
    try:
        os.stat(drain_file)
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        _log.warning(
            f"Drain sentinel path unreadable ({type(e).__name__}: {e}) — "
            "fail-CLOSED. Check the host bind mount."
        )
        return True


def handle_drain_tick(
    state: dict, retry_queue, poll_interval: float, *, drain_file: str = DRAIN_FILE
) -> bool:
    """Return True iff caller should ``continue`` the main-loop tick.

    Drain flush uses count_attempts=False so a multi-minute drain plus a
    coincident backend transient can't burn retry budget on reports we
    haven't actually given up on.
    """
    if drain_mode_active(drain_file=drain_file):
        if not state.get("logged"):
            _log.info("Drain sentinel present — pausing claim_work")
            state["logged"] = True
        # Lazy import: module-level import triggers dual-registration when
        # test_auto_update imports via `validator.metrics` while this file
        # imports via `subnet.validator.metrics`.
        from .metrics import DRAIN_TICKS_TOTAL

        DRAIN_TICKS_TOTAL.inc()
        if retry_queue.get_pending_count() > 0:
            retry_queue.process_pending(count_attempts=False)
        time.sleep(poll_interval)
        return True
    if state.get("logged"):
        _log.info("Drain sentinel cleared — resuming claim_work")
        state["logged"] = False
    return False
