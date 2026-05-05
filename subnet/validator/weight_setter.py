"""Background thread for periodic weight updates.

Builds a deterministic top-50% race-finisher weight vector via
`weight_distribution.build_metagraph_weight_vector` and submits it to the
chain. Determinism across validators is load-bearing for Yuma consensus on
subnet 15 (`kappa = 0.5`).

When no completed race is available (fresh subnet, race-system rollback,
backend outage) the tick is skipped — weight setting waits for the next
tick rather than submitting a stale or non-race vector.
"""

import threading
from typing import Optional

from bittensor.utils.btlogging import logging
from oro_sdk.types import UNSET

from .backend_client import BackendClient, BackendError
from .weight_distribution import (
    RankedFinisher,
    build_metagraph_weight_vector,
    compute_pinned_weights,
)


def _qualifiers_to_finishers(qualifiers) -> list[RankedFinisher]:
    """Reduce SDK `RaceQualifierPublic` records to ranked finishers.

    A "finisher" is a qualifier with a non-null `race_score` — i.e.
    the agent ran the race to completion and got scored. Qualifiers
    with `race_score=null` (DNF, eliminated mid-race, never executed)
    are intentionally dropped: they did not finish the race, so they
    are not protected from deregistration by this mechanism. A
    `race_score=0.0` finisher is still a finisher (they completed but
    scored zero) and competes for a top-half slot like everyone else;
    the linear taper naturally drops them out of the protected set
    when other finishers outscore them.

    Also drops entries without a `miner_hotkey` defensively so a
    partial-data Backend response can't poison the ranking.
    """
    finishers: list[RankedFinisher] = []
    for q in qualifiers:
        score = q.race_score
        if score is None or score is UNSET:
            continue
        hotkey = q.miner_hotkey
        if not hotkey or hotkey is UNSET:
            continue
        finishers.append(
            RankedFinisher(
                miner_hotkey=str(hotkey),
                agent_version_id=str(q.agent_version_id),
                race_score=float(score),
            )
        )
    return finishers


class WeightSetterThread:
    """Periodically computes the top-50% weight vector and submits it on-chain.

    Runs in a background thread, independent of the evaluation loop.
    """

    # How far back to scan `get_race_history` for the most recent
    # `RACE_COMPLETE`. The newest race may be `QUALIFYING_OPEN` or
    # `RACE_RUNNING` for ~24h of every cycle; we still want to protect
    # last race's finishers during that window. 5 covers typical race
    # cadence (one in-progress + a handful of completed) without paging.
    _RACE_HISTORY_SCAN_LIMIT = 5

    def __init__(
        self,
        backend_client: BackendClient,
        subtensor,
        metagraph,
        wallet,
        netuid: int,
        interval_seconds: int = 300,
        t_top: float = 0.25,
        t_burn: float = 0.75,
    ):
        self.backend_client = backend_client
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.wallet = wallet
        self.netuid = netuid
        self.interval_seconds = interval_seconds
        self.t_top = t_top
        self.t_burn = t_burn

        # Fail fast on misconfiguration — the validator process should not
        # start setting weights with invalid ratios. tail_sum=0 is the
        # smallest case (any t_top + t_burn = 1 will pass), validating the
        # ratio constraint without requiring a representative N.
        compute_pinned_weights(t_top, t_burn, tail_sum=0)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the weight setter background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the weight setter thread and wait for it to finish.

        Uses a generous timeout to account for:
        - HTTP request to Backend (up to 30s, race detail can be larger)
        - Blockchain transaction with wait_for_inclusion (up to 60s)
        - Buffer time (10s)
        """
        self._stop_event.set()
        if self._thread is not None:
            join_timeout = 100  # 30s HTTP + 60s blockchain + 10s buffer
            self._thread.join(timeout=join_timeout)
            if self._thread.is_alive():
                logging.warning(
                    f"Weight setter thread did not stop within {join_timeout}s"
                )

    def _fetch_race_finishers(self) -> Optional[list[RankedFinisher]]:
        """Return finishers from the most recent completed race, or None.

        Walks `get_race_history` newest-first and returns finishers from
        the first race in `RACE_COMPLETE` status. When the newest race is
        in progress (`QUALIFYING_OPEN`, `RACE_RUNNING`, etc.) we still
        want to protect last race's finishers between races — using only
        `limit=1` would skip the prior completed race and lose the
        protection for the entire current cycle.

        Returns None only when there is no completed race in the recent
        history (fresh subnet, recovery from race-system rollback). The
        caller skips weight submission for this tick rather than falling
        back to a stale vector.
        """
        history = self.backend_client.get_race_history(
            limit=self._RACE_HISTORY_SCAN_LIMIT
        )
        races = history.races if history.races is not UNSET else []
        for race in races:
            if str(race.status) == "RACE_COMPLETE":
                detail = self.backend_client.get_race_detail(race.race_id)
                qualifiers = (
                    detail.qualifiers if detail.qualifiers is not UNSET else []
                )
                return _qualifiers_to_finishers(qualifiers)
        return None

    def _build_weights_from_race(
        self, finishers: list[RankedFinisher]
    ) -> tuple[list[int], list[int]]:
        """Compute the full `(uids, u16 weights)` vector for the metagraph.

        Pure passthrough to `build_metagraph_weight_vector` — kept as a
        method so tests can target the integration without exercising
        Backend.
        """
        metagraph_hotkeys = list(self.metagraph.hotkeys)
        # Audit: log finishers whose hotkeys aren't in the current
        # metagraph, since their weight is silently dropped by
        # `build_metagraph_weight_vector`. This is the deregistration
        # / uid-recycle case — expected, not an error.
        present = set(metagraph_hotkeys)
        missing = [f.miner_hotkey for f in finishers if f.miner_hotkey not in present]
        if missing:
            logging.info(
                f"{len(missing)} race finisher(s) not in current metagraph "
                f"(deregistered between race close and weight set), "
                f"weight skipped: {missing[:5]}{'…' if len(missing) > 5 else ''}"
            )
        return build_metagraph_weight_vector(
            finishers,
            metagraph_hotkeys=metagraph_hotkeys,
            t_top=self.t_top,
            t_burn=self.t_burn,
        )

    def _submit_weights(self, uids: list[int], weights: list[int]) -> None:
        """Push `uids` / `weights` to the chain. No retries — the loop's
        next tick will retry on transient blockchain failures."""
        self.subtensor.set_weights(
            netuid=self.netuid,
            wallet=self.wallet,
            uids=uids,
            weights=weights,
            wait_for_inclusion=True,
        )

    def _tick(self) -> None:
        """One iteration of the loop — race-based weight submission."""
        self.metagraph.sync()

        finishers: Optional[list[RankedFinisher]] = None
        try:
            finishers = self._fetch_race_finishers()
        except BackendError as e:
            # Don't crash the loop — skip this tick; the next one retries.
            if e.is_transient:
                logging.warning(f"Race fetch transient error, skipping tick: {e}")
            else:
                logging.error(f"Race fetch error, skipping tick: {e}")
            return

        if not finishers:
            logging.warning(
                "No completed race available — skipping weight submission for this tick"
            )
            return

        uids, weights = self._build_weights_from_race(finishers)
        non_zero = sum(1 for w in weights if w > 0)
        logging.info(
            f"Race-based weight vector: N={len(finishers)} finishers, "
            f"{non_zero} non-zero metagraph slots"
        )

        if not weights:
            logging.warning("Skipping weight update (empty metagraph)")
            return

        self._submit_weights(uids, weights)
        logging.info("Successfully set weights")

    def _run(self) -> None:
        """Background thread main loop."""
        while not self._stop_event.is_set():
            try:
                self._tick()
            except BackendError as e:
                if e.is_auth_error:
                    logging.error(f"Weight setting auth error (will retry): {e}")
                elif e.is_transient:
                    logging.warning(f"Weight setting transient error: {e}")
                else:
                    logging.error(f"Weight setting backend error: {e}")
            except Exception as e:
                logging.error(f"Weight setting failed: {type(e).__name__}: {e}")

            self._stop_event.wait(self.interval_seconds)
