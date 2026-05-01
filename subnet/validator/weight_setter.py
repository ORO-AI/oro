"""Background thread for periodic weight updates.

Builds a deterministic top-50% race-entrant weight vector via
`weight_distribution.build_metagraph_weight_vector` and submits it to the
chain. Determinism across validators is load-bearing for Yuma consensus on
subnet 15 (`kappa = 0.5`).

Falls back to the prior top-miner-only behaviour while no completed race
is available (early subnet boot, race system temporarily unavailable).
"""

import threading
from typing import Optional

from bittensor.utils.btlogging import logging
from oro_sdk.types import UNSET

from .backend_client import BackendClient, BackendError
from .weight_distribution import (
    RankedEntrant,
    build_metagraph_weight_vector,
    compute_top_burn_weights,
)


def _qualifiers_to_entrants(qualifiers) -> list[RankedEntrant]:
    """Reduce SDK `RaceQualifierPublic` records to the ranking inputs.

    Skips entries without a `race_score` or `miner_hotkey` — those are not
    eligible for weight allocation. Drops `None` and `Unset` defensively
    so a partial-data Backend response can't poison the ranking.
    """
    entrants: list[RankedEntrant] = []
    for q in qualifiers:
        score = q.race_score
        if score is None or score is UNSET:
            continue
        hotkey = q.miner_hotkey
        if not hotkey or hotkey is UNSET:
            continue
        entrants.append(
            RankedEntrant(
                miner_hotkey=str(hotkey),
                agent_version_id=str(q.agent_version_id),
                race_score=float(score),
            )
        )
    return entrants


class WeightSetterThread:
    """Periodically computes the top-50% weight vector and submits it on-chain.

    Runs in a background thread, independent of the evaluation loop.
    """

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
        burn_uid: int = 0,
    ):
        self.backend_client = backend_client
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.wallet = wallet
        self.netuid = netuid
        self.interval_seconds = interval_seconds
        self.t_top = t_top
        self.t_burn = t_burn
        self.burn_uid = burn_uid

        # Fail fast on misconfiguration — the validator process should not
        # start setting weights with invalid ratios.
        compute_top_burn_weights(t_top, t_burn)

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

    def _fetch_race_entrants(self) -> Optional[list[RankedEntrant]]:
        """Return entrants from the latest completed race, or None.

        Returns None when there is no completed race yet, or when fetching
        the race details fails — the caller falls back to the top-miner
        path so weight setting is never silently skipped on transient errors.
        """
        history = self.backend_client.get_race_history(limit=1)
        races = history.races if history.races is not UNSET else []
        if not races:
            return None
        latest = races[0]
        if str(latest.status) != "RACE_COMPLETE":
            return None
        detail = self.backend_client.get_race_detail(latest.race_id)
        qualifiers = detail.qualifiers if detail.qualifiers is not UNSET else []
        return _qualifiers_to_entrants(qualifiers)

    def _build_weights_from_race(
        self, entrants: list[RankedEntrant]
    ) -> tuple[list[int], list[int]]:
        """Compute the full `(uids, u16 weights)` vector for the metagraph.

        Pure passthrough to `build_metagraph_weight_vector` — kept as a
        method so tests can target the integration without exercising
        Backend.
        """
        return build_metagraph_weight_vector(
            entrants,
            metagraph_hotkeys=list(self.metagraph.hotkeys),
            t_top=self.t_top,
            t_burn=self.t_burn,
            burn_uid=self.burn_uid,
        )

    def _build_weights_top_miner_fallback(
        self, top_miner_hotkey: str, emission_wt: float
    ) -> tuple[list[int], list[int]]:
        """Legacy fallback: top miner gets one slot, rest goes to burn uid.

        Used while the subnet has no completed race yet, or when the race
        endpoint is unavailable. Honours Backend's `emission_weight` so the
        same operator-knob still controls the top/burn split during the
        transition window.
        """
        n = len(self.metagraph.hotkeys)
        if n == 0:
            return [], []

        top_u16, burn_u16 = compute_top_burn_weights(self.t_top, self.t_burn)

        weights = [0] * n
        if 0 <= self.burn_uid < n:
            weights[self.burn_uid] = burn_u16

        try:
            top_idx = self.metagraph.hotkeys.index(top_miner_hotkey)
        except ValueError:
            logging.warning(
                f"Top miner {top_miner_hotkey} not found in metagraph, "
                "burning all emissions to uid 0"
            )
            return list(range(n)), weights

        # `emission_weight` < 1 means Backend asked for a smaller-than-
        # configured top share (e.g. challenger margin not yet earned).
        # Scale the top u16 by it; the burn slot keeps its full share.
        scaled_top = int(round(top_u16 * emission_wt))
        if self.burn_uid == top_idx:
            weights[top_idx] += scaled_top
        else:
            weights[top_idx] = scaled_top

        return list(range(n)), weights

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
        """One iteration of the loop — race-based path with top-miner fallback."""
        self.metagraph.sync()

        entrants: Optional[list[RankedEntrant]] = None
        try:
            entrants = self._fetch_race_entrants()
        except BackendError as e:
            # Don't crash the loop — fall back to the top-miner path so a
            # race-endpoint outage doesn't stop weight setting.
            if e.is_transient:
                logging.warning(f"Race fetch transient error, using fallback: {e}")
            else:
                logging.error(f"Race fetch error, using fallback: {e}")

        if entrants:
            uids, weights = self._build_weights_from_race(entrants)
            non_zero = sum(1 for w in weights if w > 0)
            logging.info(
                f"Race-based weight vector: N={len(entrants)} entrants, "
                f"{non_zero} non-zero metagraph slots"
            )
        else:
            top = self.backend_client.get_top_miner()
            emission_wt = (
                top.emission_weight
                if isinstance(top.emission_weight, (int, float))
                else 1.0
            )
            logging.info(
                f"No completed race available — using top-miner fallback "
                f"(top={top.top_miner_hotkey}, emission_weight={emission_wt:.3f})"
            )
            uids, weights = self._build_weights_top_miner_fallback(
                top.top_miner_hotkey, emission_wt=emission_wt
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
