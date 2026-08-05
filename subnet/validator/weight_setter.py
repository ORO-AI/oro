"""Background thread for periodic weight updates.

Builds a deterministic survivor-set race-finisher weight vector via
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

from .backend_client import BackendClient, BackendError, WeightSalt
from .weight_distribution import (
    RankedFinisher,
    _validate_burn,
    build_metagraph_weight_vector,
    compute_pinned_weights,
)


def _qualifiers_to_finishers(qualifiers) -> list[RankedFinisher]:
    """Reduce SDK `RaceQualifierPublic` records to ranked finishers.

    A "finisher" here is the deregistration-protection *survivor* set:
    a qualifier with a non-null `race_score` that was NOT eliminated or
    discarded — i.e. the agent ran the race to completion, got scored,
    and advances to the next race. Qualifiers with `race_score=null`
    (DNF, never executed) are dropped: they did not finish. A
    `race_score=0.0` survivor is still a finisher (completed but scored
    zero) and takes the smallest taper slot.

    Also drops entries without a `miner_hotkey` defensively so a
    partial-data Backend response can't poison the ranking.

    Discarded agents (admin- or auto-discarded, `is_discarded`) AND
    eliminated agents (bottom-cut at race end, `eliminated_at`) are
    dropped so they stop earning emissions via the rank-1 fallback or
    the protected tail. Tying the protected set to the survivor set
    keeps it aligned with the elimination fraction automatically — there
    is no second top-N threshold to hand-sync when that fraction moves.
    Missing or `UNSET` fields default to not-excluded for forward
    compatibility with older Backend builds that pre-date them.
    """
    finishers: list[RankedFinisher] = []
    for q in qualifiers:
        score = q.race_score
        if score is None or score is UNSET:
            continue
        hotkey = q.miner_hotkey
        if not hotkey or hotkey is UNSET:
            continue
        is_discarded = getattr(q, "is_discarded", False)
        if is_discarded is UNSET:
            is_discarded = False
        if is_discarded:
            logging.info(
                "Dropping discarded agent from race finishers: "
                f"hotkey={hotkey} agent_version_id={q.agent_version_id}"
            )
            continue
        eliminated_at = getattr(q, "eliminated_at", None)
        if eliminated_at is UNSET:
            eliminated_at = None
        if eliminated_at is not None:
            logging.info(
                "Dropping eliminated agent from race finishers: "
                f"hotkey={hotkey} agent_version_id={q.agent_version_id}"
            )
            continue
        finishers.append(
            RankedFinisher(
                miner_hotkey=str(hotkey),
                agent_version_id=str(q.agent_version_id),
                race_score=float(score),
            )
        )
    return finishers


def _standings_to_inputs(
    standings,
) -> tuple[list[RankedFinisher], Optional[str], float]:
    """Map an epoch-pinned ``EpochStandings`` to the same ``(finishers,
    top_hotkey, t_burn)`` inputs the live path produces (ORO-1704).

    Finishers go through the SAME ``_qualifiers_to_finishers`` the live path uses
    (``PinnedFinisher`` carries the identical ``miner_hotkey`` /
    ``agent_version_id`` / ``race_score`` fields), so the pinned base vector is
    byte-identical to the live one after re-sorting. ``t_burn`` is range-checked
    with the same ``_validate_burn`` the live path uses, but the failure modes
    differ: the live top-state path falls back to ``t_burn_fallback`` on an
    invalid policy value, whereas this raises — ``_resolve_pinned`` then catches
    it and degrades to live, so the safe-fallback outcome is the same.
    """
    raw = standings.finishers if standings.finishers is not UNSET else []
    finishers = _qualifiers_to_finishers(raw or [])
    top = standings.top_hotkey
    top_hotkey = str(top) if top is not None and top is not UNSET else None
    t_burn = float(standings.t_burn)
    _validate_burn(t_burn)
    return finishers, top_hotkey, t_burn


class WeightSetterThread:
    """Periodically computes the survivor-set weight vector and submits it on-chain.

    Runs in a background thread, independent of the evaluation loop.
    """

    # Construction-time validation only: `t_burn_fallback` is fed to
    # `compute_pinned_weights` in __init__ to fail fast on a misconfigured value.
    # It is NOT a runtime fallback anymore — with the public /top path removed,
    # an unusable standings payload is a miss (retain/burn), not a fallback.
    _DEFAULT_BURN_RATE = 0.75

    # Default tick cadence. 22 min sits just above the on-chain
    # WeightsSetRateLimit (100 blocks * 12s = 20 min), so every attempt
    # actually lands instead of being rejected — ~3 real weight-sets per
    # 72-min epoch. Ops can override via ORO_WEIGHT_UPDATE_INTERVAL.
    _DEFAULT_INTERVAL_SECONDS = 1320

    def __init__(
        self,
        backend_client: BackendClient,
        subtensor,
        metagraph,
        wallet,
        netuid: int,
        interval_seconds: int = _DEFAULT_INTERVAL_SECONDS,
        t_burn_fallback: float = _DEFAULT_BURN_RATE,
        reveal_period_epochs: int = 1,
    ):
        self.backend_client = backend_client
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.wallet = wallet
        self.netuid = netuid
        self.interval_seconds = interval_seconds
        self.t_burn_fallback = t_burn_fallback
        self.reveal_period_epochs = reveal_period_epochs

        # Epoch-anchored burn state (ORO-1802). We build weights only from the
        # authed epoch-pinned standings; there is NO public /top fallback. On a
        # transient miss we retain last-good (skip submit); only when a FULL
        # epoch has elapsed with no successful set do we burn to UID 0. Anchored
        # on the chain epoch index (block // (tempo * reveal_period)) so it is
        # independent of tick cadence.
        self._last_success_epoch: Optional[int] = None

        # Fail fast on misconfiguration of the fallback value. The live
        # value pulled from Backend is validated each tick by
        # `compute_pinned_weights`.
        compute_pinned_weights(t_burn_fallback, tail_sum=0)

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

    def _current_epoch(self) -> Optional[int]:
        """Chain commit-reveal epoch index, or None if the chain is unreadable.

        `epoch = block // (tempo * reveal_period_epochs)` — identical to the
        Backend's `epoch_index_for_block`, so the validator's burn anchor lines
        up with the epoch the pinned standings were snapshotted for. Reads tempo
        from chain each call (not cached) so a tempo change on-chain can't drift
        the anchor away from the Backend's.
        """
        try:
            block = int(self.metagraph.block)
            tempo = int(self.subtensor.tempo(self.netuid))
            period = max(tempo, 1) * max(self.reveal_period_epochs, 1)
            return block // period
        except Exception as e:  # noqa: BLE001 — unreadable chain → no anchor
            logging.warning(f"Could not derive current epoch: {e}")
            return None

    def _build_weights_from_race(
        self,
        finishers: list[RankedFinisher],
        top_hotkey: Optional[str],
        t_burn: float,
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
        if top_hotkey is not None and top_hotkey not in present:
            logging.warning(
                f"Designated top miner {top_hotkey} not in current metagraph; "
                f"top share folds into burn slot"
            )
        return build_metagraph_weight_vector(
            finishers,
            metagraph_hotkeys=metagraph_hotkeys,
            t_burn=t_burn,
            top_hotkey=top_hotkey,
        )

    def _build_from_standings(
        self, standings
    ) -> Optional[tuple[list[int], list[int]]]:
        """Build the weight vector from the authed epoch-pinned standings only.

        Returns the `(uids, weights)` vector when the standings carry a usable
        (non-empty) finisher set — every validator in the epoch builds from the
        SAME snapshot and so agrees regardless of tick phase (the ORO-1704 fix).

        Returns ``None`` (a "miss") when there are no standings, an empty
        snapshot (epoch-transition lag), or an unusable payload. There is NO
        public /top fallback (ORO-1802): the caller retains last-good weights on
        a transient miss and burns to UID 0 only after a full epoch of misses.
        """
        if standings is None:
            return None
        try:
            p_fin, p_top, p_burn = _standings_to_inputs(standings)
        except Exception as e:  # noqa: BLE001 — unusable payload → miss
            logging.warning(f"epoch-pinned standings unusable: {e}")
            return None
        if not p_fin:
            logging.warning("epoch-pinned standings empty (snapshot lag?)")
            return None
        logging.info(
            f"Weight vector from epoch-pinned standings: N={len(p_fin)} "
            f"finishers, top={p_top or '(none — top share burns)'}, "
            f"t_burn={p_burn:.3f}"
        )
        return self._build_weights_from_race(p_fin, p_top, p_burn)

    def _burn_vector(self) -> tuple[list[int], list[int]]:
        """Full burn-to-UID-0 vector (no finishers, no top, t_burn=1.0)."""
        return self._build_weights_from_race([], None, 1.0)

    def _submit_weights(self, uids: list[int], weights: list[int]) -> bool:
        """Push `uids` / `weights` to the chain and return whether it was accepted.

        No retries — the loop's next tick retries on transient failures. A
        rate-limited / rejected set returns False so the caller does NOT advance
        the burn anchor on a set that never landed.
        """
        result = self.subtensor.set_weights(
            netuid=self.netuid,
            wallet=self.wallet,
            uids=uids,
            weights=weights,
            wait_for_inclusion=True,
        )
        # Read the actual success signal. bittensor 10.x returns an
        # `ExtrinsicResponse` (has `.success`; its `__len__` is a constant 2 so a
        # bare `bool(result)` is ALWAYS True — the bug this guards against). Older
        # paths return a (success, message) tuple or a bare bool.
        if isinstance(result, tuple):
            return bool(result[0])
        if hasattr(result, "success"):
            return bool(result.success)
        return bool(result)

    def _tick(self) -> None:
        """One iteration of the loop — epoch-pinned weight submission.

        Weights come ONLY from the authed epoch-pinned standings (ORO-1802);
        there is no public /top fallback. On a transient miss we retain the
        last-good on-chain weights (skip submit); only when a full epoch has
        elapsed with no successful set do we burn to UID 0.
        """
        self.metagraph.sync()

        # One participation-gated fetch → overlay + epoch-pinned standings for
        # THIS epoch (ORO-1704), so they can't straddle an epoch boundary.
        # fetch_weight_salt never raises: on any error it returns an empty
        # overlay + None standings (i.e. a miss).
        salt: WeightSalt = self.backend_client.fetch_weight_salt()
        built = self._build_from_standings(salt.epoch_standings)

        if built is None:
            self._handle_miss()
            return

        uids, weights = built

        if not weights:
            logging.warning("Skipping weight update (empty metagraph)")
            return

        if not self._submit_weights(uids, weights):
            logging.warning("Weight set not accepted this tick (rate-limited?)")
            return
        # Record the epoch of this ACCEPTED set; the burn anchor measures
        # elapsed epochs from here.
        epoch = self._current_epoch()
        if epoch is not None:
            self._last_success_epoch = epoch
        logging.info("Successfully set weights")

    def _handle_miss(self) -> None:
        """No usable authed standings this tick. Retain last-good, or burn.

        Retain (skip submit → on-chain weights persist) until a FULL epoch has
        elapsed with no accepted set. Because a set can land late in epoch N,
        burning requires `epoch >= last_success + 2` — i.e. the whole of epoch
        N+1 passed with no success. Merely ticking into N+1 is a partial epoch
        (a transient miss), so it retains. The first ever miss (no prior success)
        establishes the epoch baseline and retains, giving a fresh subnet grace.
        """
        epoch = self._current_epoch()
        if epoch is None:
            logging.warning(
                "No usable standings and chain epoch unknown; retaining last-good"
            )
            return
        if self._last_success_epoch is None:
            self._last_success_epoch = epoch
            logging.warning(
                "No usable standings; establishing epoch baseline, retaining last-good"
            )
            return
        if epoch < self._last_success_epoch + 2:
            logging.warning(
                "No usable standings this tick; retaining last-good "
                f"(epoch {epoch}, last success {self._last_success_epoch})"
            )
            return

        # A full epoch has elapsed with no accepted set → burn to UID 0.
        uids, weights = self._burn_vector()
        if not weights:
            logging.warning("Skipping burn (empty metagraph)")
            return
        logging.error(
            f"No usable standings for a full epoch "
            f"(last success epoch {self._last_success_epoch}, now {epoch}); "
            "burning to UID 0"
        )
        self._submit_weights(uids, weights)

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
