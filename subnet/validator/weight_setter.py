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
    apply_weight_overlay,
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

    # How far back to scan `get_race_history` for the most recent
    # `RACE_COMPLETE`. The newest race may be `QUALIFYING_OPEN` or
    # `RACE_RUNNING` for ~24h of every cycle; we still want to protect
    # last race's finishers during that window. 5 covers typical race
    # cadence (one in-progress + a handful of completed) without paging.
    _RACE_HISTORY_SCAN_LIMIT = 5

    # Fallback burn rate when Backend policy is unreachable. Matches the
    # historical default; ops can flip the live value via Backend env
    # without a validator release.
    _DEFAULT_BURN_RATE = 0.75

    def __init__(
        self,
        backend_client: BackendClient,
        subtensor,
        metagraph,
        wallet,
        netuid: int,
        interval_seconds: int = 300,
        t_burn_fallback: float = _DEFAULT_BURN_RATE,
    ):
        self.backend_client = backend_client
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.wallet = wallet
        self.netuid = netuid
        self.interval_seconds = interval_seconds
        self.t_burn_fallback = t_burn_fallback

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

    def _fetch_top_state(self) -> tuple[Optional[str], float]:
        """Return `(top_hotkey, t_burn)` from `GET /v1/public/top`.

        `top_hotkey` is the canonical "current top miner" for emissions.
        None when no admin-designated top exists (fresh subnet, suite
        switch, or designated top was discarded) — in those cases the
        top share folds into the burn slot.

        `t_burn` is the live `emission_baseline_burn_rate` from the
        Backend policy block. Lets ops flip burn (CDK env var) without
        a validator release. On any Backend error OR a missing/invalid
        policy value, falls back to `self.t_burn_fallback`.
        """
        try:
            top = self.backend_client.get_top_miner()
        except BackendError as e:
            logging.warning(
                f"Top fetch failed, using fallback burn rate: {e}"
            )
            return None, self.t_burn_fallback

        hk = top.top_miner_hotkey
        top_hotkey = (
            str(hk) if hk is not None and hk is not UNSET else None
        )

        t_burn = self.t_burn_fallback
        policy = top.policy
        if policy is not None and policy is not UNSET:
            try:
                value = policy["emission_baseline_burn_rate"]
                t_burn = float(value)
                _validate_burn(t_burn)
            except (KeyError, TypeError, ValueError) as e:
                logging.warning(
                    f"Invalid emission_baseline_burn_rate in policy, "
                    f"using fallback {self.t_burn_fallback}: {e}"
                )
                t_burn = self.t_burn_fallback

        return top_hotkey, t_burn

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

    def _build_base_vector(
        self,
        standings,
        live_finishers: list[RankedFinisher],
        live_top: Optional[str],
        live_t_burn: float,
    ) -> tuple[list[int], list[int]]:
        """Build the base weight vector, preferring the epoch-pinned standings.

        When the Backend serves epoch-pinned standings (its
        ``epoch_pinned_standings_enabled`` flag is on), every validator in the
        epoch builds from the SAME snapshot and so agrees regardless of tick
        phase — the ORO-1704 fix. Falls back to this validator's own live
        standings when the Backend serves none, serves an empty snapshot, or the
        payload is unusable (ORO-1704 D1) — the pin must never wedge weights or
        burn an epoch. Empty pinned finishers degrade to live specifically
        because reaching here means the live path DID find finishers (guarded in
        ``_tick``), so an empty snapshot is snapshot lag / an epoch transition,
        not a genuinely empty race.
        """
        if standings is not None:
            try:
                p_fin, p_top, p_burn = _standings_to_inputs(standings)
                if p_fin:
                    logging.info(
                        f"Weight vector from epoch-pinned standings: "
                        f"N={len(p_fin)} finishers, "
                        f"top={p_top or '(none — top share burns)'}, "
                        f"t_burn={p_burn:.3f}"
                    )
                    return self._build_weights_from_race(p_fin, p_top, p_burn)
                logging.warning(
                    "epoch-pinned standings empty; using live standings"
                )
            except Exception as e:  # noqa: BLE001 — never wedge weights; fall back to live
                logging.warning(
                    f"epoch-pinned standings unusable; using live standings: {e}"
                )

        logging.info(
            f"Weight vector from live standings: N={len(live_finishers)} "
            f"finishers, top={live_top or '(none — top share burns)'}, "
            f"t_burn={live_t_burn:.3f}"
        )
        return self._build_weights_from_race(
            live_finishers, live_top, live_t_burn
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

        top_hotkey, t_burn = self._fetch_top_state()

        # One participation-gated fetch → overlay + epoch-pinned standings for
        # THIS epoch (ORO-1704), so they can't straddle an epoch boundary.
        salt: WeightSalt = self.backend_client.fetch_weight_salt()

        # Prefer the epoch-pinned base vector (all validators in the epoch agree);
        # fall back to this validator's live standings when none are served.
        uids, weights = self._build_base_vector(
            salt.epoch_standings, finishers, top_hotkey, t_burn
        )

        # Apply any Backend-provided supplementary weight assignments to the base
        # we submit. An empty overlay leaves the base unchanged — the safe default
        # every validator in the same state also produces.
        overlay = salt.overlay
        if overlay:
            weights = apply_weight_overlay(weights, overlay)
            logging.info(
                f"Applied {len(overlay)} supplementary weight assignment(s)"
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
