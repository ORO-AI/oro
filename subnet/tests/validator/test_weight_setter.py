import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from oro_sdk.types import UNSET

from oro_sdk.models.epoch_standings import EpochStandings
from oro_sdk.models.pinned_finisher import PinnedFinisher

from validator.backend_client import BackendError, WeightSalt
from validator.weight_distribution import compute_pinned_weights
from validator.weight_setter import WeightSetterThread, _qualifiers_to_finishers


def _standings(finishers: list[dict], top_hotkey, t_burn=0.75):
    """EpochStandings payload from finisher dicts — the authed weight source."""
    return EpochStandings(
        top_hotkey=top_hotkey,
        t_burn=t_burn,
        finishers=[
            PinnedFinisher(
                miner_hotkey=f["miner_hotkey"],
                agent_version_id=f["agent_version_id"],
                race_score=f["race_score"],
            )
            for f in finishers
        ],
    )


def _salt(finishers, top_hotkey, overlay=None):
    return WeightSalt(
        overlay=overlay or {},
        epoch_standings=_standings(finishers, top_hotkey),
    )


def _race_detail(qualifiers: list[dict]):
    """Build a mock RaceDetailResponse with the supplied qualifier dicts.

    Each dict needs `miner_hotkey`, `agent_version_id`, `race_score` and may
    optionally provide `is_discarded` (defaults to False — set True to
    exercise the ORO-1111 filter path) or `eliminated_at` (defaults to None —
    set to a timestamp to exercise the survivor-set filter path).

    Both flags are set explicitly: a bare `MagicMock` auto-returns a truthy
    child mock for any unset attribute, which would make every qualifier
    look eliminated/discarded.
    """
    detail = MagicMock()
    detail.qualifiers = []
    for q in qualifiers:
        m = MagicMock()
        m.miner_hotkey = q["miner_hotkey"]
        m.agent_version_id = q["agent_version_id"]
        m.race_score = q["race_score"]
        m.is_discarded = q.get("is_discarded", False)
        m.eliminated_at = q.get("eliminated_at", None)
        detail.qualifiers.append(m)
    return detail


class TestWeightSetterThread:
    @pytest.fixture
    def mock_wallet(self, mock_wallet_simple):
        return mock_wallet_simple

    # --- thread lifecycle ---

    def test_start_creates_thread(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
        setter = WeightSetterThread(
            backend_client=mock_backend_client,
            subtensor=mock_subtensor,
            metagraph=mock_metagraph,
            wallet=mock_wallet,
            netuid=1,
            interval_seconds=1,
        )
        setter.start()
        assert setter._thread is not None
        assert setter._thread.is_alive()
        setter.stop()

    def test_stop_terminates_thread(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
        setter = WeightSetterThread(
            backend_client=mock_backend_client,
            subtensor=mock_subtensor,
            metagraph=mock_metagraph,
            wallet=mock_wallet,
            netuid=1,
            interval_seconds=1,
        )
        setter.start()
        setter.stop()
        assert not setter._thread.is_alive()

    def test_invalid_burn_fallback_raises_at_construction(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
        with pytest.raises(ValueError):
            WeightSetterThread(
                backend_client=mock_backend_client,
                subtensor=mock_subtensor,
                metagraph=mock_metagraph,
                wallet=mock_wallet,
                netuid=1,
                t_burn_fallback=1.1,  # > 1
            )

    # --- race-based path (only path remaining) ---

    def test_no_race_skips_submission(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
        """No completed race in history → skip the tick, do not submit weights."""
        setter = WeightSetterThread(
            backend_client=mock_backend_client,
            subtensor=mock_subtensor,
            metagraph=mock_metagraph,
            wallet=mock_wallet,
            netuid=1,
            interval_seconds=0.1,
        )
        setter.start()
        time.sleep(0.15)
        setter.stop()

        mock_subtensor.set_weights.assert_not_called()

    def test_continues_on_backend_error(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
        """A raise inside the tick is caught and the loop keeps ticking."""
        mock_backend_client.fetch_weight_salt.side_effect = [
            BackendError("Network error"),  # no status/sdk_error → transient
            WeightSalt(overlay={}, epoch_standings=None),
            WeightSalt(overlay={}, epoch_standings=None),
        ]
        setter = WeightSetterThread(
            backend_client=mock_backend_client,
            subtensor=mock_subtensor,
            metagraph=mock_metagraph,
            wallet=mock_wallet,
            netuid=1,
            interval_seconds=0.1,
        )
        setter.start()
        time.sleep(0.25)
        setter.stop()

        assert mock_backend_client.fetch_weight_salt.call_count >= 2

    def test_race_path_distributes_to_survivors(
        self, mock_backend_client, mock_subtensor, mock_wallet
    ):
        """6 survivors, all in metagraph: the top slot goes to 5HK0 and the
        other 5 survivors all get a taper entry (5,4,3,2,1) — no bottom cut.
        With every survivor present, no drift correction needed — top_u16
        lands at 25% of the submitted vector exactly. Driven by the authed
        epoch-pinned standings (ORO-1802: the only source).
        """
        finishers = [
            {"miner_hotkey": f"5HK{i}", "agent_version_id": str(uuid4()), "race_score": 0.9 - i * 0.05}
            for i in range(6)
        ]

        metagraph = MagicMock()
        metagraph.hotkeys = ["5BurnUid"] + [e["miner_hotkey"] for e in finishers]
        metagraph.uids = list(range(len(metagraph.hotkeys)))
        metagraph.block = 100

        mock_backend_client.fetch_weight_salt.return_value = _salt(finishers, "5HK0")

        setter = WeightSetterThread(
            backend_client=mock_backend_client,
            subtensor=mock_subtensor,
            metagraph=metagraph,
            wallet=mock_wallet,
            netuid=1,
            interval_seconds=0.1,
        )
        setter.start()
        time.sleep(0.15)
        setter.stop()

        weights = mock_subtensor.set_weights.call_args.kwargs["weights"]
        # Survivor tail (ranks 2..6) = [5, 4, 3, 2, 1] → tail_sum_actual = 15.
        top_u16, burn_u16 = compute_pinned_weights(0.75, tail_sum=15)
        assert weights[0] == burn_u16
        assert weights[1] == top_u16
        assert weights[2] == 5
        assert weights[3] == 4
        assert weights[4] == 3
        assert weights[5] == 2
        assert weights[6] == 1

    def test_drift_correction_when_protected_finishers_deregistered(
        self, mock_backend_client, mock_subtensor, mock_wallet
    ):
        """When some protected finishers are missing from the metagraph,
        top_u16 / burn_u16 are recomputed from the *actual* tail_sum so the
        top miner's normalised share stays at exactly t_top.
        """
        # 6 survivors; only rank-1 (5HK0, the top) is in the metagraph, so the
        # entire survivor tail (5HK1..5HK5) is deregistered → tail_sum → 0.
        finishers = [
            {"miner_hotkey": f"5HK{i}", "agent_version_id": str(uuid4()), "race_score": 0.9 - i * 0.05}
            for i in range(6)
        ]
        metagraph = MagicMock()
        metagraph.hotkeys = ["5BurnUid", "5HK0"]  # only burn + rank 1
        metagraph.uids = list(range(len(metagraph.hotkeys)))
        metagraph.block = 100

        mock_backend_client.fetch_weight_salt.return_value = _salt(finishers, "5HK0")

        setter = WeightSetterThread(
            backend_client=mock_backend_client,
            subtensor=mock_subtensor,
            metagraph=metagraph,
            wallet=mock_wallet,
            netuid=1,
            interval_seconds=0.1,
        )
        setter.start()
        time.sleep(0.15)
        setter.stop()

        weights = mock_subtensor.set_weights.call_args.kwargs["weights"]
        # Tail dereg'd → tail_sum_actual = 0 → recompute pins top, burn.
        top_u16, burn_u16 = compute_pinned_weights(0.75, tail_sum=0)
        assert weights[0] == burn_u16
        assert weights[1] == top_u16
        # Submitted top share matches t_top exactly.
        total = sum(weights)
        top_share = weights[1] / total
        assert abs(top_share - 0.25) < 1e-3


class TestQualifiersToFinishersIsDiscarded:
    """ORO-1111: drop is_discarded=True qualifiers from the finisher set."""

    @staticmethod
    def _q(hotkey: str, score: float, *, is_discarded=False, with_field: bool = True):
        attrs = {
            "miner_hotkey": hotkey,
            "agent_version_id": uuid4(),
            "race_score": score,
        }
        if with_field:
            attrs["is_discarded"] = is_discarded
        return SimpleNamespace(**attrs)

    def test_drops_discarded_keeps_non_discarded(self):
        qualifiers = [
            self._q("5HKkept", 0.9, is_discarded=False),
            self._q("5HKdiscarded", 0.85, is_discarded=True),
            self._q("5HKalsoKept", 0.8, is_discarded=False),
        ]
        finishers = _qualifiers_to_finishers(qualifiers)
        hotkeys = {f.miner_hotkey for f in finishers}
        assert hotkeys == {"5HKkept", "5HKalsoKept"}

    def test_missing_is_discarded_field_defaults_to_false(self):
        """Forward-compat with pre-ORO-1111 SDK builds: missing field = keep."""
        qualifiers = [self._q("5HKlegacy", 0.7, with_field=False)]
        finishers = _qualifiers_to_finishers(qualifiers)
        assert [f.miner_hotkey for f in finishers] == ["5HKlegacy"]

    def test_unset_is_discarded_treated_as_false(self):
        qualifiers = [self._q("5HKunset", 0.6, is_discarded=UNSET)]
        finishers = _qualifiers_to_finishers(qualifiers)
        assert [f.miner_hotkey for f in finishers] == ["5HKunset"]


class TestQualifiersToFinishersEliminated:
    """Survivor-set filter: drop `eliminated_at` qualifiers (bottom-cut at
    race end) so the protected tail tracks the survivor set, not a fixed
    top-N. An agent that finished but was eliminated keeps its non-null
    `race_score`, so only this filter removes it."""

    @staticmethod
    def _q(hotkey: str, score: float, *, eliminated_at=None, with_field: bool = True):
        attrs = {
            "miner_hotkey": hotkey,
            "agent_version_id": uuid4(),
            "race_score": score,
        }
        if with_field:
            attrs["eliminated_at"] = eliminated_at
        return SimpleNamespace(**attrs)

    def test_drops_eliminated_keeps_survivors(self):
        elim_ts = datetime(2026, 7, 8, tzinfo=timezone.utc)
        qualifiers = [
            self._q("5HKsurvivor", 0.9),
            self._q("5HKeliminated", 0.85, eliminated_at=elim_ts),
            self._q("5HKalsoSurvivor", 0.8),
        ]
        finishers = _qualifiers_to_finishers(qualifiers)
        assert {f.miner_hotkey for f in finishers} == {"5HKsurvivor", "5HKalsoSurvivor"}

    def test_missing_eliminated_at_field_defaults_to_kept(self):
        """Forward-compat with SDK builds pre-dating `eliminated_at`: missing = keep."""
        qualifiers = [self._q("5HKlegacy", 0.7, with_field=False)]
        finishers = _qualifiers_to_finishers(qualifiers)
        assert [f.miner_hotkey for f in finishers] == ["5HKlegacy"]

    def test_unset_eliminated_at_treated_as_kept(self):
        qualifiers = [self._q("5HKunset", 0.6, eliminated_at=UNSET)]
        finishers = _qualifiers_to_finishers(qualifiers)
        assert [f.miner_hotkey for f in finishers] == ["5HKunset"]
