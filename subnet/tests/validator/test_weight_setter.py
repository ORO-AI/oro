import time
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from validator.weight_distribution import compute_pinned_weights
from validator.weight_setter import WeightSetterThread


def _empty_history():
    history = MagicMock()
    history.races = []
    return history


def _race_complete_history(race_id: UUID):
    history = MagicMock()
    race = MagicMock()
    race.race_id = race_id
    race.status = "RACE_COMPLETE"
    history.races = [race]
    return history


def _race_with_status(race_id: UUID, status: str):
    race = MagicMock()
    race.race_id = race_id
    race.status = status
    return race


def _history_with_races(races: list):
    history = MagicMock()
    history.races = races
    return history


def _race_detail(qualifiers: list[dict]):
    """Build a mock RaceDetailResponse with the supplied qualifier dicts.

    Each dict needs `miner_hotkey`, `agent_version_id`, `race_score`.
    """
    detail = MagicMock()
    detail.qualifiers = []
    for q in qualifiers:
        m = MagicMock()
        m.miner_hotkey = q["miner_hotkey"]
        m.agent_version_id = q["agent_version_id"]
        m.race_score = q["race_score"]
        detail.qualifiers.append(m)
    return detail


class TestWeightSetterThread:
    @pytest.fixture
    def mock_backend_client(self, mock_backend_client_with_top_miner):
        mock_backend_client_with_top_miner.get_race_history.return_value = (
            _empty_history()
        )
        return mock_backend_client_with_top_miner

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

    def test_invalid_ratio_raises_at_construction(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
        with pytest.raises(ValueError):
            WeightSetterThread(
                backend_client=mock_backend_client,
                subtensor=mock_subtensor,
                metagraph=mock_metagraph,
                wallet=mock_wallet,
                netuid=1,
                t_top=0.6,
                t_burn=0.5,  # sum > 1
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
        """Transient race fetch error skips the tick but doesn't crash the loop."""
        mock_backend_client.get_race_history.side_effect = [
            Exception("Network error"),
            _empty_history(),
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

        assert mock_backend_client.get_race_history.call_count >= 2

    def test_race_path_distributes_to_top_half(
        self, mock_backend_client, mock_subtensor, mock_wallet
    ):
        """6 finishers, all in metagraph: top 3 (floor(6/2)) get u16; bottom 3 zero.
        With every protected finisher present, no drift correction needed —
        top_u16 lands at 25% of the submitted vector exactly.
        """
        finishers = [
            {"miner_hotkey": f"5HK{i}", "agent_version_id": str(uuid4()), "race_score": 0.9 - i * 0.05}
            for i in range(6)
        ]

        metagraph = MagicMock()
        metagraph.hotkeys = ["5BurnUid"] + [e["miner_hotkey"] for e in finishers]
        metagraph.uids = list(range(len(metagraph.hotkeys)))

        race_id = uuid4()
        mock_backend_client.get_race_history.return_value = _race_complete_history(race_id)
        mock_backend_client.get_race_detail.return_value = _race_detail(finishers)

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
        # K=3, tail (ranks 2..3) = [2, 1] → tail_sum_actual = 3.
        top_u16, burn_u16 = compute_pinned_weights(0.25, 0.75, tail_sum=3)
        assert weights[0] == burn_u16
        assert weights[1] == top_u16
        assert weights[2] == 2
        assert weights[3] == 1
        assert weights[4] == 0
        assert weights[5] == 0
        assert weights[6] == 0

    def test_race_path_skips_in_progress_and_uses_prior_completed_race(
        self, mock_backend_client, mock_subtensor, mock_wallet
    ):
        """Newest race is in-progress — walk history and use the most recent
        RACE_COMPLETE so the prior race's finishers stay protected during
        the current cycle.
        """
        finishers = [
            {"miner_hotkey": f"5HK{i}", "agent_version_id": str(uuid4()), "race_score": 0.9 - i * 0.05}
            for i in range(6)
        ]

        metagraph = MagicMock()
        metagraph.hotkeys = ["5BurnUid"] + [f["miner_hotkey"] for f in finishers]
        metagraph.uids = list(range(len(metagraph.hotkeys)))

        in_progress_id = uuid4()
        completed_id = uuid4()
        mock_backend_client.get_race_history.return_value = _history_with_races(
            [
                _race_with_status(in_progress_id, "QUALIFYING_OPEN"),
                _race_with_status(completed_id, "RACE_COMPLETE"),
            ]
        )
        mock_backend_client.get_race_detail.return_value = _race_detail(finishers)

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

        assert mock_backend_client.get_race_detail.call_count >= 1
        for call in mock_backend_client.get_race_detail.call_args_list:
            assert call.args == (completed_id,) or call.kwargs == {"race_id": completed_id}
        weights = mock_subtensor.set_weights.call_args.kwargs["weights"]
        top_u16, _ = compute_pinned_weights(0.25, 0.75, tail_sum=3)
        assert weights[1] == top_u16

    def test_drift_correction_when_protected_finishers_deregistered(
        self, mock_backend_client, mock_subtensor, mock_wallet
    ):
        """When some protected finishers are missing from the metagraph,
        top_u16 / burn_u16 are recomputed from the *actual* tail_sum so the
        top miner's normalised share stays at exactly t_top.
        """
        # 6 finishers; rank 2 (5HK1) and rank 3 (5HK2) are deregistered →
        # protected set drops from 3 (K=floor(6/2)) to 1 in the metagraph.
        finishers = [
            {"miner_hotkey": f"5HK{i}", "agent_version_id": str(uuid4()), "race_score": 0.9 - i * 0.05}
            for i in range(6)
        ]
        metagraph = MagicMock()
        metagraph.hotkeys = ["5BurnUid", "5HK0"]  # only burn + rank 1
        metagraph.uids = list(range(len(metagraph.hotkeys)))

        mock_backend_client.get_race_history.return_value = _race_complete_history(uuid4())
        mock_backend_client.get_race_detail.return_value = _race_detail(finishers)

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
        top_u16, burn_u16 = compute_pinned_weights(0.25, 0.75, tail_sum=0)
        assert weights[0] == burn_u16
        assert weights[1] == top_u16
        # Submitted top share matches t_top exactly.
        total = sum(weights)
        top_share = weights[1] / total
        assert abs(top_share - 0.25) < 1e-3
