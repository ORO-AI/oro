import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from oro_sdk.models import TopAgentResponse

from validator.weight_distribution import compute_pinned_weights
from validator.weight_setter import WeightSetterThread


def _empty_history():
    """A `get_race_history` response with no races — drives the fallback path."""
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
    """Integration tests for WeightSetterThread.

    Uses fixtures from conftest.py.
    """

    @pytest.fixture
    def mock_backend_client(self, mock_backend_client_with_top_miner):
        """Top-miner fixture extended with an empty race history so tests
        that don't set up a race fall through to the fallback path."""
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

    # --- top-miner fallback (no completed race) ---

    def test_fallback_burns_when_top_miner_not_in_metagraph(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
        mock_backend_client.get_top_miner.return_value = TopAgentResponse(
            suite_id=789,
            top_agent_version_id=UUID("87654321-4321-4321-4321-210987654321"),
            top_miner_hotkey="5UnknownHotkey...",
            top_score=0.92,
            computed_at=datetime.now(),
            emission_weight=1.0,
        )
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

        mock_subtensor.set_weights.assert_called()
        weights = mock_subtensor.set_weights.call_args.kwargs["weights"]
        _, burn_u16 = compute_pinned_weights(0.25, 0.75, tail_sum=0)
        assert weights[0] == burn_u16
        assert all(w == 0 for w in weights[1:])

    def test_fallback_top_miner_full_emission(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
        """emission_weight == 1.0 → top miner gets full top_u16, burn slot full burn_u16."""
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

        weights = mock_subtensor.set_weights.call_args.kwargs["weights"]
        top_u16, burn_u16 = compute_pinned_weights(0.25, 0.75, tail_sum=0)
        assert weights[0] == burn_u16  # uid 0 = burn
        assert weights[1] == top_u16  # 5GrwvaEF... index in fixture
        assert weights[2] == 0

    def test_fallback_emission_weight_partial(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
        """emission_weight=0.5 scales top slot to half top_u16; burn slot keeps full share."""
        mock_backend_client.get_top_miner.return_value = TopAgentResponse(
            suite_id=789,
            top_agent_version_id=UUID("87654321-4321-4321-4321-210987654321"),
            top_miner_hotkey="5GrwvaEF...",
            top_score=0.92,
            computed_at=datetime.now(),
            emission_weight=0.5,
        )
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

        weights = mock_subtensor.set_weights.call_args.kwargs["weights"]
        top_u16, burn_u16 = compute_pinned_weights(0.25, 0.75, tail_sum=0)
        assert weights[0] == burn_u16
        assert weights[1] == round(top_u16 * 0.5)

    def test_continues_on_error(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
        mock_backend_client.get_top_miner.side_effect = [
            Exception("Network error"),
            TopAgentResponse(
                suite_id=789,
                top_agent_version_id=UUID("87654321-4321-4321-4321-210987654321"),
                top_miner_hotkey="5GrwvaEF...",
                top_score=0.92,
                computed_at=datetime.now(),
                emission_weight=1.0,
            ),
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

        assert mock_backend_client.get_top_miner.call_count >= 2

    # --- race-based path ---

    def test_race_path_distributes_to_top_half(
        self, mock_backend_client, mock_subtensor, mock_wallet
    ):
        """With a completed race of 6 finishers, the top 3 (floor(6/2)) get
        non-zero u16 weights and the bottom 3 get 0."""
        # 6 finishers, scores descending; metagraph indexes them after the burn uid.
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
        # K=3, tail = [2, 1] → tail_sum = 3.
        top_u16, burn_u16 = compute_pinned_weights(0.25, 0.75, tail_sum=3)
        # Burn slot.
        assert weights[0] == burn_u16
        # Rank 1 finisher — uid 1 in this metagraph.
        assert weights[1] == top_u16
        # Ranks 2..K (K=3) — taper K-1, K-2 = 2, 1.
        assert weights[2] == 2
        assert weights[3] == 1
        # Bottom half — zero.
        assert weights[4] == 0
        assert weights[5] == 0
        assert weights[6] == 0

    def test_race_path_falls_back_when_no_completed_race_in_history(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
        """Every race in history is in-progress → fall back to top-miner path.

        Only happens on a fresh subnet or after a race-system rollback.
        """
        mock_backend_client.get_race_history.return_value = _history_with_races(
            [
                _race_with_status(uuid4(), "QUALIFYING_OPEN"),
                _race_with_status(uuid4(), "RACE_RUNNING"),
            ]
        )

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

        # `get_race_detail` should never be called when no race is complete.
        mock_backend_client.get_race_detail.assert_not_called()
        mock_backend_client.get_top_miner.assert_called()

    def test_race_path_skips_in_progress_and_uses_prior_completed_race(
        self, mock_backend_client, mock_subtensor, mock_wallet
    ):
        """Newest race is `QUALIFYING_OPEN` (current cycle) — we still want
        last race's finishers protected. Walk the history and use the most
        recent `RACE_COMPLETE`.
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
        # Detail call should target the completed race, not the in-progress one.
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

        # Loop may tick more than once at this interval; what matters is
        # the call always targeted the completed race id, never the
        # in-progress one.
        assert mock_backend_client.get_race_detail.call_count >= 1
        for call in mock_backend_client.get_race_detail.call_args_list:
            assert call.args == (completed_id,) or call.kwargs == {"race_id": completed_id}
        weights = mock_subtensor.set_weights.call_args.kwargs["weights"]
        # K=3, tail = [2, 1] → tail_sum = 3.
        top_u16, _ = compute_pinned_weights(0.25, 0.75, tail_sum=3)
        assert weights[1] == top_u16  # rank-1 finisher protected
