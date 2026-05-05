import time
from datetime import datetime
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
        # Top missing AND emission_wt=1.0 → all to burn fallback.
        assert weights[0] == 65535
        assert all(w == 0 for w in weights[1:])

    def test_fallback_top_miner_full_emission(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
        """emission_weight=1.0 → 100% to top, 0% to burn (literal share)."""
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
        top_u16, burn_u16 = compute_pinned_weights(1.0, 0.0, tail_sum=0)
        assert weights[0] == burn_u16  # 0 — no burn share
        assert weights[1] == top_u16  # 65535 — full top share
        assert weights[2] == 0

    def test_fallback_emission_weight_quarter(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
        """emission_weight=0.25 → 25% top / 75% burn (matches pre-rewrite behaviour)."""
        mock_backend_client.get_top_miner.return_value = TopAgentResponse(
            suite_id=789,
            top_agent_version_id=UUID("87654321-4321-4321-4321-210987654321"),
            top_miner_hotkey="5GrwvaEF...",
            top_score=0.92,
            computed_at=datetime.now(),
            emission_weight=0.25,
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
        # Burn pinned at u16::MAX, top derived = 21845 → 25/75 split.
        assert weights[0] == burn_u16  # 65535
        assert weights[1] == top_u16  # 21845

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

    def test_shadow_mode_no_race_uses_fallback(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
        """shadow_mode=True with no completed race: still submits via fallback.

        Shadow mode must never silently stop weight setting; the legacy
        top-miner path stays live so prod doesn't lose vTrust on deploy.
        """
        setter = WeightSetterThread(
            backend_client=mock_backend_client,
            subtensor=mock_subtensor,
            metagraph=mock_metagraph,
            wallet=mock_wallet,
            netuid=1,
            interval_seconds=0.1,
            shadow_mode=True,
        )
        setter.start()
        time.sleep(0.15)
        setter.stop()

        mock_subtensor.set_weights.assert_called()
        weights = mock_subtensor.set_weights.call_args.kwargs["weights"]
        # Default fixture: emission_weight unset → 1.0 → 100% top, 0% burn.
        top_u16, burn_u16 = compute_pinned_weights(1.0, 0.0, tail_sum=0)
        assert weights[0] == burn_u16
        assert weights[1] == top_u16  # 5GrwvaEF... fixture top miner

    def test_shadow_mode_with_race_submits_fallback_vector(
        self, mock_backend_client, mock_subtensor, mock_wallet
    ):
        """shadow_mode=True with a completed race: submits the legacy
        top-miner vector (NOT the race-based one), even though a race
        result is available. New algorithm only runs in live mode.
        """
        finishers = [
            {
                "miner_hotkey": f"5Hotkey{i}",
                "agent_version_id": str(uuid4()),
                "race_score": 0.9 - i * 0.1,
            }
            for i in range(6)
        ]

        metagraph = MagicMock()
        metagraph.hotkeys = [
            "5BurnUid",
            "5GrwvaEF...",  # fallback top-miner fixture hotkey
        ] + [f["miner_hotkey"] for f in finishers]
        metagraph.uids = list(range(len(metagraph.hotkeys)))

        completed_id = uuid4()
        mock_backend_client.get_race_history.return_value = _race_complete_history(
            completed_id
        )
        mock_backend_client.get_race_detail.return_value = _race_detail(finishers)

        setter = WeightSetterThread(
            backend_client=mock_backend_client,
            subtensor=mock_subtensor,
            metagraph=metagraph,
            wallet=mock_wallet,
            netuid=1,
            interval_seconds=0.1,
            shadow_mode=True,
        )
        setter.start()
        time.sleep(0.15)
        setter.stop()

        mock_subtensor.set_weights.assert_called()
        weights = mock_subtensor.set_weights.call_args.kwargs["weights"]
        # Default fixture: emission_weight unset → 1.0 → fallback puts
        # full top share in uid 1, zero in burn.
        top_u16, burn_u16 = compute_pinned_weights(1.0, 0.0, tail_sum=0)
        # Fallback vector: only top-miner slot gets weight here; race
        # finishers (uids 2..7) all stay at 0 because the race path was
        # suppressed by shadow_mode.
        assert weights[0] == burn_u16
        assert weights[1] == top_u16
        assert all(w == 0 for w in weights[2:])
