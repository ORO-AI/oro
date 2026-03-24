import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pytest

# Add test-subnet to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from oro_sdk.models import TopAgentResponse

from validator.weight_setter import WeightSetterThread


class TestWeightSetterThread:
    """Tests for WeightSetterThread.

    Uses mock_backend_client_with_top_miner, mock_metagraph, mock_subtensor,
    and mock_wallet_simple fixtures from conftest.py.
    """

    @pytest.fixture
    def mock_backend_client(self, mock_backend_client_with_top_miner):
        """Alias to use the pre-configured top miner fixture."""
        return mock_backend_client_with_top_miner

    @pytest.fixture
    def mock_wallet(self, mock_wallet_simple):
        """Alias to use simple wallet (no signing needed)."""
        return mock_wallet_simple

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

    def test_sets_weights_for_top_miner_only(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
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
        call_args = mock_subtensor.set_weights.call_args
        weights = call_args.kwargs.get("weights") or call_args[1].get("weights")
        # Index 1 is the top miner (5GrwvaEF...) - only they get weight
        assert weights[0] == 0.0
        assert weights[1] == 1.0
        assert weights[2] == 0.0

    def test_skips_weights_when_top_miner_not_in_metagraph(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
        mock_backend_client.get_top_miner.return_value = TopAgentResponse(
            suite_id=789,
            top_agent_version_id=UUID("87654321-4321-4321-4321-210987654321"),
            top_miner_hotkey="5UnknownHotkey...",
            top_score=0.92,
            computed_at=datetime.now(),
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

        # Should NOT call set_weights when top miner isn't in metagraph
        mock_subtensor.set_weights.assert_not_called()

    def test_emission_decay_splits_weight_to_burn_uid(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
        mock_backend_client.get_top_miner.return_value = TopAgentResponse(
            suite_id=789,
            top_agent_version_id=UUID("87654321-4321-4321-4321-210987654321"),
            top_miner_hotkey="5GrwvaEF...",
            top_score=0.92,
            computed_at=datetime.now(),
            emission_weight=0.97,
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

        call_args = mock_subtensor.set_weights.call_args
        weights = call_args.kwargs.get("weights") or call_args[1].get("weights")
        assert abs(weights[0] - 0.03) < 1e-9  # burn UID gets remainder
        assert abs(weights[1] - 0.97) < 1e-9  # top miner gets emission_wt
        assert weights[2] == 0.0

    def test_emission_weight_none_treated_as_full(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
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

        call_args = mock_subtensor.set_weights.call_args
        weights = call_args.kwargs.get("weights") or call_args[1].get("weights")
        assert weights[0] == 0.0  # no burn
        assert weights[1] == 1.0  # full weight

    def test_emission_decay_top_miner_is_uid_zero(
        self, mock_backend_client, mock_subtensor, mock_metagraph, mock_wallet
    ):
        """When top miner IS UID 0, they get emission_wt + burn_wt = 1.0."""
        mock_backend_client.get_top_miner.return_value = TopAgentResponse(
            suite_id=789,
            top_agent_version_id=UUID("87654321-4321-4321-4321-210987654321"),
            top_miner_hotkey="5BurnAddr...",
            top_score=0.92,
            computed_at=datetime.now(),
            emission_weight=0.90,
        )

        # Make UID 0 the top miner's hotkey
        mock_metagraph.hotkeys = ["5BurnAddr...", "5GrwvaEF...", "5FHneW46..."]

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

        call_args = mock_subtensor.set_weights.call_args
        weights = call_args.kwargs.get("weights") or call_args[1].get("weights")
        assert abs(weights[0] - 1.0) < 1e-9  # 0.90 + 0.10 = 1.0
        assert weights[1] == 0.0
        assert weights[2] == 0.0

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
