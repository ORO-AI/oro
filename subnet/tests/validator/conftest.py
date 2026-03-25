"""Shared pytest fixtures for validator tests."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from bittensor_wallet import Keypair, Wallet

from oro_sdk.models import HeartbeatResponse, TopAgentResponse

from validator.backend_client import BackendClient


# =============================================================================
# Common Temp File Fixtures
# =============================================================================


@pytest.fixture
def temp_storage_path():
    """Create a temporary file path for retry queue storage."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    yield path
    # Explicit cleanup since delete=False
    if path.exists():
        path.unlink()


@pytest.fixture
def temp_output_file():
    """Create a temporary JSONL file for progress reporter tests."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = Path(f.name)
    yield path
    # Explicit cleanup since delete=False
    if path.exists():
        path.unlink()


# =============================================================================
# Basic Mock Fixtures (can be overridden in test files)
# =============================================================================


@pytest.fixture
def mock_backend_client():
    """Create a basic mock BackendClient.

    Test files can configure specific method return values as needed.
    """
    return MagicMock(spec=BackendClient)


@pytest.fixture
def mock_wallet():
    """Create a mock wallet with a real keypair for signing.

    Uses a deterministic test keypair for reproducible tests.
    """
    keypair = Keypair.create_from_uri("//TestValidator")
    wallet = MagicMock(spec=Wallet)
    wallet.hotkey = keypair
    return wallet


@pytest.fixture
def mock_wallet_simple():
    """Create a simple mock wallet without real keypair.

    Use this when signing is not needed.
    """
    return MagicMock()


@pytest.fixture
def mock_subtensor():
    """Create a mock subtensor."""
    subtensor = MagicMock()
    subtensor.blocks_since_last_update.return_value = 0
    subtensor.tempo.return_value = 100
    return subtensor


@pytest.fixture
def mock_metagraph():
    """Create a mock metagraph with basic hotkeys and uids.

    Hotkey "5GrwvaEF..." at index 1 is used as the top miner in weight setter tests.
    """
    metagraph = MagicMock()
    metagraph.hotkeys = ["5Hotkey1...", "5GrwvaEF...", "5Hotkey3..."]
    metagraph.uids = [0, 1, 2]
    metagraph.S = [1.0, 1.0, 1.0]
    metagraph.axons = [MagicMock(), MagicMock(), MagicMock()]
    return metagraph


# =============================================================================
# Pre-configured Mock Fixtures for Specific Use Cases
# =============================================================================


@pytest.fixture
def mock_backend_client_with_heartbeat(mock_backend_client):
    """Mock BackendClient configured for heartbeat tests."""
    mock_backend_client.heartbeat.return_value = HeartbeatResponse(
        lease_expires_at=datetime.now() + timedelta(minutes=5)
    )
    return mock_backend_client


@pytest.fixture
def mock_backend_client_with_top_miner(mock_backend_client):
    """Mock BackendClient configured for weight setter tests.

    Returns "5GrwvaEF..." as top miner, which is at index 1 in mock_metagraph.
    """
    mock_backend_client.get_top_miner.return_value = TopAgentResponse(
        suite_id=789,
        top_agent_version_id=UUID("87654321-4321-4321-4321-210987654321"),
        top_miner_hotkey="5GrwvaEF...",  # Matches mock_metagraph index 1
        top_score=0.92,
        computed_at=datetime.now(),
    )
    return mock_backend_client
