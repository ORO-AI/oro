"""Validator CLI argument parsing + bittensor logging configuration."""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

from bittensor.core.config import Config
from bittensor.core.subtensor import Subtensor
from bittensor.utils.btlogging import logging
from bittensor_wallet import Wallet


METRICS_PORT = 9100


def build_config(argv: Optional[Sequence[str]] = None) -> Config:
    """Parse validator CLI args + bt args into a Config, create the log dir."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem-file",
        default="data/synthesize_test.jsonl",
        help="Path to the problem JSONL file for agent evaluation.",
    )
    parser.add_argument(
        "--workspace-dir",
        default=os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ),
        help="Path to the ShoppingBench workspace root directory.",
    )
    parser.add_argument(
        "--sandbox-timeout",
        type=int,
        default=int(os.environ.get("SANDBOX_TIMEOUT") or "1800"),
        help="Timeout in seconds for the entire sandbox subprocess (env: SANDBOX_TIMEOUT, default: 1800 = 30 min).",
    )
    parser.add_argument(
        "--sandbox-max-workers",
        type=int,
        default=int(os.environ.get("SANDBOX_MAX_WORKERS") or "15"),
        help="Number of parallel problem workers in sandbox (env: SANDBOX_MAX_WORKERS).",
    )
    parser.add_argument(
        "--sandbox-problem-timeout",
        type=float,
        default=float(os.environ.get("SANDBOX_PROBLEM_TIMEOUT") or "300"),
        help="Timeout in seconds per problem in sandbox (env: SANDBOX_PROBLEM_TIMEOUT, default: 300 = 5 min).",
    )
    parser.add_argument(
        "--reasoning-max-workers",
        type=int,
        default=int(os.environ.get("REASONING_MAX_WORKERS") or "4"),
        help="Number of parallel reasoning judge workers (env: REASONING_MAX_WORKERS).",
    )
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("ORO_BACKEND_URL", "https://api.oroagents.com"),
        help="Backend API base URL (env: ORO_BACKEND_URL)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=int(os.environ.get("ORO_POLL_INTERVAL", "30")),
        help="Seconds between work claim attempts when no work (env: ORO_POLL_INTERVAL)",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=int(os.environ.get("ORO_HEARTBEAT_INTERVAL", "30")),
        help="Seconds between heartbeats during execution (env: ORO_HEARTBEAT_INTERVAL)",
    )
    parser.add_argument(
        "--weight-update-interval",
        type=int,
        default=int(os.environ.get("ORO_WEIGHT_UPDATE_INTERVAL", "300")),
        help="Seconds between weight updates from leaderboard (env: ORO_WEIGHT_UPDATE_INTERVAL)",
    )
    parser.add_argument("--netuid", type=int, default=15, help="The chain subnet uid.")
    Subtensor.add_args(parser)
    logging.add_args(parser)
    Wallet.add_args(parser)

    config = Config(parser)
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/validator".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
        )
    )
    os.makedirs(config.full_path, exist_ok=True)
    return config


def configure_logging(config: Config) -> None:
    """Apply bt logging config — default to INFO if neither debug nor trace is set."""
    if not config.logging.debug and not config.logging.trace:
        config.logging.info = True
    logging(config=config, logging_dir=config.full_path)
    logging.info(
        f"Running validator for subnet: {config.netuid} on network: {config.subtensor.network} with config:"
    )
    logging.info(config)
