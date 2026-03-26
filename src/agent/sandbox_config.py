"""
Configuration management for sandbox execution.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""

    # Problem file configuration
    problem_file: str = "data/synthesize_product_test.jsonl"

    # Execution parameters
    max_workers: Optional[int] = None  # None = use CPU count
    timeout_per_problem: float = 300.0  # seconds

    # Proxy configuration
    sandbox_proxy_url: str = "http://proxy:80"

    # Output configuration
    output_file: Optional[str] = None  # None = print to stdout

    # Agent file configuration
    agent_file: Optional[str] = None  # None = use default src.agent.agent

    @classmethod
    def from_env(cls) -> "SandboxConfig":
        """Load configuration from environment variables."""
        return cls(
            problem_file=os.getenv(
                "SANDBOX_PROBLEM_FILE", "data/synthesize_product_test.jsonl"
            ),
            max_workers=int(os.getenv("SANDBOX_MAX_WORKERS", "0")) or None,
            timeout_per_problem=float(os.getenv("SANDBOX_TIMEOUT", "300.0")),
            sandbox_proxy_url=os.getenv("SANDBOX_PROXY_URL", "http://proxy:80"),
            output_file=os.getenv("SANDBOX_OUTPUT_FILE"),
            agent_file=os.getenv("SANDBOX_AGENT_FILE"),
        )

    def validate(self) -> None:
        """Validate configuration values."""
        if not os.path.exists(self.problem_file):
            raise FileNotFoundError(f"Problem file not found: {self.problem_file}")

        if self.timeout_per_problem <= 0:
            raise ValueError(
                f"timeout_per_problem must be positive, got {self.timeout_per_problem}"
            )

        if self.agent_file is not None and not os.path.exists(self.agent_file):
            raise FileNotFoundError(f"Agent file not found: {self.agent_file}")
