"""Analytics primitives that operate on captured agent trajectories.

This package is intentionally side-effect-free: nothing in here touches
the network, the sandbox, or the Backend. Pure functions over JSON
trajectory bundles so miners can reproduce the metrics locally.
"""

from src.analytics.agentic_richness import (
    AgenticRichnessResult,
    calc_agentic_richness,
    calc_agentic_richness_for_step,
)

__all__ = [
    "AgenticRichnessResult",
    "calc_agentic_richness",
    "calc_agentic_richness_for_step",
]
