"""Deterministic weight distribution for the top half of race entrants.

Replaces the prior "all weight to the single top miner + burn" model with a
linear-taper across the top `floor(N/2)` ranked entrants, so legitimate
competitors retain `Emission[uid] > 0` and survive `get_neuron_to_prune`
(which ranks by emission asc, reg_block asc, uid asc).

The function in this module is pure — same `(qualifiers, t_top, t_burn)`
yields byte-identical u16 weight vectors across validators. That property
is load-bearing for Yuma consensus on subnet 15 (`kappa = 0.5`): if
validators emit different weight vectors for the tail, the median collapses
to 0 and the protection fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# u16 cap on each weight entry submitted to the chain. The chain normalises
# the submitted vector, so we pin the larger of `t_top` / `t_burn` at this
# value to minimise the tail's share of total emission while preserving the
# configured ratio.
U16_MAX = 65535


@dataclass(frozen=True)
class RankedEntrant:
    """A single race qualifier reduced to the fields needed for ranking.

    Validators only need the score (for ordering), the agent_version_id
    (for tie-breaks), and the hotkey (for mapping to metagraph uid).
    """

    miner_hotkey: str
    agent_version_id: str
    race_score: float


def rank_entrants(qualifiers: Iterable[RankedEntrant]) -> list[RankedEntrant]:
    """Sort qualifiers into a canonical order shared by every validator.

    Primary key: `race_score` descending. Tie-break: `agent_version_id`
    ascending (UUIDs are deterministic strings). The combination is total —
    two qualifiers with identical score AND identical agent_version_id
    cannot exist (agent_version_id is unique per submission).
    """
    return sorted(
        qualifiers,
        key=lambda e: (-e.race_score, e.agent_version_id),
    )


def compute_top_burn_weights(t_top: float, t_burn: float) -> tuple[int, int]:
    """Return `(top_u16, burn_u16)` with the larger pinned at `U16_MAX`.

    Pinning the larger side at `U16_MAX` minimises the tail's share of the
    normalised emission vector while preserving the configured ratio.
    """
    if t_top < 0 or t_burn < 0:
        raise ValueError("t_top and t_burn must be non-negative")
    if t_top + t_burn > 1:
        raise ValueError("t_top + t_burn must be <= 1")
    if t_top == 0 and t_burn == 0:
        raise ValueError("at least one of t_top / t_burn must be > 0")

    if t_top >= t_burn:
        top = U16_MAX
        burn = round(U16_MAX * t_burn / t_top) if t_top > 0 else 0
    else:
        burn = U16_MAX
        top = round(U16_MAX * t_top / t_burn) if t_burn > 0 else 0
    return top, burn


def compute_hotkey_weights(
    qualifiers: Iterable[RankedEntrant],
    t_top: float,
    t_burn: float,
) -> dict[str, int]:
    """Compute hotkey → u16 weight for the top 50% of race entrants.

    Bottom 50% (and ties at the rank-K boundary, by tiebreak) get no entry.

    The "top" entry receives `top_u16` (or the smaller of the two pinned
    values if `t_burn > t_top`). Ranks 2..K receive a linear taper
    `K + 1 - rank`, so rank 2 = K-1, rank 3 = K-2, ..., rank K = 1.
    """
    ranked = rank_entrants(qualifiers)
    n = len(ranked)
    k = n // 2  # floor
    if k == 0:
        return {}

    top_u16, _ = compute_top_burn_weights(t_top, t_burn)
    weights: dict[str, int] = {}
    weights[ranked[0].miner_hotkey] = top_u16

    # Ranks 2..K (1-indexed) → indices 1..K-1 in `ranked`.
    for idx in range(1, k):
        rank_1based = idx + 1  # 2..K
        weights[ranked[idx].miner_hotkey] = k + 1 - rank_1based

    return weights


def build_metagraph_weight_vector(
    qualifiers: Iterable[RankedEntrant],
    metagraph_hotkeys: list[str],
    t_top: float,
    t_burn: float,
    burn_uid: int = 0,
) -> tuple[list[int], list[int]]:
    """Produce `(uids, weights_u16)` aligned to the metagraph.

    Steps:

    1. Rank qualifiers and compute hotkey → top-half u16 weights.
    2. Compute the burn u16 from the configured ratio.
    3. Map every hotkey-weight onto its metagraph index. A hotkey present
       in the race but absent from the metagraph (deregistered between
       race close and weight set) is silently dropped — its weight does
       not redistribute, so the burn share grows slightly. This matches
       the existing "top miner missing → burn everything" fallback.
    4. Add `burn_u16` at `burn_uid` (uid 0 on subnet 15 is a literal burn).

    Returns:
        Two parallel lists of length `len(metagraph_hotkeys)`. `uids[i]`
        is `i` (the metagraph index), `weights_u16[i]` is the u16 weight
        for that uid (0 if the hotkey is not in the top 50% and not the
        burn uid).
    """
    n_meta = len(metagraph_hotkeys)
    if n_meta == 0:
        return [], []

    hotkey_weights = compute_hotkey_weights(qualifiers, t_top, t_burn)
    top_u16, burn_u16 = compute_top_burn_weights(t_top, t_burn)
    # `compute_top_burn_weights` returns (top, burn) regardless of which
    # side is dominant; only `burn_u16` is needed here for the burn slot.

    weights = [0] * n_meta
    hotkey_to_idx = {hk: i for i, hk in enumerate(metagraph_hotkeys)}
    for hk, w in hotkey_weights.items():
        idx = hotkey_to_idx.get(hk)
        if idx is None:
            continue
        weights[idx] = w

    if 0 <= burn_uid < n_meta:
        # Burn uid may coincidentally collide with a top-half hotkey on
        # very small testnet metagraphs — the burn weight is additive so
        # the slot is never accidentally zeroed by the top-half pass.
        weights[burn_uid] += burn_u16

    return list(range(n_meta)), weights
