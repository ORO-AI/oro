"""Deterministic weight distribution for the top half of race finishers.

A finisher is a qualifier from the most recent completed race that
actually finished the race (has a non-null `race_score`). Qualifiers
that DNF'd or were eliminated mid-race land in the public race detail
with `race_score=null` and are dropped at the boundary in
`weight_setter._qualifiers_to_finishers`. By the time finishers reach
this module the list is already filtered.

The protection target is the half of last race's finishers with the
highest scores: those who actively competed and did not finish at the
bottom of the pack. They keep `Emission[uid] > 0` between races and
survive `get_neuron_to_prune` (which ranks by emission asc, reg_block
asc, uid asc) when their `immunity_period` expires.

The function in this module is pure — same `(finishers, t_top, t_burn)`
yields byte-identical u16 weight vectors across validators. That
property is load-bearing for Yuma consensus on subnet 15 (`kappa = 0.5`):
if validators emit different weight vectors for the tail, the median
collapses to 0 and the protection fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# u16 cap on each weight entry submitted to the chain.
U16_MAX = 65535


# Allocation model
# ----------------
# Top miner receives exactly `t_top` of normalised emission. Burn uid +
# tail (top-half ranks 2..K) together receive exactly `t_burn`.
# `t_top + t_burn == 1` is required — there is no third bucket.
#
# Concretely with `t_top=0.25`, `t_burn=0.75`:
#   * burn slot pinned at U16_MAX = 65535
#   * total = (U16_MAX + tail_sum) / t_burn
#   * top u16 = round(t_top * total)
# so the top miner's share is invariant under `tail_sum` (i.e. under N),
# and the tail "comes out of" the burn allocation rather than from both
# top and burn proportionally.


@dataclass(frozen=True)
class RankedFinisher:
    """A single race qualifier reduced to the fields needed for ranking.

    Validators only need the score (for ordering), the agent_version_id
    (for tie-breaks), and the hotkey (for mapping to metagraph uid).
    """

    miner_hotkey: str
    agent_version_id: str
    race_score: float


def rank_finishers(qualifiers: Iterable[RankedFinisher]) -> list[RankedFinisher]:
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


def _validate_ratios(t_top: float, t_burn: float) -> None:
    if t_top < 0 or t_burn < 0:
        raise ValueError("t_top and t_burn must be non-negative")
    if abs((t_top + t_burn) - 1.0) > 1e-9:
        raise ValueError(
            "t_top + t_burn must equal 1; the tail comes out of t_burn's share"
        )
    if t_top == 0 and t_burn == 0:
        raise ValueError("at least one of t_top / t_burn must be > 0")


def _tail_sum_for(k: int) -> int:
    """Sum of tail u16 weights for ranks 2..K (linear taper K-1, K-2, ..., 1).

    Closed form: (K - 1) * K // 2. Returns 0 for K < 2.
    """
    if k < 2:
        return 0
    return (k - 1) * k // 2


def compute_pinned_weights(
    t_top: float, t_burn: float, tail_sum: int
) -> tuple[int, int]:
    """Return `(top_u16, burn_u16)` such that the chain-normalised vector
    yields exactly `t_top` for the top miner and `t_burn` for the burn uid
    plus tail combined.

    The tail allocation comes out of `t_burn`'s share, not from both
    proportionally — top miner stays at exactly `t_top` regardless of N.

    The larger of `t_top` / `t_burn` is pinned at `U16_MAX`; the smaller
    is derived from the ratio + the tail sum so the integrated total
    sums to the right normalised shares.
    """
    _validate_ratios(t_top, t_burn)
    if tail_sum < 0:
        raise ValueError("tail_sum must be non-negative")

    if t_burn >= t_top:
        # Pin burn slot at U16_MAX. Burn share equals (burn + tail) / total,
        # so total = (U16_MAX + tail_sum) / t_burn and top = t_top * total.
        burn = U16_MAX
        if t_burn == 0:
            return 0, 0
        total = (U16_MAX + tail_sum) / t_burn
        top = round(t_top * total) if t_top > 0 else 0
        return top, burn

    # Pin top at U16_MAX. Total = U16_MAX / t_top; burn slot is the
    # share of t_burn that's left after the tail consumes its part.
    top = U16_MAX
    if t_top == 0:
        return 0, 0
    total = U16_MAX / t_top
    burn = round(t_burn * total) - tail_sum
    if burn < 0:
        # Tail consumed more than the burn share — the configured t_burn
        # is too small for the race size at this t_top. The caller would
        # need to either drop t_top, raise t_burn, or shrink K. Fail loud
        # rather than silently emitting a negative burn.
        raise ValueError(
            f"tail_sum {tail_sum} exceeds burn share at "
            f"t_top={t_top}, t_burn={t_burn}; lower N or adjust ratios"
        )
    return top, burn


def compute_hotkey_weights(
    qualifiers: Iterable[RankedFinisher],
    t_top: float,
    t_burn: float,
    top_hotkey: str | None = None,
) -> dict[str, int]:
    """Compute hotkey → u16 weight for the top emission slot + last-race
    deregistration-protection tail.

    The top slot (`top_u16`) goes to `top_hotkey` if provided (the canonical
    "current top for emissions" from `/v1/public/top`). When `top_hotkey`
    is None — no admin-designated top exists (fresh subnet, suite switch,
    or the designated top was discarded) — the top share is **not**
    assigned to anyone here; the caller (`build_metagraph_weight_vector`)
    routes it to the burn slot. We deliberately do not synthesize a top
    from rank-1 of finishers: the 25% slot belongs to the admin-designated
    top only.

    The tail is the top 50% of last-race finishers minus the top hotkey if
    they overlap. Tail entries receive a linear taper M, M-1, ..., 1 in rank
    order, where M is the number of tail entries. The tail's share comes
    out of `t_burn` — the top miner's share does not move with N.

    Bottom 50% (and ties at the rank-K boundary, by tiebreak) get no entry.
    """
    ranked = rank_finishers(qualifiers)
    k = len(ranked) // 2  # floor — protected-set size

    # Tail = top-K finishers excluding `top_hotkey` if they overlap.
    # When `top_hotkey` is None the filter never matches, so the tail is
    # the full top-K. When `top_hotkey` is rank-1 of finishers, the tail
    # is ranks 2..K (the historical shape).
    tail_finishers = [f for f in ranked[:k] if f.miner_hotkey != top_hotkey]
    m = len(tail_finishers)
    tail_sum = m * (m + 1) // 2  # M + (M-1) + ... + 1

    top_u16, _ = compute_pinned_weights(t_top, t_burn, tail_sum)
    weights: dict[str, int] = {}
    if top_hotkey is not None:
        weights[top_hotkey] = top_u16
    for idx, finisher in enumerate(tail_finishers):
        weights[finisher.miner_hotkey] = m - idx

    return weights


def build_metagraph_weight_vector(
    qualifiers: Iterable[RankedFinisher],
    metagraph_hotkeys: list[str],
    t_top: float,
    t_burn: float,
    top_hotkey: str | None = None,
) -> tuple[list[int], list[int]]:
    """Produce `(uids, weights_u16)` aligned to the metagraph.

    `top_hotkey` is the canonical "current top for emissions" (from
    `/v1/public/top`). When set and present in the metagraph, that hotkey
    receives the top emission slot. When `top_hotkey` is None — no admin-
    designated top (fresh subnet, suite switch, or the designated top was
    discarded) — OR the designated top has deregistered between Backend
    designation and the weight set, the top share rolls into the burn
    slot. We do not synthesize a top from rank-1 of finishers: the 25%
    slot belongs to the admin-designated top only.

    Steps:

    1. Rank qualifiers and compute hotkey → top-slot + tail u16 weights.
    2. Compute the burn u16 from the configured ratio.
    3. Map every hotkey-weight onto its metagraph index. A hotkey present
       in the race but absent from the metagraph (deregistered between
       race close and weight set) is silently dropped — its weight does
       not redistribute, so the burn share grows slightly.
    4. Add `burn_u16` at uid 0; if there is no eligible top, also fold
       `top_u16` into uid 0 so the burn slot absorbs the full 25%.

    Returns:
        Two parallel lists of length `len(metagraph_hotkeys)`. `uids[i]`
        is `i` (the metagraph index), `weights_u16[i]` is the u16 weight
        for that uid (0 if the hotkey is not in the top slot, not in the
        last-race tail, and not the burn uid).
    """
    n_meta = len(metagraph_hotkeys)
    if n_meta == 0:
        return [], []

    hotkey_to_idx = {hk: i for i, hk in enumerate(metagraph_hotkeys)}

    # The 25% top slot is allocated to the admin-designated top — and
    # *only* if that hotkey is in the current metagraph. Otherwise (no
    # admin designation, or designated top deregistered after Backend
    # picked them) the slot burns: `weights[0]` gets only `burn_u16`,
    # and the unallocated top share normalises into burn on-chain.
    has_eligible_top = top_hotkey is not None and top_hotkey in hotkey_to_idx
    effective_top = top_hotkey if has_eligible_top else None

    hotkey_weights = compute_hotkey_weights(
        list(qualifiers), t_top, t_burn, top_hotkey=effective_top
    )

    # Place the tail. The top entry (if any) is pinned below from the
    # recomputed `top_u16`; here we only iterate the dereg-protection
    # tail and track the actual tail sum after any dereg drops.
    weights = [0] * n_meta
    tail_sum_actual = 0
    for hk, w in hotkey_weights.items():
        if hk == effective_top:
            continue
        idx = hotkey_to_idx.get(hk)
        if idx is None:
            continue  # finisher deregistered; share folds into burn on-chain
        weights[idx] = w
        tail_sum_actual += w

    # Pin top + burn from the *actual* tail sum so the top miner lands at
    # exactly `t_top` of the submitted vector. Additive assignment handles
    # the rare testnet case where uid 0 is itself the top hotkey.
    top_u16, burn_u16 = compute_pinned_weights(t_top, t_burn, tail_sum_actual)
    weights[0] += burn_u16
    if has_eligible_top:
        weights[hotkey_to_idx[top_hotkey]] += top_u16

    return list(range(n_meta)), weights
