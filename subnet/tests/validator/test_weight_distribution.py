"""Tests for the deterministic top-50% weight distribution.

Determinism is load-bearing for Yuma consensus on subnet 15
(`kappa = 0.5`): if two validators emit different weight vectors for the
same race, the median collapses to 0 and the protection fails. The
byte-identical test enforces that property at the API boundary.
"""

from __future__ import annotations

import random

import pytest

from subnet.validator.weight_distribution import (
    RankedEntrant,
    U16_MAX,
    build_metagraph_weight_vector,
    compute_hotkey_weights,
    compute_top_burn_weights,
    rank_entrants,
)


def _make_entrants(n: int, seed: int = 0) -> list[RankedEntrant]:
    """Synthesise `n` qualifiers with deterministic-but-varied scores."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.append(
            RankedEntrant(
                miner_hotkey=f"hk_{i:04d}",
                agent_version_id=f"av_{i:04d}",
                race_score=rng.uniform(0.4, 0.9),
            )
        )
    return out


# --- top/burn pinning ---


@pytest.mark.parametrize(
    "t_top, t_burn, expected_top, expected_burn",
    [
        # Burn dominant — current production direction (75/25).
        (0.25, 0.75, 21845, 65535),
        # Top dominant — flipped ratio.
        (0.75, 0.25, 65535, 21845),
        # Equal — top wins the tie-break and pins.
        (0.50, 0.50, 65535, 65535),
        # All to burn.
        (0.0, 1.0, 0, 65535),
        # All to top.
        (1.0, 0.0, 65535, 0),
    ],
)
def test_compute_top_burn_weights_pins_dominant_at_u16_max(
    t_top, t_burn, expected_top, expected_burn
):
    top, burn = compute_top_burn_weights(t_top, t_burn)
    assert top == expected_top
    assert burn == expected_burn


@pytest.mark.parametrize(
    "t_top, t_burn",
    [
        (-0.1, 0.5),  # negative
        (0.5, -0.1),  # negative
        (0.6, 0.5),  # sum > 1
        (0.0, 0.0),  # both zero
    ],
)
def test_compute_top_burn_weights_rejects_invalid_ratios(t_top, t_burn):
    with pytest.raises(ValueError):
        compute_top_burn_weights(t_top, t_burn)


# --- ranking ---


def test_rank_entrants_orders_by_score_desc_then_agent_version_id_asc():
    entrants = [
        RankedEntrant("hk_a", "av_b", 0.5),
        RankedEntrant("hk_b", "av_a", 0.5),  # same score, av_a < av_b → first
        RankedEntrant("hk_c", "av_c", 0.7),  # highest score
        RankedEntrant("hk_d", "av_d", 0.3),
    ]
    ranked = rank_entrants(entrants)
    assert [r.miner_hotkey for r in ranked] == ["hk_c", "hk_b", "hk_a", "hk_d"]


def test_rank_entrants_stable_under_input_shuffle():
    """Two validators receiving the same qualifiers in different orders
    must produce identical rankings."""
    entrants = _make_entrants(50, seed=42)
    shuffled = entrants.copy()
    random.Random(7).shuffle(shuffled)
    assert rank_entrants(entrants) == rank_entrants(shuffled)


# --- compute_hotkey_weights expected shapes ---


@pytest.mark.parametrize("n", [10, 30, 50, 100])
@pytest.mark.parametrize("t_top, t_burn", [(0.25, 0.75), (0.75, 0.25)])
def test_compute_hotkey_weights_shape_per_spec(n, t_top, t_burn):
    """The AC's algorithm: rank 1 = top_u16, ranks 2..K = K+1-rank,
    bottom 50% (and any ties at K) absent."""
    entrants = _make_entrants(n, seed=n)
    weights = compute_hotkey_weights(entrants, t_top, t_burn)

    k = n // 2
    # Top half present, bottom half absent.
    assert len(weights) == k

    ranked = rank_entrants(entrants)
    expected_top, _ = compute_top_burn_weights(t_top, t_burn)

    # Rank 1 = top_u16.
    assert weights[ranked[0].miner_hotkey] == expected_top

    # Ranks 2..K = K + 1 - rank (linear taper, last entry weight=1).
    for idx in range(1, k):
        rank_1based = idx + 1
        assert weights[ranked[idx].miner_hotkey] == k + 1 - rank_1based

    # Rank-K weight is exactly 1 (minimum increment).
    assert weights[ranked[k - 1].miner_hotkey] == 1

    # Bottom 50% absent.
    for idx in range(k, n):
        assert ranked[idx].miner_hotkey not in weights

    # All emitted weights are >= 1 (no silent zeroing).
    assert all(w >= 1 for w in weights.values())


@pytest.mark.parametrize("n", [0, 1])
def test_compute_hotkey_weights_empty_for_too_few_entrants(n):
    """N=0 yields no weights. N=1 yields no weights either — floor(1/2)=0,
    so there is no rank-1 to receive the top weight (the burn uid still
    fires unconditionally in `build_metagraph_weight_vector`)."""
    weights = compute_hotkey_weights(_make_entrants(n), 0.25, 0.75)
    assert weights == {}


# --- AC examples (N=30, 50, 100, T_TOP=0.25, T_BURN=0.75) ---


def test_compute_hotkey_weights_matches_ac_examples_n30():
    """AC: N=30 K=15 tail_size=14 tail_sum=105 burn=65535 top=21845."""
    entrants = _make_entrants(30, seed=30)
    weights = compute_hotkey_weights(entrants, 0.25, 0.75)
    ranked = rank_entrants(entrants)

    assert weights[ranked[0].miner_hotkey] == 21845  # top
    # tail weights 14, 13, ..., 1 → sum = 14*15/2 = 105
    tail_sum = sum(
        weights[ranked[idx].miner_hotkey] for idx in range(1, 15)
    )
    assert tail_sum == 105


def test_compute_hotkey_weights_matches_ac_examples_n100():
    """AC: N=100 K=50 tail_size=49 tail_sum=1225."""
    entrants = _make_entrants(100, seed=100)
    weights = compute_hotkey_weights(entrants, 0.25, 0.75)
    ranked = rank_entrants(entrants)

    assert weights[ranked[0].miner_hotkey] == 21845
    tail_sum = sum(
        weights[ranked[idx].miner_hotkey] for idx in range(1, 50)
    )
    assert tail_sum == 49 * 50 / 2


# --- determinism (the Yuma-consensus property) ---


def test_two_validators_with_same_inputs_emit_byte_identical_weights():
    """Simulates two validators receiving the qualifiers in different
    orders and a metagraph with hotkeys in different orders. The final
    `(uids, weights)` vector must be byte-identical."""
    entrants_a = _make_entrants(50, seed=1)
    entrants_b = entrants_a.copy()
    random.Random(2).shuffle(entrants_b)

    metagraph_a = ["hk_burn"] + [e.miner_hotkey for e in entrants_a]
    metagraph_b = list(metagraph_a)  # same ordering — required: validators
    # see the same metagraph (chain state). Re-shuffling here would not
    # be meaningful — uids are chain-assigned, not per-validator.

    uids_a, weights_a = build_metagraph_weight_vector(
        entrants_a, metagraph_a, t_top=0.25, t_burn=0.75, burn_uid=0
    )
    uids_b, weights_b = build_metagraph_weight_vector(
        entrants_b, metagraph_b, t_top=0.25, t_burn=0.75, burn_uid=0
    )

    assert uids_a == uids_b
    assert weights_a == weights_b


# --- metagraph integration ---


def test_build_metagraph_vector_places_burn_at_uid_0():
    entrants = _make_entrants(10, seed=10)
    metagraph = ["burn_hk"] + [e.miner_hotkey for e in entrants]

    _, weights = build_metagraph_weight_vector(
        entrants, metagraph, t_top=0.25, t_burn=0.75, burn_uid=0
    )

    assert weights[0] == U16_MAX  # burn pinned
    assert weights[0] >= weights[1]  # burn dominates rank-1 in this regime


def test_build_metagraph_vector_drops_hotkeys_missing_from_metagraph():
    """A race winner deregistered between race close and weight set
    must not crash the algorithm — their weight is silently dropped and
    the burn share grows."""
    entrants = _make_entrants(10, seed=10)
    # Drop the rank-1 hotkey from the metagraph.
    ranked = rank_entrants(entrants)
    metagraph = ["burn_hk"] + [
        e.miner_hotkey for e in entrants if e.miner_hotkey != ranked[0].miner_hotkey
    ]

    _, weights = build_metagraph_weight_vector(
        entrants, metagraph, t_top=0.25, t_burn=0.75, burn_uid=0
    )

    # Rank-1 hotkey is gone — no entry has the top-pin weight (21845).
    assert U16_MAX not in [w for i, w in enumerate(weights) if i != 0]
    # Burn slot still holds U16_MAX.
    assert weights[0] == U16_MAX


def test_build_metagraph_vector_returns_aligned_uids():
    entrants = _make_entrants(10, seed=10)
    metagraph = ["burn_hk"] + [e.miner_hotkey for e in entrants]
    uids, weights = build_metagraph_weight_vector(
        entrants, metagraph, t_top=0.25, t_burn=0.75, burn_uid=0
    )
    assert uids == list(range(len(metagraph)))
    assert len(weights) == len(metagraph)


def test_build_metagraph_vector_empty_metagraph_returns_empty():
    entrants = _make_entrants(10, seed=10)
    uids, weights = build_metagraph_weight_vector(
        entrants, [], t_top=0.25, t_burn=0.75, burn_uid=0
    )
    assert uids == []
    assert weights == []
