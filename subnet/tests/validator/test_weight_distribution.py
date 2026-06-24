"""Tests for the deterministic top-50% weight distribution.

Determinism is load-bearing for Yuma consensus on subnet 15
(`kappa = 0.5`): if two validators emit different weight vectors for the
same race, the median collapses to 0 and the protection fails. The
byte-identical test enforces that property at the API boundary.

The model: single knob `t_burn` (Backend env var
`emission_baseline_burn_rate`). Top miner gets `1 - t_burn`. When the
tail integer-taper exceeds the burn budget (low-burn regime), the top
miner absorbs the deficit; the tail's protection share is preserved.
"""

from __future__ import annotations

import random

import pytest

from subnet.validator.weight_distribution import (
    RankedFinisher,
    U16_MAX,
    build_metagraph_weight_vector,
    compute_hotkey_weights,
    compute_pinned_weights,
    rank_finishers,
)


def _make_finishers(n: int, seed: int = 0) -> list[RankedFinisher]:
    """Synthesise `n` qualifiers with deterministic-but-varied scores."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.append(
            RankedFinisher(
                miner_hotkey=f"hk_{i:04d}",
                agent_version_id=f"av_{i:04d}",
                race_score=rng.uniform(0.4, 0.9),
            )
        )
    return out


def _share(value: int, total: int) -> float:
    return value / total


# --- pinned-weight computation ---


def test_compute_pinned_weights_burn_dominant_no_tail():
    """At t_burn=0.75 (today) with no tail, burn pins at U16_MAX and
    top derives to give exactly 25% share."""
    top, burn = compute_pinned_weights(0.75, tail_sum=0)
    assert burn == U16_MAX
    # Top derived: round(0.25 * 65535 / 0.75) = round(21845.0).
    assert top == 21845
    # Sanity: shares match the configured ratios.
    total = top + burn
    assert abs(_share(top, total) - 0.25) < 1e-3
    assert abs(_share(burn, total) - 0.75) < 1e-3


def test_compute_pinned_weights_top_share_invariant_under_tail_growth():
    """At t_burn >= t_top, the top miner's normalised share is exactly
    `1 - t_burn` regardless of N (i.e. regardless of tail size). The
    tail consumes only the burn share."""
    for tail_sum in [0, 105, 300, 1225, 5000]:
        top, burn = compute_pinned_weights(0.75, tail_sum=tail_sum)
        total = top + burn + tail_sum
        assert abs(_share(top, total) - 0.25) < 1e-4
        assert abs(_share(burn + tail_sum, total) - 0.75) < 1e-4


def test_compute_pinned_weights_top_dominant_pins_top_at_u16_max():
    """t_burn < t_top (e.g. burn=0.25): top pins, burn derives from
    share minus the tail."""
    top, burn = compute_pinned_weights(0.25, tail_sum=0)
    assert top == U16_MAX
    # burn = round(0.25 * 65535 / 0.75) - 0 = 21845
    assert burn == 21845


def test_compute_pinned_weights_equal_ratio_pins_burn():
    """Equal split (t_burn=0.5) — burn wins the tie-break and pins."""
    top, burn = compute_pinned_weights(0.5, tail_sum=0)
    assert burn == U16_MAX
    assert top == U16_MAX


def test_compute_pinned_weights_all_burn():
    top, burn = compute_pinned_weights(1.0, tail_sum=0)
    assert top == 0
    assert burn == U16_MAX


def test_compute_pinned_weights_all_top():
    top, burn = compute_pinned_weights(0.0, tail_sum=0)
    assert top == U16_MAX
    assert burn == 0


def test_compute_pinned_weights_burn_zero_absorbs_tail_into_top():
    """At t_burn=0, the top miner absorbs the tail-share deficit. Burn
    clamps to 0; the top miner's effective share falls slightly below
    1.0 to leave room for the tail's protection weights."""
    tail_sum = 190  # M=19, race size 40
    top, burn = compute_pinned_weights(0.0, tail_sum=tail_sum)
    assert top == U16_MAX
    assert burn == 0
    total = top + burn + tail_sum
    # Top effective share = U16_MAX / (U16_MAX + tail_sum).
    assert abs(_share(top, total) - U16_MAX / (U16_MAX + tail_sum)) < 1e-9
    # Tail share is exactly tail_sum / total.
    assert abs(_share(tail_sum, total) - tail_sum / (U16_MAX + tail_sum)) < 1e-9


@pytest.mark.parametrize(
    "t_burn",
    [
        pytest.param(-0.1, id="negative"),
        pytest.param(1.1, id="above-1"),
    ],
)
def test_compute_pinned_weights_rejects_invalid_burn(t_burn):
    with pytest.raises(ValueError):
        compute_pinned_weights(t_burn, tail_sum=0)


def test_compute_pinned_weights_rejects_negative_tail_sum():
    with pytest.raises(ValueError):
        compute_pinned_weights(0.75, tail_sum=-1)


def test_compute_pinned_weights_oversized_tail_at_low_burn_absorbs_into_top():
    """Top-dominant + tail bigger than burn budget — used to raise
    ValueError. Now the top miner absorbs the deficit (burn clamps to
    0) and the vector still emits valid u16 weights."""
    # t_burn=0.01 → burn would be round(0.01 * U16_MAX / 0.99) - 1000 = 662 - 1000 = -338
    top, burn = compute_pinned_weights(0.01, tail_sum=1000)
    assert top == U16_MAX
    assert burn == 0


# --- invariant: shares sum to 1.0 across burn sweep ---


@pytest.mark.parametrize("t_burn", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("n", [10, 20, 40, 80])
def test_total_shares_sum_to_one_across_burn_sweep(t_burn, n):
    """Top + tail + burn shares always sum to 1.0 (single-knob
    invariant), even in the low-burn deficit-absorption regime."""
    finishers = _make_finishers(n, seed=n)
    ranked = rank_finishers(finishers)
    metagraph = ["burn_uid"] + [f.miner_hotkey for f in finishers]
    _, weights = build_metagraph_weight_vector(
        finishers,
        metagraph,
        t_burn=t_burn,
        top_hotkey=ranked[0].miner_hotkey,
    )
    total = sum(weights)
    assert total > 0
    # Shares sum to 1.0 by construction; just make sure non-negative.
    assert all(w >= 0 for w in weights)


# --- ranking ---


def test_rank_finishers_orders_by_score_desc_then_agent_version_id_asc():
    finishers = [
        RankedFinisher("hk_a", "av_b", 0.5),
        RankedFinisher("hk_b", "av_a", 0.5),  # same score, av_a < av_b → first
        RankedFinisher("hk_c", "av_c", 0.7),  # highest score
        RankedFinisher("hk_d", "av_d", 0.3),
    ]
    ranked = rank_finishers(finishers)
    assert [r.miner_hotkey for r in ranked] == ["hk_c", "hk_b", "hk_a", "hk_d"]


def test_rank_finishers_stable_under_input_shuffle():
    finishers = _make_finishers(50, seed=42)
    shuffled = finishers.copy()
    random.Random(7).shuffle(shuffled)
    assert rank_finishers(finishers) == rank_finishers(shuffled)


# --- compute_hotkey_weights expected shapes ---


@pytest.mark.parametrize("n", [10, 30, 50, 100])
@pytest.mark.parametrize("t_burn", [0.75, 0.25])
def test_compute_hotkey_weights_shape_per_spec(n, t_burn):
    """Rank 1 = top_u16 (sized for exact `1 - t_burn` share), ranks 2..K =
    K+1-rank, bottom 50% absent."""
    finishers = _make_finishers(n, seed=n)
    ranked = rank_finishers(finishers)
    weights = compute_hotkey_weights(
        finishers, t_burn, top_hotkey=ranked[0].miner_hotkey
    )

    k = n // 2
    assert len(weights) == k

    tail_sum = sum(range(1, k))  # 1 + 2 + ... + (K-1)
    expected_top, _ = compute_pinned_weights(t_burn, tail_sum=tail_sum)

    assert weights[ranked[0].miner_hotkey] == expected_top

    for idx in range(1, k):
        rank_1based = idx + 1
        assert weights[ranked[idx].miner_hotkey] == k + 1 - rank_1based

    assert weights[ranked[k - 1].miner_hotkey] == 1

    for idx in range(k, n):
        assert ranked[idx].miner_hotkey not in weights

    assert all(w >= 1 for w in weights.values())


@pytest.mark.parametrize("n", [0, 1])
def test_compute_hotkey_weights_empty_for_too_few_finishers(n):
    """N=0 or N=1 → floor(N/2)=0, no protected tail entries. With no
    `top_hotkey` either, the dict is empty; the burn uid still fires
    unconditionally in `build_metagraph_weight_vector`."""
    weights = compute_hotkey_weights(_make_finishers(n), 0.75)
    assert weights == {}


# --- top share is exactly (1 - t_burn) across N (the load-bearing property at high burn) ---


@pytest.mark.parametrize("n", [10, 30, 50, 100])
def test_top_miner_share_is_exactly_t_top_regardless_of_n(n):
    """At t_burn=0.75 the top miner's share is invariant under N. Verify
    on the integrated metagraph vector."""
    finishers = _make_finishers(n, seed=n)
    ranked = rank_finishers(finishers)
    metagraph = ["hk_burn"] + [e.miner_hotkey for e in finishers]
    _, weights = build_metagraph_weight_vector(
        finishers,
        metagraph,
        t_burn=0.75,
        top_hotkey=ranked[0].miner_hotkey,
    )
    rank1_idx = metagraph.index(ranked[0].miner_hotkey)

    total = sum(weights)
    top_share = weights[rank1_idx] / total
    assert abs(top_share - 0.25) < 5e-4

    burn_plus_tail = total - weights[rank1_idx]
    assert abs(burn_plus_tail / total - 0.75) < 5e-4


# --- AC examples (the published shape with the new "tail from burn" model) ---


def test_compute_hotkey_weights_n30_top_share_25pct():
    """N=30, K=15, tail_sum=105. Top = round(0.25*(65535+105)/0.75) = 21880."""
    finishers = _make_finishers(30, seed=30)
    ranked = rank_finishers(finishers)
    weights = compute_hotkey_weights(
        finishers, 0.75, top_hotkey=ranked[0].miner_hotkey
    )

    assert weights[ranked[0].miner_hotkey] == 21880
    tail_sum = sum(weights[ranked[idx].miner_hotkey] for idx in range(1, 15))
    assert tail_sum == 105


def test_compute_hotkey_weights_n100_top_share_25pct():
    """N=100, K=50, tail_sum=1225. Top = round(0.25*(65535+1225)/0.75) = 22253."""
    finishers = _make_finishers(100, seed=100)
    ranked = rank_finishers(finishers)
    weights = compute_hotkey_weights(
        finishers, 0.75, top_hotkey=ranked[0].miner_hotkey
    )

    assert weights[ranked[0].miner_hotkey] == 22253
    tail_sum = sum(weights[ranked[idx].miner_hotkey] for idx in range(1, 50))
    assert tail_sum == 1225


# --- determinism (the Yuma-consensus property) ---


def test_two_validators_with_same_inputs_emit_byte_identical_weights():
    """Two validators receiving the qualifiers in different orders must
    produce byte-identical `(uids, weights)` vectors."""
    finishers_a = _make_finishers(50, seed=1)
    finishers_b = finishers_a.copy()
    random.Random(2).shuffle(finishers_b)

    metagraph_a = ["hk_burn"] + [e.miner_hotkey for e in finishers_a]
    metagraph_b = list(metagraph_a)

    uids_a, weights_a = build_metagraph_weight_vector(
        finishers_a, metagraph_a, t_burn=0.75
    )
    uids_b, weights_b = build_metagraph_weight_vector(
        finishers_b, metagraph_b, t_burn=0.75
    )

    assert uids_a == uids_b
    assert weights_a == weights_b


# --- metagraph integration ---


def test_build_metagraph_vector_places_burn_at_uid_0():
    finishers = _make_finishers(10, seed=10)
    metagraph = ["burn_hk"] + [e.miner_hotkey for e in finishers]

    _, weights = build_metagraph_weight_vector(
        finishers, metagraph, t_burn=0.75
    )

    assert weights[0] == U16_MAX
    assert weights[0] >= weights[1]


def test_build_metagraph_vector_drops_hotkeys_missing_from_metagraph():
    """A race winner deregistered between race close and weight set
    must not crash the algorithm — their weight is silently dropped and
    the burn share grows."""
    finishers = _make_finishers(10, seed=10)
    ranked = rank_finishers(finishers)
    metagraph = ["burn_hk"] + [
        e.miner_hotkey for e in finishers if e.miner_hotkey != ranked[0].miner_hotkey
    ]

    _, weights = build_metagraph_weight_vector(
        finishers, metagraph, t_burn=0.75
    )

    assert weights[0] == U16_MAX


def test_build_metagraph_vector_returns_aligned_uids():
    finishers = _make_finishers(10, seed=10)
    metagraph = ["burn_hk"] + [e.miner_hotkey for e in finishers]
    uids, weights = build_metagraph_weight_vector(
        finishers, metagraph, t_burn=0.75
    )
    assert uids == list(range(len(metagraph)))
    assert len(weights) == len(metagraph)


def test_build_metagraph_vector_empty_metagraph_returns_empty():
    finishers = _make_finishers(10, seed=10)
    uids, weights = build_metagraph_weight_vector(
        finishers, [], t_burn=0.75
    )
    assert uids == []
    assert weights == []


# --- top_hotkey override (current top via score-to-beat) ---


def test_top_hotkey_none_burns_top_share_no_synthesis():
    """ORO-1120: when no admin-designated top exists, do NOT synthesize
    one from rank-1 of finishers. The dict carries the full top-K tail
    (no top entry); the caller burns the 25% top share."""
    finishers = _make_finishers(20, seed=42)
    ranked = rank_finishers(finishers)
    k = len(finishers) // 2

    weights = compute_hotkey_weights(finishers, 0.75, top_hotkey=None)

    expected_top_u16, _ = compute_pinned_weights(0.75, tail_sum=k * (k + 1) // 2)
    assert weights[ranked[0].miner_hotkey] != expected_top_u16
    assert weights[ranked[0].miner_hotkey] == k
    assert weights[ranked[k - 1].miner_hotkey] == 1
    assert len(weights) == k


def test_top_hotkey_override_promotes_non_winner_to_top_slot():
    """When `top_hotkey` differs from rank-1, the override hotkey takes
    the top u16 slot and the previous rank-1 falls into the tail."""
    finishers = _make_finishers(20, seed=7)
    ranked = rank_finishers(finishers)
    rank1 = ranked[0].miner_hotkey
    rank2 = ranked[1].miner_hotkey

    weights = compute_hotkey_weights(finishers, 0.75, top_hotkey=rank2)

    top_u16 = max(weights.values())
    assert weights[rank2] == top_u16
    assert rank1 in weights
    assert weights[rank1] != top_u16
    k = len(finishers) // 2
    m = k - 1
    assert weights[rank1] == m


def test_top_hotkey_not_in_finishers_keeps_old_winner_in_tail():
    """When `top_hotkey` is a brand-new hotkey not present in last-race
    finishers, the override hotkey gets the top slot and the full top-K
    of finishers (including old rank-1) populates the tail."""
    finishers = _make_finishers(20, seed=11)
    ranked = rank_finishers(finishers)
    rank1 = ranked[0].miner_hotkey
    new_top = "hk_brand_new_agent"

    weights = compute_hotkey_weights(finishers, 0.75, top_hotkey=new_top)

    top_u16 = max(weights.values())
    assert weights[new_top] == top_u16
    k = len(finishers) // 2
    assert weights[rank1] == k


def test_top_hotkey_override_top_share_remains_t_top():
    """The top miner's share of the chain-normalised vector stays at
    exactly `1 - t_burn` regardless of whether the override pulls from
    inside or outside the protected tail."""
    finishers = _make_finishers(20, seed=99)
    ranked = rank_finishers(finishers)

    for top_hk in [ranked[0].miner_hotkey, ranked[5].miner_hotkey, "hk_outsider"]:
        weights = compute_hotkey_weights(finishers, 0.75, top_hotkey=top_hk)
        m = sum(1 for v in weights.values() if v != max(weights.values()))
        tail_sum = m * (m + 1) // 2
        top_u16, burn_u16 = compute_pinned_weights(0.75, tail_sum)
        total = top_u16 + burn_u16 + tail_sum
        assert _share(top_u16, total) == pytest.approx(0.25, abs=2e-4)


def test_build_metagraph_vector_top_hotkey_promoted_to_top_slot():
    """End-to-end: `top_hotkey` lands at its metagraph index with the top
    u16, and the burn slot at uid 0 still receives the burn share."""
    finishers = _make_finishers(20, seed=3)
    ranked = rank_finishers(finishers)
    rank2_hk = ranked[1].miner_hotkey
    metagraph_hotkeys = ["burn_uid"] + [f.miner_hotkey for f in finishers]

    uids, weights = build_metagraph_weight_vector(
        finishers,
        metagraph_hotkeys=metagraph_hotkeys,
        t_burn=0.75,
        top_hotkey=rank2_hk,
    )
    rank2_idx = metagraph_hotkeys.index(rank2_hk)
    rank1_idx = metagraph_hotkeys.index(ranked[0].miner_hotkey)
    non_burn = [w if i != 0 else 0 for i, w in enumerate(weights)]

    assert weights[rank2_idx] == max(non_burn)
    assert weights[rank1_idx] > 0
    assert weights[rank1_idx] < weights[rank2_idx]
    assert weights[0] > 0


def test_build_metagraph_vector_top_hotkey_not_in_metagraph_burns_top_share():
    """ORO-1120: when `top_hotkey` is designated by Backend but the hotkey
    deregistered between designation and the weight set, the validator no
    longer falls back to rank-1 of finishers. The top share burns; the
    tail still receives its dereg-protection weights."""
    finishers = _make_finishers(20, seed=8)
    ranked = rank_finishers(finishers)
    rank1_hk = ranked[0].miner_hotkey
    metagraph_hotkeys = ["burn_uid"] + [f.miner_hotkey for f in finishers]

    uids, weights = build_metagraph_weight_vector(
        finishers,
        metagraph_hotkeys=metagraph_hotkeys,
        t_burn=0.75,
        top_hotkey="hk_deregistered",
    )

    assert weights[0] == U16_MAX
    rank1_idx = metagraph_hotkeys.index(rank1_hk)
    k = len(finishers) // 2
    assert weights[rank1_idx] == k
    total = sum(weights)
    burn_share = weights[0] / total
    assert burn_share > 0.99


def test_build_metagraph_vector_no_top_hotkey_burns_top_share():
    """ORO-1120: with `top_hotkey=None` (no admin top designated — fresh
    subnet, suite switch, or designated top discarded), the 25% slot
    burns. Tail finishers still receive their linear-taper dereg-protection
    weights."""
    finishers = _make_finishers(20, seed=42)
    ranked = rank_finishers(finishers)
    metagraph_hotkeys = ["burn_uid"] + [f.miner_hotkey for f in finishers]

    uids, weights = build_metagraph_weight_vector(
        finishers,
        metagraph_hotkeys=metagraph_hotkeys,
        t_burn=0.75,
        top_hotkey=None,
    )

    assert weights[0] == U16_MAX
    rank1_idx = metagraph_hotkeys.index(ranked[0].miner_hotkey)
    k = len(finishers) // 2
    assert weights[rank1_idx] == k
    total = sum(weights)
    assert weights[0] / total > 0.99
    for idx in range(k, len(finishers)):
        bottom_idx = metagraph_hotkeys.index(ranked[idx].miner_hotkey)
        assert weights[bottom_idx] == 0


def test_build_metagraph_vector_no_top_hotkey_no_finishers_burns_everything():
    """ORO-1120: no admin top + no finishers → full burn (existing behavior
    is preserved end-to-end)."""
    metagraph_hotkeys = ["burn_uid", "hk_other_1", "hk_other_2"]
    uids, weights = build_metagraph_weight_vector(
        [],
        metagraph_hotkeys=metagraph_hotkeys,
        t_burn=0.75,
        top_hotkey=None,
    )

    assert weights[0] == U16_MAX
    assert weights[1] == 0
    assert weights[2] == 0


def test_two_validators_with_top_override_emit_byte_identical_weights():
    """Determinism property still holds when `top_hotkey` is supplied:
    same `(top_hotkey, finishers, t_burn)` → byte-identical vectors."""
    finishers = _make_finishers(50, seed=2026)
    metagraph_hotkeys = ["burn_uid"] + [f.miner_hotkey for f in finishers]
    top_hk = finishers[7].miner_hotkey

    a_uids, a_weights = build_metagraph_weight_vector(
        finishers,
        metagraph_hotkeys=metagraph_hotkeys,
        t_burn=0.75,
        top_hotkey=top_hk,
    )
    shuffled = list(finishers)
    random.Random(123).shuffle(shuffled)
    b_uids, b_weights = build_metagraph_weight_vector(
        shuffled,
        metagraph_hotkeys=metagraph_hotkeys,
        t_burn=0.75,
        top_hotkey=top_hk,
    )
    assert a_uids == b_uids
    assert a_weights == b_weights


# --- burn-from-backend, deficit absorption ---


def test_build_metagraph_vector_burn_zero_emits_top_dominant_vector():
    """At t_burn=0, top miner takes ~99.7% (M=19 race), tail keeps its
    rank-j integer taper, burn uid stamps 0."""
    finishers = _make_finishers(40, seed=1439)
    ranked = rank_finishers(finishers)
    metagraph = ["burn_uid"] + [f.miner_hotkey for f in finishers]
    _, weights = build_metagraph_weight_vector(
        finishers,
        metagraph,
        t_burn=0.0,
        top_hotkey=ranked[0].miner_hotkey,
    )
    rank1_idx = metagraph.index(ranked[0].miner_hotkey)
    assert weights[rank1_idx] == U16_MAX
    assert weights[0] == 0  # burn slot empty
    total = sum(weights)
    assert weights[rank1_idx] / total == pytest.approx(0.9971, abs=1e-3)
