"""ORO-1704 epoch-pinned standings consumer tests.

The load-bearing test is `test_pinned_vector_equals_live_regardless_of_order`:
the pinned base vector must be byte-identical to the live one (that is the whole
point — pin *what* validators adopt, not the resulting vector). The rest cover
the switch-to-endpoint-or-fall-back behaviour: use the pinned standings when the
Backend serves them, otherwise fall back to this validator's live standings.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

from oro_sdk.models.epoch_standings import EpochStandings
from oro_sdk.models.pinned_finisher import PinnedFinisher
from validator.backend_client import WeightSalt
from validator.weight_setter import (
    WeightSetterThread,
    _qualifiers_to_finishers,
    _standings_to_inputs,
)

from .test_weight_setter import _race_detail


def _finishers(n=5):
    return [
        {"miner_hotkey": f"5HK{i}", "agent_version_id": str(uuid4()), "race_score": 0.9 - i * 0.05}
        for i in range(n)
    ]


def _standings(finishers, top_hotkey, t_burn=0.75):
    return EpochStandings(
        top_hotkey=top_hotkey,
        t_burn=t_burn,
        finishers=[
            PinnedFinisher(
                miner_hotkey=f["miner_hotkey"],
                agent_version_id=f["agent_version_id"],
                race_score=f["race_score"],
            )
            for f in finishers
        ],
    )


def _setter(metagraph, mock_backend_client, mock_subtensor, mock_wallet):
    return WeightSetterThread(
        backend_client=mock_backend_client,
        subtensor=mock_subtensor,
        metagraph=metagraph,
        wallet=mock_wallet,
        netuid=1,
        interval_seconds=0.1,
    )


def _metagraph(finishers):
    mg = MagicMock()
    mg.hotkeys = ["5BurnUid"] + [f["miner_hotkey"] for f in finishers] + ["5Extra"]
    mg.uids = list(range(len(mg.hotkeys)))
    # Real int block so `_current_epoch()` computes: block 100 / (tempo 100 *
    # reveal 1) = epoch 1.
    mg.block = 100
    return mg


# --- mapping ---------------------------------------------------------------

def test_standings_to_inputs_maps_fields():
    fins = _finishers(3)
    st = _standings(fins, top_hotkey="5HK0", t_burn=0.6)
    finishers, top, t_burn = _standings_to_inputs(st)
    assert top == "5HK0"
    assert t_burn == 0.6
    assert [(f.miner_hotkey, f.agent_version_id, f.race_score) for f in finishers] == [
        (f["miner_hotkey"], f["agent_version_id"], f["race_score"]) for f in fins
    ]


def test_standings_to_inputs_none_top_burns():
    st = _standings(_finishers(2), top_hotkey=None)
    _f, top, _t = _standings_to_inputs(st)
    assert top is None


# --- the equivalence guarantee --------------------------------------------

def test_pinned_vector_equals_live_regardless_of_order(
    mock_backend_client, mock_subtensor, mock_wallet
):
    """Pinned base vector == live base vector even when the pinned finishers are
    served in a different order — the validator re-sorts, so serve order is
    irrelevant and the two vectors must match exactly."""
    fins = _finishers(6)
    mg = _metagraph(fins)
    setter = _setter(mg, mock_backend_client, mock_subtensor, mock_wallet)

    live_finishers = _qualifiers_to_finishers(_race_detail(fins).qualifiers)
    live_uids, live_w = setter._build_weights_from_race(live_finishers, "5HK0", 0.75)

    # pinned standings with the SAME finishers but reversed order
    st = _standings(list(reversed(fins)), top_hotkey="5HK0", t_burn=0.75)
    p_fin, p_top, p_burn = _standings_to_inputs(st)
    pin_uids, pin_w = setter._build_weights_from_race(p_fin, p_top, p_burn)

    assert (pin_uids, pin_w) == (live_uids, live_w)


# --- tick: switch to pinned, else fall back to live ------------------------

def _run_tick_once(setter):
    setter.start()
    time.sleep(0.15)
    setter.stop()


def test_uses_pinned_standings_when_served(
    mock_backend_client, mock_subtensor, mock_wallet
):
    """Standings present ⇒ submit the pinned vector; top slot lands on the pinned
    hotkey (ORO-1802: the authed epoch-pinned standings are the ONLY source)."""
    fins = _finishers(6)
    mg = _metagraph(fins)
    mock_backend_client.fetch_weight_salt.return_value = WeightSalt(
        overlay={}, epoch_standings=_standings(fins, top_hotkey="5HK1")
    )
    setter = _setter(mg, mock_backend_client, mock_subtensor, mock_wallet)
    _run_tick_once(setter)

    weights = mock_subtensor.set_weights.call_args.kwargs["weights"]
    assert weights[2] > weights[1]  # pinned top 5HK1 (uid 2)


def test_no_standings_retains_last_good(
    mock_backend_client, mock_subtensor, mock_wallet
):
    """No standings (first miss) ⇒ retain last-good: do NOT submit, do NOT read
    any public fallback. There is no live /top path anymore (ORO-1802)."""
    mg = _metagraph(_finishers(6))
    mock_backend_client.fetch_weight_salt.return_value = WeightSalt(
        overlay={}, epoch_standings=None
    )
    setter = _setter(mg, mock_backend_client, mock_subtensor, mock_wallet)
    _run_tick_once(setter)

    mock_subtensor.set_weights.assert_not_called()


def test_empty_standings_retains(mock_backend_client, mock_subtensor, mock_wallet):
    """An empty pinned snapshot (snapshot lag / epoch transition) is a transient
    miss ⇒ retain last-good, no submit."""
    mg = _metagraph(_finishers(6))
    mock_backend_client.fetch_weight_salt.return_value = WeightSalt(
        overlay={},
        epoch_standings=EpochStandings(top_hotkey="5HK1", t_burn=0.75, finishers=[]),
    )
    setter = _setter(mg, mock_backend_client, mock_subtensor, mock_wallet)
    _run_tick_once(setter)

    mock_subtensor.set_weights.assert_not_called()


def test_malformed_standings_retains(mock_backend_client, mock_subtensor, mock_wallet):
    """A payload that fails validation (out-of-range t_burn) is an unusable miss
    ⇒ retain last-good, no submit."""
    fins = _finishers(6)
    mg = _metagraph(fins)
    mock_backend_client.fetch_weight_salt.return_value = WeightSalt(
        overlay={}, epoch_standings=_standings(fins, top_hotkey="5HK1", t_burn=1.5)
    )
    setter = _setter(mg, mock_backend_client, mock_subtensor, mock_wallet)
    _run_tick_once(setter)

    mock_subtensor.set_weights.assert_not_called()


def test_rejected_set_does_not_advance_burn_anchor(
    mock_backend_client, mock_subtensor, mock_wallet
):
    """A rejected set (ExtrinsicResponse-like `success=False`) must NOT count as
    a successful set — the burn anchor stays put. Regression for the bug where
    `bool(result)` was always True (ExtrinsicResponse.__len__ == 2)."""
    fins = _finishers(6)
    mg = _metagraph(fins)
    mock_backend_client.fetch_weight_salt.return_value = WeightSalt(
        overlay={}, epoch_standings=_standings(fins, top_hotkey="5HK0")
    )
    # An object that is truthy under bool() but reports success=False — exactly
    # the ExtrinsicResponse trap.
    mock_subtensor.set_weights.return_value = SimpleNamespace(success=False)
    setter = _setter(mg, mock_backend_client, mock_subtensor, mock_wallet)

    _run_tick_once(setter)

    mock_subtensor.set_weights.assert_called()  # it attempts the set
    assert setter._last_success_epoch is None  # but does not record success


def test_sustained_miss_burns_to_uid0(
    mock_backend_client, mock_subtensor, mock_wallet
):
    """A full epoch elapsed with no successful set ⇒ burn to UID 0. Prior success
    was in epoch 0; block 200 / (tempo 100 * reveal 1) = epoch 2, so a whole
    epoch (1) fully elapsed with no set (guard: epoch >= last_success + 2)."""
    mg = _metagraph(_finishers(6))
    mg.block = 200  # epoch 2 with the conftest tempo of 100
    mock_backend_client.fetch_weight_salt.return_value = WeightSalt(
        overlay={}, epoch_standings=None
    )
    setter = _setter(mg, mock_backend_client, mock_subtensor, mock_wallet)
    setter._last_success_epoch = 0  # last good set was a full epoch ago

    _run_tick_once(setter)

    weights = mock_subtensor.set_weights.call_args.kwargs["weights"]
    # Burn vector: all weight on the burn slot (uid 0), nothing elsewhere.
    assert weights[0] == max(weights)
    assert sum(weights[1:]) == 0


def test_partial_epoch_miss_retains_not_burn(
    mock_backend_client, mock_subtensor, mock_wallet
):
    """Off-by-one guard: a miss only one epoch after the last set (a set landed
    late in epoch 0, now epoch 1) is a partial-epoch transient miss ⇒ retain,
    NOT burn. Burning here would zero emissions minutes after a good set."""
    mg = _metagraph(_finishers(6))
    mg.block = 100  # epoch 1 with the conftest tempo of 100
    mock_backend_client.fetch_weight_salt.return_value = WeightSalt(
        overlay={}, epoch_standings=None
    )
    setter = _setter(mg, mock_backend_client, mock_subtensor, mock_wallet)
    setter._last_success_epoch = 0  # set landed in epoch 0

    _run_tick_once(setter)

    mock_subtensor.set_weights.assert_not_called()
