"""ORO-1704 epoch-pinned standings consumer tests.

The load-bearing test is `test_pinned_vector_equals_live_regardless_of_order`:
the pinned base vector must be byte-identical to the live one (that is the whole
point — pin *what* validators adopt, not the resulting vector). Plus shadow/live
mode selection and the safe fallbacks.
"""

import time
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from oro_sdk.models.epoch_standings import EpochStandings
from oro_sdk.models.pinned_finisher import PinnedFinisher

from validator.backend_client import WeightSalt
from validator.weight_setter import WeightSetterThread, _standings_to_inputs

from .test_weight_setter import _race_complete_history, _race_detail


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


def _setter(metagraph, mock_backend_client, mock_subtensor, mock_wallet, mode="off"):
    return WeightSetterThread(
        backend_client=mock_backend_client,
        subtensor=mock_subtensor,
        metagraph=metagraph,
        wallet=mock_wallet,
        netuid=1,
        interval_seconds=0.1,
        epoch_pin_mode=mode,
    )


def _metagraph(finishers):
    mg = MagicMock()
    mg.hotkeys = ["5BurnUid"] + [f["miner_hotkey"] for f in finishers] + ["5Extra"]
    mg.uids = list(range(len(mg.hotkeys)))
    return mg


# --- mapping ---------------------------------------------------------------

def test_standings_to_inputs_maps_fields():
    fins = _finishers(3)
    st = _standings(fins, top_hotkey="5HK0", t_burn=0.6)
    finishers, top, t_burn = _standings_to_inputs(st)
    assert top == "5HK0"
    assert t_burn == pytest.approx(0.6)
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

    # live finishers, in score order
    from validator.weight_setter import _qualifiers_to_finishers

    live_finishers = _qualifiers_to_finishers(_race_detail(fins).qualifiers)
    live_uids, live_w = setter._build_weights_from_race(live_finishers, "5HK0", 0.75)

    # pinned standings with the SAME finishers but reversed order
    st = _standings(list(reversed(fins)), top_hotkey="5HK0", t_burn=0.75)
    p_fin, p_top, p_burn = _standings_to_inputs(st)
    pin_uids, pin_w = setter._build_weights_from_race(p_fin, p_top, p_burn)

    assert (pin_uids, pin_w) == (live_uids, live_w)


# --- mode selection at the tick level -------------------------------------

def _run_tick_once(setter):
    setter.start()
    time.sleep(0.15)
    setter.stop()


def _tick_setup(mock_backend_client, fins, live_top, pin_top, mode):
    """Live path yields `live_top`; pinned standings name `pin_top` — so the two
    base vectors differ in the top slot, making the submitted choice observable."""
    mock_backend_client.get_top_miner.return_value.top_miner_hotkey = live_top
    mock_backend_client.get_top_miner.return_value.policy = None  # -> fallback t_burn
    mock_backend_client.get_race_history.return_value = _race_complete_history(uuid4())
    mock_backend_client.get_race_detail.return_value = _race_detail(fins)
    mock_backend_client.fetch_weight_salt.return_value = WeightSalt(
        overlay={}, epoch_standings=_standings(fins, top_hotkey=pin_top)
    )


def test_shadow_mode_submits_live(
    mock_backend_client, mock_subtensor, mock_wallet
):
    fins = _finishers(6)
    mg = _metagraph(fins)
    _tick_setup(mock_backend_client, fins, live_top="5HK0", pin_top="5HK1", mode="shadow")
    setter = _setter(mg, mock_backend_client, mock_subtensor, mock_wallet, mode="shadow")
    _run_tick_once(setter)

    weights = mock_subtensor.set_weights.call_args.kwargs["weights"]
    # Live top = 5HK0 (uid 1) gets the 25% top slot; pinned top 5HK1 (uid 2) does not.
    assert weights[1] > weights[2]  # submitted the LIVE vector


def test_live_mode_submits_pinned(
    mock_backend_client, mock_subtensor, mock_wallet
):
    fins = _finishers(6)
    mg = _metagraph(fins)
    _tick_setup(mock_backend_client, fins, live_top="5HK0", pin_top="5HK1", mode="live")
    setter = _setter(mg, mock_backend_client, mock_subtensor, mock_wallet, mode="live")
    _run_tick_once(setter)

    weights = mock_subtensor.set_weights.call_args.kwargs["weights"]
    # Pinned top = 5HK1 (uid 2) gets the top slot; live top 5HK0 (uid 1) does not.
    assert weights[2] > weights[1]  # submitted the PINNED vector


def test_no_standings_falls_back_to_live(
    mock_backend_client, mock_subtensor, mock_wallet
):
    fins = _finishers(6)
    mg = _metagraph(fins)
    mock_backend_client.get_top_miner.return_value.top_miner_hotkey = "5HK0"
    mock_backend_client.get_top_miner.return_value.policy = None  # -> fallback t_burn
    mock_backend_client.get_race_history.return_value = _race_complete_history(uuid4())
    mock_backend_client.get_race_detail.return_value = _race_detail(fins)
    mock_backend_client.fetch_weight_salt.return_value = WeightSalt(
        overlay={}, epoch_standings=None
    )
    setter = _setter(mg, mock_backend_client, mock_subtensor, mock_wallet, mode="live")
    _run_tick_once(setter)

    weights = mock_subtensor.set_weights.call_args.kwargs["weights"]
    assert weights[1] > weights[2]  # live top 5HK0 (uid 1), no pin available


def test_invalid_mode_rejected(mock_backend_client, mock_subtensor, mock_wallet):
    with pytest.raises(ValueError):
        _setter(MagicMock(), mock_backend_client, mock_subtensor, mock_wallet, mode="bogus")
