"""Tests for the ORO-1458 guarded corroboration title rule.

Uses a stub sentence model so tests stay hermetic (no BAAI/bge download).
The guarded rule credits a title constraint whose cosine is in the
0.50–0.72 band when (a) the product passes every non-title constraint,
and (b) the reward has ≥1 distinctive non-title constraint key. Either
condition alone must NOT flip the credit — those are the two arms of the
2026-07-30 corroboration exploit surface.
"""

from __future__ import annotations

import pytest

from src.agent.rewards import orm


class _StubModel:
    """Returns a fixed sim per (product_title, gt_title) pair."""

    def __init__(self, sim_map: dict[tuple[str, str], float]) -> None:
        self._sims = sim_map
        self._embedding_dim = 8

    def get_sentence_embedding_dimension(self):
        return self._embedding_dim

    def encode(self, titles):
        return [{"title": t} for t in titles]

    def similarity(self, product_embs, gt_embs):
        # embeddings are our stub dicts; recover titles and look up sim
        pt = product_embs[0]["title"]
        gt = gt_embs[0]["title"]
        return [[self._sims.get((pt, gt), 0.0)]]


@pytest.fixture
def stub_model(monkeypatch):
    def _install(sim_map):
        model = _StubModel(sim_map)
        monkeypatch.setattr(orm, "_get_sentence_model", lambda: model)
        return model

    return _install


def _product(title: str, *, attributes=None, sku_options=None, price=0.0, service=None):
    return {
        "product_id": "cand-1",
        "title": title,
        "attributes": attributes or {},
        "sku_options": sku_options or {},
        "price": price,
        "service": service or [],
    }


def _reward(*, title=None, attributes=None, sku_options=None, price=None, service=None, product_id="gt-1"):
    r = {"product_id": product_id}
    if title is not None:
        r["title"] = title if isinstance(title, list) else [title]
    if attributes is not None:
        r["attributes"] = attributes
    if sku_options is not None:
        r["sku_options"] = sku_options
    if price is not None:
        r["price"] = price
    if service is not None:
        r["service"] = service
    return r


# ── guard-active: recovers 0.50–0.72 band FN ─────────────────────
def test_title_credited_when_sim_in_band_and_all_non_title_pass(stub_model):
    stub_model({("cand-title", "gt-title"): 0.60})
    product = _product("cand-title", attributes={"console_type": ["nintendo"]})
    reward = _reward(
        title="gt-title",
        attributes=[{"console_type": ["nintendo"]}],
    )
    score, total, hit = orm.rule_score_reward(product, reward)
    # 2 constraints total (1 title + 1 attr), both credited → 1.0
    assert score == pytest.approx(1.0)
    assert hit["title"] == 1
    assert total["title"] == 1


def test_title_credited_at_hard_threshold_without_guard(stub_model):
    stub_model({("cand", "gt"): 0.75})
    product = _product("cand", attributes={"color": ["red"]})
    reward = _reward(title="gt", attributes=[{"color": ["red"]}])
    score, total, hit = orm.rule_score_reward(product, reward)
    assert score == pytest.approx(1.0)
    assert hit["title"] == 1


# ── guard-inactive: exploit surface stays closed ─────────────────
def test_guard_inactive_when_no_distinctive_key(stub_model):
    # generic-only reward — the exploit pattern (`colour=pink` for pink
    # hair curler passes any pink product). Guard must NOT credit title.
    stub_model({("phone-case", "hair-curler"): 0.60})
    product = _product("phone-case", attributes={"color": ["pink"]})
    reward = _reward(title="hair-curler", attributes=[{"color": ["pink"]}])
    score, total, hit = orm.rule_score_reward(product, reward)
    # 2 constraints: color hit, title NOT hit → 0.5
    assert score == pytest.approx(0.5)
    assert hit["title"] == 0


def test_guard_inactive_when_non_title_constraint_fails(stub_model):
    stub_model({("cand", "gt"): 0.60})
    product = _product("cand", attributes={"console_type": ["playstation"]})
    reward = _reward(title="gt", attributes=[{"console_type": ["nintendo"]}])
    score, total, hit = orm.rule_score_reward(product, reward)
    # 2 constraints total, both miss → 0
    assert score == pytest.approx(0.0)
    assert hit["title"] == 0


def test_guard_inactive_when_reward_has_zero_non_title(stub_model):
    # Post-ORO-1776 the synth never emits 0-non-title rewards, but the
    # grader defends against a legacy problem still in the bank.
    stub_model({("cand", "gt"): 0.60})
    product = _product("cand")
    reward = _reward(title="gt")
    score, total, hit = orm.rule_score_reward(product, reward)
    # 1 constraint (title), missed → 0
    assert score == pytest.approx(0.0)
    assert hit["title"] == 0


def test_guard_inactive_when_only_price_service_non_title(stub_model):
    # price + service alone don't count as distinctive — they widen the
    # pool. Reviewer regression case from the ORO-1458 exploit scan.
    stub_model({("cand", "gt"): 0.60})
    product = _product("cand", price=100.0, service=["freeShipping"])
    reward = _reward(
        title="gt",
        price=[{"less than": [0, 200]}],
        service=["freeShipping"],
    )
    score, total, hit = orm.rule_score_reward(product, reward)
    # 3 constraints total: price + service credited; title not
    assert hit["title"] == 0
    assert score == pytest.approx(2 / 3)


def test_title_still_missed_when_sim_below_guarded_floor(stub_model):
    # 0.40 < 0.50 guarded threshold — must not credit even with clean corroboration.
    stub_model({("cand", "gt"): 0.40})
    product = _product("cand", attributes={"console_type": ["nintendo"]})
    reward = _reward(title="gt", attributes=[{"console_type": ["nintendo"]}])
    score, total, hit = orm.rule_score_reward(product, reward)
    assert hit["title"] == 0
    assert score == pytest.approx(0.5)


# ── flag-off restores pre-1458 strict behavior ────────────────────
def test_flag_off_ignores_band(stub_model, monkeypatch):
    monkeypatch.setenv("TITLE_CORROBORATION_GUARD", "0")
    stub_model({("cand", "gt"): 0.60})
    product = _product("cand", attributes={"console_type": ["nintendo"]})
    reward = _reward(title="gt", attributes=[{"console_type": ["nintendo"]}])
    score, total, hit = orm.rule_score_reward(product, reward)
    # 2 constraints: attr hit, title miss (0.60 < 0.72) → 0.5
    assert score == pytest.approx(0.5)
    assert hit["title"] == 0


def test_flag_default_on(monkeypatch):
    monkeypatch.delenv("TITLE_CORROBORATION_GUARD", raising=False)
    assert orm._title_guard_enabled() is True


# ── distinctive-key counter ───────────────────────────────────────
def test_distinctive_count_ignores_generic_and_ps():
    r = {
        "attributes": [{"color": ["red"]}, {"material": ["cotton"]}],
        "price": [{"less than": [0, 100]}],
        "service": ["freeShipping"],
    }
    assert orm._distinctive_reward_key_count(r) == 0


def test_distinctive_count_finds_specific_keys():
    r = {
        "attributes": [{"color": ["red"]}, {"console_type": ["nintendo"]}],
        "sku_options": [{"pack_size": ["2pcs"]}],
    }
    assert orm._distinctive_reward_key_count(r) == 2


def test_distinctive_count_normalizes_key_variants():
    # color_family / colour / material are all in the generic set —
    # regardless of separator spelling.
    r = {
        "attributes": [{"color family": ["red"]}, {"COLOUR": ["blue"]}],
    }
    assert orm._distinctive_reward_key_count(r) == 0


# ── review-regression: guard must fail when price/service fails ───
def test_guard_inactive_when_price_constraint_fails(stub_model):
    # Reviewer's blocking case: distinctive attr passes, price fails,
    # title sim in band. `max_hit == max_total` isn't enough — the guard
    # must see the price outcome too.
    stub_model({("cand", "gt"): 0.60})
    product = _product(
        "cand",
        attributes={"console_type": ["nintendo"]},
        price=100.0,
    )
    reward = _reward(
        title="gt",
        attributes=[{"console_type": ["nintendo"]}],
        price=[{"less than": [0, 50]}],
    )
    score, total, hit = orm.rule_score_reward(product, reward)
    # 3 constraints: attr hit, price miss, title miss (guard suppressed)
    assert hit["title"] == 0
    assert score == pytest.approx(1 / 3)


def test_guard_inactive_when_service_constraint_fails(stub_model):
    # Companion case — service is the other non-title constraint class
    # not covered by max_hit/max_total.
    stub_model({("cand", "gt"): 0.60})
    product = _product(
        "cand",
        attributes={"console_type": ["nintendo"]},
        service=["COD"],  # freeShipping missing
    )
    reward = _reward(
        title="gt",
        attributes=[{"console_type": ["nintendo"]}],
        service=["freeShipping"],
    )
    score, total, hit = orm.rule_score_reward(product, reward)
    # 3 constraints: attr hit, service miss, title miss
    assert hit["title"] == 0
    assert score == pytest.approx(1 / 3)


def test_guard_active_when_price_and_service_both_pass(stub_model):
    # Positive case for the extended check — every non-title constraint
    # (attr + price + service) passes, so the guard credits the band title.
    stub_model({("cand", "gt"): 0.60})
    product = _product(
        "cand",
        attributes={"console_type": ["nintendo"]},
        price=45.0,
        service=["freeShipping"],
    )
    reward = _reward(
        title="gt",
        attributes=[{"console_type": ["nintendo"]}],
        price=[{"less than": [0, 50]}],
        service=["freeShipping"],
    )
    score, total, hit = orm.rule_score_reward(product, reward)
    # 4 constraints all credited
    assert hit["title"] == 1
    assert score == pytest.approx(1.0)


# ── GT match short-circuits guard entirely ────────────────────────
def test_gt_match_scores_1_regardless_of_sim(stub_model):
    stub_model({("gt-title", "gt-title"): 0.0})  # sim unused
    product = {**_product("gt-title"), "product_id": "gt-1"}
    reward = _reward(title="gt-title", product_id="gt-1", attributes=[{"color": ["red"]}])
    score, total, hit = orm.rule_score_reward(product, reward)
    assert score == 1
    assert hit["title"] == 1
