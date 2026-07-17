"""Tests for the normalized attribute matcher in rewards/orm.py.

Covers value/key normalization + same-key token-subset matching, curated
cross-key aliases, and the invariants that keep it exact-after-normalize
(distinct enums never collide, non-allowlisted keys never cross-match).
"""

from src.agent.rewards.orm import (
    _attr_constraint_hit,
    _build_attr_index,
    _is_token_subsequence,
    normalize_attr_key,
    normalize_attr_value,
)


def _hit(reward_key, reward_value, product_pairs):
    val_keys, key_tokens = _build_attr_index(product_pairs)
    return _attr_constraint_hit(reward_key, reward_value, val_keys, key_tokens)


# ── normalization ────────────────────────────────────────────────
def test_value_normalization_brackets_and_case():
    assert normalize_attr_value("7*9cm【10pcs】") == "7*9cm 10pcs"
    assert normalize_attr_value("（1pcs）2#- 48mm") == "1pcs 2#- 48mm"
    assert normalize_attr_value("MEDIUM") == "medium"


def test_key_normalization_space_underscore_plural():
    assert normalize_attr_key("color family") == normalize_attr_key("color_family")
    assert normalize_attr_key("Color_Family") == normalize_attr_key("color family")
    assert normalize_attr_key("flavors") == normalize_attr_key("flavor")
    assert normalize_attr_key("power_consumption") == normalize_attr_key("power consumption")
    # distinct keys stay distinct
    assert normalize_attr_key("color") != normalize_attr_key("color_family")


def test_token_subsequence_boundaries():
    assert _is_token_subsequence(["48mm"], ["1pcs", "2", "48mm"])
    assert _is_token_subsequence(["suede"], ["genuine", "suede", "leather"])
    # whole-token: 8gb is not a subsequence of the single token 128gb
    assert not _is_token_subsequence(["8gb"], ["128gb"])
    assert not _is_token_subsequence(["48mm"], ["50mm"])
    assert not _is_token_subsequence([], ["anything"])


# ── positive matches (should recover) ────────────────────────────
def test_key_string_normalization_match():
    assert _hit("color family", "blue", [("color_family", "blue")])
    assert _hit("flavors", "vanilla", [("flavor", "vanilla")])


def test_value_bracket_normalization_match():
    assert _hit("color_family", "7*9cm 10pcs", [("color_family", "7*9cm【10pcs】")])


def test_token_subset_match_same_key():
    assert _hit("size", "48mm", [("size", "1pcs 2#- 48mm")])
    assert _hit("material", "suede", [("material", "genuine suede leather")])


def test_exact_match_still_passes():
    assert _hit("color", "red", [("color", "red")])


# ── negative matches (must NOT collide) ──────────────────────────
def test_distinct_enum_values_do_not_match():
    assert not _hit("color", "blue", [("color", "black")])
    assert not _hit("model", "2023", [("model", "2024")])
    assert not _hit("color", "v1 red lining", [("color", "v1 black lining")])


def test_numeric_unit_collision_avoided():
    # 8gb must not match 128gb (token boundary), nor 43 match 42
    assert not _hit("ram", "8gb", [("ram", "128gb")])
    assert not _hit("size", "43", [("size", "42")])


def test_product_underspecifies_does_not_match():
    # reward wants "yellow-24inch"; product only says "yellow" -> reject
    assert not _hit("color", "yellow-24inch", [("color", "yellow")])


# ── curated cross-key aliases ────────────────────────────────────
def test_cross_key_alias_positive():
    # same value under a curated semantically-equivalent sibling key matches
    assert _hit("color_family", "blue", [("color", "blue")])
    assert _hit("color", "blue", [("color_family", "blue")])  # symmetric
    # British spelling: product carries the colour under `colour`, reward `color_family`
    assert _hit("color_family", "black", [("colour", "black")])
    # miner-reported cases
    assert _hit("colored_gem_type", "no stones", [("main_stone", "no stones")])
    assert _hit("model", "samsung galaxy a04e", [("compatibility_by_model", "samsung galaxy a04e")])
    assert _hit("plate_size", "30x40cm", [("size", "30x40cm")])
    assert _hit("concern_oral_care", "tooth decay", [("oral care benefits", "tooth decay")])


def test_cross_key_alias_requires_exact_value():
    # cross-key is exact-value only: a distinct value does not match even under
    # an allowlisted sibling key
    assert not _hit("color_family", "blue", [("color", "black")])
    assert not _hit("model", "iphone 15", [("compatibility_by_model", "iphone 14")])


def test_cross_key_non_allowlisted_pairs_never_match():
    # unrelated keys sharing a value must NOT cross-match (the over-match the
    # curated allowlist exists to prevent)
    assert not _hit("voltage", "4 modes 12v", [("color", "4 modes 12v")])
    assert not _hit("color_family", "washi tape", [("size", "washi tape")])
    assert not _hit("size", "large", [("style", "large")])
