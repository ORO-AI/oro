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


# ── leading compat-prefix strip, key-gated (ORO-1695) ────────────
def test_compat_prefix_stripped_on_compat_key():
    # Case 3 from race-104 miner audit: reward "for vivo y02a" vs product "vivo y02a"
    # under a compat-family key. Symmetric — either side can carry the prefix.
    assert _hit("compatibility_by_model", "for vivo y02a", [("compatibility_by_model", "vivo y02a")])
    assert _hit("compatibility_by_model", "vivo y02a", [("compatibility_by_model", "for vivo y02a")])
    assert _hit("compatibility", "fits samsung a04", [("compatibility", "samsung a04")])


def test_compat_prefix_NOT_stripped_on_non_compat_key():
    # `feature: "for indoor use"` must NOT be truncated to "indoor use" —
    # would false-match `feature: "indoor use only"`.
    assert not _hit("feature", "for indoor use", [("feature", "indoor use only")])
    # Distinct values under an unrelated key stay distinct even when the
    # reward starts with a compat phrase.
    assert not _hit("usage", "for outdoor", [("usage", "outdoor recreation")])


def test_normalize_attr_value_does_not_strip_prefix():
    # `normalize_attr_value` stays global-safe — prefix stripping is now
    # gated inside `_attr_constraint_hit`, not applied here.
    assert normalize_attr_value("for vivo y02a") == "for vivo y02a"
    assert normalize_attr_value("gift for mom") == "gift for mom"


# ── size letter-code expansion, key-gated to exact `size` (ORO-1695) ──
def test_size_letter_codes_expand_under_size_key():
    # Q3 shorts case: reward "int: medium" vs product "int:m"
    assert _hit("size", "int: medium", [("size", "int:m")])
    assert _hit("size", "int:large", [("size", "int:l")])
    assert _hit("size", "int: xlarge", [("size", "int:xl")])
    assert _hit("size", "int: 2xlarge", [("size", "int:xxl")])
    # Symmetric — product has long form, reward has short
    assert _hit("size", "int:m", [("size", "int: medium")])


def test_size_expansion_NOT_on_ring_screen_package_size():
    # `ring_size`, `screen_size`, `package_size` all contain "size" but use
    # letter codes for other purposes. Expansion is gated to the exact
    # `size` slot only.
    assert not _hit("ring_size", "m", [("ring_size", "medium")])
    assert not _hit("screen_size", "l", [("screen_size", "large")])
    assert not _hit("package_size", "s", [("package_size", "small")])


def test_size_expansion_NOT_on_unrelated_keys():
    # Letter codes in non-size slots must NOT expand
    assert not _hit("flavor", "flavor m", [("flavor", "flavor medium")])


def test_size_distinct_letters_do_not_collide():
    # `s` and `m` are distinct sizes; expanded to `small` vs `medium`, still distinct
    assert not _hit("size", "m", [("size", "s")])
    assert not _hit("size", "l", [("size", "xl")])


# ── curtain-type alias (ORO-1695) ────────────────────────────────
def test_curtain_type_alias():
    # Q1 curtains case: reward `type_curtain` vs product `curtain_type` on the same value
    assert _hit("type_curtain", "window curtain", [("curtain_type", "window curtain")])
    assert _hit("curtain_type", "curtains", [("type_curtain", "curtains")])


# ── word-order guarantee preserved (regression test for review finding #1) ──
def test_order_preserved_for_multi_token_values():
    # If we admitted an unordered subset match, permuted dimensional values
    # would false-hit (reward "10 20" vs product "20 10"). Same-key check
    # is strictly ordered.
    assert not _hit("size", "10 20", [("size", "20 10")])
    assert not _hit("dimensions", "3 5", [("dimensions", "5 3")])
