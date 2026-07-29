"""Tests for the normalized attribute matcher in rewards/orm.py.

Covers value/key normalization + same-key token-subset matching, curated
cross-key aliases, and the invariants that keep it exact-after-normalize
(distinct enums never collide, non-allowlisted keys never cross-match).
"""

from src.agent.rewards.orm import (
    _attr_constraint_hit,
    _build_attr_index,
    _is_token_subsequence,
    _is_token_subset,
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


# ── leading compatibility prefix strip (ORO-1695) ────────────────
def test_leading_for_prefix_stripped():
    # Case 3 from race-104 miner audit: reward "for vivo y02a" vs product "vivo y02a"
    assert _hit("compatibility_by_model", "for vivo y02a", [("compatibility_by_model", "vivo y02a")])
    assert _hit("compatibility", "fits samsung a04", [("compatibility", "samsung a04")])
    assert _hit("model", "compatible with iphone 15", [("model", "iphone 15")])


def test_prefix_strip_symmetric_when_product_has_prefix_and_reward_does_not():
    # normalization runs on both sides, so either direction hits
    assert _hit("compatibility_by_model", "vivo y02a", [("compatibility_by_model", "for vivo y02a")])


def test_prefix_strip_only_leading():
    # "for" appearing mid-value must NOT be stripped
    assert normalize_attr_value("gift for mom") == "gift for mom"
    # a bare non-compat word must NOT be stripped
    assert normalize_attr_value("form fit") == "form fit"


# ── size letter-code expansion (ORO-1695) ────────────────────────
def test_size_letter_codes_expand_under_size_key():
    # Q3 shorts case: reward "int: medium" vs product "int:m"
    assert _hit("size", "int: medium", [("size", "int:m")])
    assert _hit("size", "int:large", [("size", "int:l")])
    assert _hit("size", "int: xlarge", [("size", "int:xl")])
    assert _hit("size", "int: 2xlarge", [("size", "int:xxl")])
    # Symmetric — product has long form, reward has short
    assert _hit("size", "int:m", [("size", "int: medium")])


def test_size_expansion_only_on_size_family_keys():
    # Letter codes in unrelated slots must NOT expand (avoids `m` ⇒ `medium`
    # collision on non-size attributes like flavor codes)
    assert not _hit("flavor", "flavor m", [("flavor", "flavor medium")])
    # `size` in the key name (anywhere) triggers the expansion
    assert _hit("int_size", "m", [("int_size", "medium")])


def test_size_distinct_letters_do_not_collide():
    # `s` and `m` are distinct sizes; expanded to `small` vs `medium`, still distinct
    assert not _hit("size", "m", [("size", "s")])
    assert not _hit("size", "l", [("size", "xl")])


# ── token-set unordered match for compound values (ORO-1695) ─────
def test_token_subset_basic():
    assert _is_token_subset(["blue", "50l"], ["50l", "blue", "pedal"])
    assert not _is_token_subset(["red", "50l"], ["50l", "blue", "pedal"])


def test_token_subset_requires_min_two_tokens():
    # single-token guard: "black" alone must NOT match "black steel wheel"
    assert not _is_token_subset(["black"], ["black", "steel", "wheel"])
    assert not _is_token_subset(["small"], ["small", "medium"])


def test_compound_value_word_order_same_key():
    # Q5-adjacent: same key, tokens reordered — reward "pedal blue 50l" hits
    # product "50l blue pedal" via token-subset match under same key
    assert _hit("color_family", "pedal blue 50l", [("color_family", "50l blue pedal")])
    # separator drift also normalized (hyphens become spaces via NFKC/tokenize)
    assert _hit("color_family", "pedal-blue-50l", [("color_family", "50l/blue/pedal")])


def test_compound_value_disjoint_token_does_not_match():
    # anti-FP: a single reward token not present in the product must NOT hit
    # via any path — token-subset requires ≥2 tokens; subseq requires exact
    # whole-token presence
    assert not _hit("color_family", "green", [("color_family", "50l blue pedal")])


def test_compound_value_missing_token_does_not_match():
    # reward requires 3 tokens; product has only 2 — must NOT hit
    assert not _hit("color_family", "red blue green", [("color_family", "blue red")])


# ── curtain-type alias (ORO-1695) ────────────────────────────────
def test_curtain_type_alias():
    # Q1 curtains case: reward `type_curtain` vs product `curtain_type` on the same value
    assert _hit("type_curtain", "window curtain", [("curtain_type", "window curtain")])
    assert _hit("curtain_type", "curtains", [("type_curtain", "curtains")])
