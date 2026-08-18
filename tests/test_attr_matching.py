"""Tests for the normalized attribute matcher in rewards/orm.py.

Covers value/key normalization + same-key token-subset matching, curated
cross-key aliases, and the invariants that keep it exact-after-normalize
(distinct enums never collide, non-allowlisted keys never cross-match).
"""

import pytest

from src.agent.rewards.orm import (
    _attr_constraint_hit,
    _build_attr_index,
    _is_token_subsequence,
    _stem_form,
    _stem_token,
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


def test_value_normalization_numeric_unit_spacing():
    # spaced number+unit fuses to the unspaced atom: both spellings agree
    assert normalize_attr_value("1200 ml") == normalize_attr_value("1200ml") == "1200ml"
    assert normalize_attr_value("2 pcs") == normalize_attr_value("2pcs") == "2pcs"
    assert normalize_attr_value("3.0 w") == normalize_attr_value("3.0w") == "3.0w"
    # model/format codes and size words stay intact
    assert normalize_attr_value("samsung s22 plus") == "samsung s22 plus"
    assert normalize_attr_value("pixel 4a") == "pixel 4a"
    assert normalize_attr_value("a4") == "a4"
    assert normalize_attr_value("int: 2xlarge") == "int:2xlarge"


def test_numeric_unit_fuse_never_over_matches():
    # a bare-number reward must NOT match a unit-bearing product value —
    # the fuse direction keeps `42w` a single atom (reviewer regression set)
    assert not _hit("size", "42", [("size", "42w")])
    assert not _hit("capacity", "3000", [("capacity", "3000mah")])
    assert not _hit("size", "5", [("size", "5w")])
    assert not _hit("model", "pixel 4", [("model", "pixel 4a")])


def test_value_normalization_colon_spacing():
    assert normalize_attr_value("xs: 30x24x15cm") == normalize_attr_value("xs:30x24x15cm")
    assert normalize_attr_value("eu: 38") == normalize_attr_value("eu:38")


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


def test_numeric_unit_spacing_match():
    # ORO-1797 class 2: identical value, number/unit spacing drift across listings
    assert _hit("capacity", "1200 ml", [("capacity", "1200ml")])
    assert _hit("color_family", "2 pcs", [("color_family", "2pcs")])
    assert _hit("size", "xs:30x24x15cm", [("size", "xs: 30x24x15cm")])


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
    # ORO-1797 miner-audit batch
    assert _hit("glove size", "xl", [("size wear", "xl")])
    assert _hit("stationery_cutter_type", "scissors", [("tool_type_of_accessories", "scissors")])
    # exact-value only — different values under aliased keys must not match
    assert not _hit("glove size", "xl", [("size wear", "xxl")])
    assert not _hit("stationery_cutter_type", "scissors", [("tool_type_of_accessories", "screwdriver")])


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


# ── word-order guarantee preserved (regression test) ─────────────
def test_order_preserved_for_multi_token_values():
    # If we admitted an unordered subset match, permuted dimensional values
    # would false-hit (reward "10 20" vs product "20 10"). Same-key check
    # is strictly ordered.
    assert not _hit("size", "10 20", [("size", "20 10")])
    assert not _hit("dimensions", "3 5", [("dimensions", "5 3")])


# ── size expansion does NOT fire on dimensional / unit values (regression) ──
def test_size_expansion_skipped_on_dimensional_values():
    # `l` is a common dimension abbreviation (length, liters). It must not
    # expand to `large` when the surrounding tokens indicate dimensions or
    # units rather than a clothing size.
    assert not _hit("size", "large", [("size", "l x w x h")])
    assert not _hit("size", "large", [("size", "l: 30cm w: 20cm")])
    assert not _hit("size", "large", [("size", "5 l")])
    assert not _hit("size", "small", [("size", "s hook 10pcs")])


def test_size_expansion_fires_only_on_clothing_shape():
    # Bare size letter/word: expands
    assert _hit("size", "large", [("size", "l")])
    assert _hit("size", "l", [("size", "large")])
    # Region-prefixed size (int/us/eu/uk/jp/asia/cn) + size: expands
    assert _hit("size", "us large", [("size", "us l")])
    assert _hit("size", "eu xlarge", [("size", "eu xl")])
    # Multi-token value with no clothing-size shape: does not expand
    assert not _hit("size", "l", [("size", "5 l")])


# ── compat prefix — single-token subseq broadening (reviewer note) ──
def test_compat_single_token_subseq_after_strip():
    # After the leading `for ` is stripped from the reward side, `[iphone]`
    # becomes a valid single-token subsequence of any product value that
    # contains `iphone` under the same compat key. This IS intended — same
    # semantics as the pre-1695 single-token subseq path — but the widening
    # from compat-strip warrants an explicit test.
    assert _hit("compatibility", "for iphone", [("compatibility", "iphone 15 pro max")])


def test_compat_cross_key_alias_symmetric_with_prefix():
    # Compat cross-key alias must work regardless of which side carries the
    # `for ` prefix. Both raw and stripped variants are indexed for compat
    # keys, so a reward under `compatibility` matches a product under its
    # curated sibling `compatibility_by_model` with the prefix on either side.
    assert _hit("compatibility", "for vivo y02a", [("compatibility_by_model", "vivo y02a")])
    assert _hit("compatibility", "vivo y02a", [("compatibility_by_model", "for vivo y02a")])


# ── ORO-1924 morphological stem ─────────────────────────────────
def test_stem_token_strips_common_suffixes():
    assert _stem_token("mixed") == "mix"
    assert _stem_token("powders") == "powder"
    assert _stem_token("printing") == "print"
    assert _stem_token("adults") == "adult"
    # Regex is greedy on the `[a-z]{3,}` base, so `boxes` / `dishes` /
    # `boxes` lose only the trailing `s` (→ `boxe`, `dishe`). That leaves
    # some `-es` plurals unmerged, but the backtest confirmed the
    # backwards-compat behavior of the plural-`s` strip is enough to catch
    # 74 clean flips on 10 races with 0 false accepts. Broadening to a
    # proper Porter-style `es` rule is deferred until real data shows the
    # gap. Pinning current behavior here so a future change is deliberate.
    assert _stem_token("boxes") == "boxe"
    assert _stem_token("dishes") == "dishe"


def test_stem_token_min_base_length():
    # base < 3 chars: don't strip — avoids `used → us`, `ing → i`.
    assert _stem_token("used") == "used"
    assert _stem_token("ing") == "ing"
    assert _stem_token("ed") == "ed"


def test_stem_token_double_consonant_s_kept():
    # `pass → pas` is exactly the collision we're guarding against.
    assert _stem_token("pass") == "pass"
    assert _stem_token("class") == "class"
    assert _stem_token("dress") == "dress"


def test_stem_token_no_suffix_unchanged():
    for w in ("black", "cotton", "wall", "brand", "mix", "no"):
        assert _stem_token(w) == w


def test_stem_form_empty_when_no_ascii_tokens():
    # Empty-stem bucket would collect `.`, `-`, non-ASCII values under one
    # bogus canonical. `_stem_form` must return "" so `_attr_constraint_hit`
    # skips the stem branch instead of forcing a false collision.
    assert _stem_form("") == ""
    assert _stem_form(".") == ""
    assert _stem_form("...") == ""
    assert _stem_form("黑色") == ""


def test_stem_form_joins_and_stems_tokens():
    assert _stem_form("mix brands") == "mix brand"
    assert _stem_form("mixed brands") == "mix brand"
    assert _stem_form("no stones") == "no stone"
    assert _stem_form("32 ohms") == "32 ohm"


# Miner-reported ORO-1924 case + representative flips from the 10-race
# backtest (75 flips: 74 clean, 1 borderline, 0 false accepts). Each pair
# was previously a false FAIL; after the stem fix each must PASS in both
# directions (reward=singular vs plural is symmetric in the wild).
_STEM_FLIP_PAIRS = [
    # (attribute_key, side_a, side_b, comment)
    ("brand", "mixed brands", "mix brands", "ORO-1924 target case"),
    ("mainstone", "no stones", "no stone", "top flip pattern, ×11 in backtest"),
    ("formulationsupplement", "powder", "powders", "×7"),
    ("typeofhangers/hook", "wall mounted", "wall mount", "×3"),
    ("impedance", "32 ohms", "32 ohm", "×3, numeric-unit intact through stem"),
    ("ingredient", "bcaas", "bcaa", "×3, uppercase acronym pluralized"),
    ("stationerypentype", "ballpoint pens", "ballpoint pen", "×2"),
    ("topstype", "t-shirt", "t-shirts", "×1, plural drift"),
    ("swimweartype", "bikini sets", "bikini set", "×1"),
    ("packagingtype", "sachet", "sachets", "×1"),
    ("skirtstyle", "ruffle skirts", "ruffle skirt", "×2"),
    ("fageneralstyle", "basics", "basic", "×2"),
    ("lifestage", "adult", "adults", "×1"),
    ("fmltskincare", "cream", "creams", "×1"),
]


@pytest.mark.parametrize("key,side_a,side_b,comment", _STEM_FLIP_PAIRS)
def test_stem_flip_pairs_match_both_directions(key, side_a, side_b, comment):
    assert _hit(key, side_a, [(key, side_b)]), (
        f"{comment}: reward={side_a!r} did not match product={side_b!r}"
    )
    assert _hit(key, side_b, [(key, side_a)]), (
        f"{comment}: reward={side_b!r} did not match product={side_a!r}"
    )


def test_stem_does_not_bucket_empty_string_values():
    # The one failure mode found in the backtest: values that normalize to
    # zero tokens (`.`, `-`, `..`, `...`, `@`) would all share stem `""` and
    # be treated as equivalent. Empty-stem guard blocks that.
    assert not _hit("colorfamily", ".", [("colorfamily", "-")])
    assert not _hit("colorfamily", "...", [("colorfamily", "@")])
    assert not _hit("color", "黑色", [("color", "红色")])


def test_stem_does_not_collide_across_short_tokens():
    # Distinct enum values that share a suffix but have <3-char bases must
    # not stem into each other. `red` vs `reds` fine (both stem to `red`),
    # but `red` vs `use` — no overlap.
    assert not _hit("color", "red", [("color", "blue")])
    assert not _hit("color", "use", [("color", "red")])
    assert not _hit("gender", "men", [("gender", "women")])


def test_stem_does_not_over_match_distinct_semantics():
    # Distinct SKU dimensions must stay distinct through stem — the number+
    # unit fuse and 3-char-base guard together prevent cross-value creep.
    assert not _hit("size", "42w", [("size", "42")])
    assert not _hit("capacity", "3000mah", [("capacity", "3000")])
    assert not _hit("color", "black", [("color", "white")])


def test_stem_symmetric_with_token_subseq():
    # Stem branch composes with token-subsequence: a plural reward can hit
    # a longer-tokened product whose stemmed form contains the reward's
    # stemmed tokens as a subsequence.
    assert _hit(
        "stationerypenciltype",
        "graphite pencil",
        [("stationerypenciltype", "graphite & lead pencils")],
    )
    assert _hit(
        "packagingtype",
        "bottle",
        [("packagingtype", "plastic bottles")],
    )


def test_stem_does_not_regress_existing_exact_matches():
    # Simple positive: after the fix the exact-match path is still fastest
    # and stem branch never fires when it isn't needed.
    assert _hit("brand", "nike", [("brand", "nike")])
    assert _hit("color", "black", [("color", "black")])
