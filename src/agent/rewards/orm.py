import json
import logging
import os
import re
import unicodedata
from collections import Counter

SENTENCE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
TITLE_SIM_THRESHOLD = 0.72

# Shadow scoring: when SHADOW_SCORE_MODEL is set, encode the same
# (product_title, gt_title) pair with the candidate model and log the
# canonical + shadow cosine similarity so we can grep CW Logs Insights and
# replay any threshold offline before promoting a model swap. Threshold is
# never enforced by shadow — we only need the raw similarity.
SHADOW_MODEL_ENV = "SHADOW_SCORE_MODEL"

_sentence_model = None
_sentence_model_unavailable = False
_shadow_model = None
_shadow_unavailable = False

_logger = logging.getLogger(__name__)

# ── Attribute matching ──────────────────────────────────────────────
# Reward attribute/sku values were historically matched to the product by
# exact (key, value) tuple. That produced false-negatives when a valid
# product encodes the same attribute under a different key spelling or with
# extra descriptor tokens (e.g. reward "size":"48mm" vs product "size":
# "1pcs 2#- 48mm"; reward "color family" vs product "color_family"). These
# helpers add value + key string-normalization and a *same-key* token-subset
# match. Everything stays exact-after-normalize — no fuzzy/embedding — so
# distinct values (43≠42, blue≠black) never collide. Semantic cross-key
# aliases (color↔color_family) are intentionally NOT handled here; they need
# corpus-derived key co-occurrence stats and are a separate change.

_WORD_RE = re.compile(r"[a-z0-9]+")

# Leading compatibility phrases that carry no discriminating information when a
# value is compared under a compat-family key (e.g. reward
# `compatibility_by_model: "for vivo y02a"` vs product
# `compatibility_by_model: "vivo y02a"`). Stripped inside `_attr_constraint_hit`
# only for keys that pass `_is_compat_key`, so unrelated slots like
# `feature: "for indoor use"` are not silently truncated.
_LEADING_COMPAT_PREFIXES = (
    "compatible with ",
    "compatible for ",
    "suitable for ",
    "fits ",
    "for ",
)

# Keys where the values carry compatibility statements and the leading phrase
# is boilerplate rather than semantic content. Narrow allowlist — expand only
# when a concrete miner case demonstrates the drift on a new key.
_COMPAT_KEYS = frozenset({"compatibility", "compatibilitybymodel", "compatiblewith", "compatiblefor"})


# Clothing size abbreviations expanded to their canonical long form so that
# reward `size: "int: medium"` can hit product `size: "int:m"` via same-key
# token-subsequence. Only applied when the value's key is EXACTLY the `size`
# slot (see `_is_size_key`) — avoids collisions on `ring_size` (letter codes
# denote ring letters), `screen_size` (inches), `package_size`, etc.
_SIZE_TOKEN_EXPANSION = {
    "s": "small",
    "m": "medium",
    "l": "large",
    "xl": "xlarge",
    "xxl": "2xlarge",
    "xxxl": "3xlarge",
    "2xl": "2xlarge",
    "3xl": "3xlarge",
    "4xl": "4xlarge",
}

# Regional size prefixes that legitimately precede a size letter/word
# (e.g. `int: medium`, `us s`, `eu xl`). Any other 2+-token size value is
# treated as dimensional / unit content (`l x w x h`, `5 l`, `s hook 10pcs`)
# and does NOT trigger expansion.
_SIZE_PREFIX_TOKENS = frozenset({"int", "us", "eu", "uk", "jp", "asia", "cn"})
_SIZE_WORDS = frozenset(_SIZE_TOKEN_EXPANSION.values())


def _looks_like_clothing_size(tokens: list) -> bool:
    """True when tokens match a clothing-size shape — either a bare size
    letter/word, or a region prefix followed by one. Rejects dimensional /
    unit values (e.g. `["l","x","w","h"]`, `["5","l"]`, `["s","hook","10pcs"]`)
    so `_expand_size_tokens` never fires on those."""
    if len(tokens) == 1:
        t = tokens[0]
        return t in _SIZE_TOKEN_EXPANSION or t in _SIZE_WORDS
    if len(tokens) == 2:
        prefix, sz = tokens
        return prefix in _SIZE_PREFIX_TOKENS and (sz in _SIZE_TOKEN_EXPANSION or sz in _SIZE_WORDS)
    return False


def _strip_compat_prefix(s: str) -> str:
    for prefix in _LEADING_COMPAT_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def _strip_compat_prefix_tokens(toks: list) -> list:
    """Token-list variant — drops the leading prefix tokens if the first N
    tokens match any prefix in `_LEADING_COMPAT_PREFIXES`. Avoids the
    `" ".join(toks)` → re-tokenize round-trip in the hot loop."""
    for prefix in _LEADING_COMPAT_PREFIXES:
        pt = prefix.strip().split()
        n = len(pt)
        if len(toks) >= n and toks[:n] == pt:
            return toks[n:]
    return toks


# Number+unit spacing: `1200 ml` ≡ `1200ml` (ORO-1797 / ex-ORO-1702 class).
# FUSES a spaced number + known unit suffix into one token (never splits) —
# a fused `1200ml` atom can only match another `1200ml` atom, so a bare-number
# reward (`42`) can never subsequence-match a unit-bearing product value
# (`42w`), and digit-first model codes (`pixel 4a`, `5g`) stay intact.
# Longest-first so `mm` wins over `m`, `pcs` over `pc`.
_NUM_UNIT_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s+"
    r"(mah|inch|pcs|mm|cm|km|ml|kg|kw|gb|mb|tb|oz|ft|pc|in|g|l|m|w|v|a)\b"
)


def normalize_attr_value(value) -> str:
    """NFKC + lowercase, strip bracket chars (full-width incl.), split
    digit→letter boundaries, join colon spacing, and collapse whitespace.
    Values are still compared for *exact equality* after this. Key-gated
    normalizations (compat-prefix strip, size expansion) run inside
    `_attr_constraint_hit`, not here — see those helpers.

    The number+unit fuse makes `1200 ml` ≡ `1200ml` and `2 pcs` ≡ `2pcs`
    (sellers space number/unit inconsistently across listings). Both spellings
    canonicalize to the FUSED atom, so token-subsequence semantics only get
    stricter — a bare number never gains a match against a unit-bearing
    value. Colon spacing (`eu: 38` ≡ `eu:38`) likewise joins; neither
    transform ever increases token count.
    """
    s = unicodedata.normalize("NFKC", str(value)).lower()
    s = re.sub(r"[【】\[\]\(\)（）]", " ", s)
    s = _NUM_UNIT_RE.sub(r"\1\2", s)
    s = re.sub(r"\s*:\s*", ":", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_attr_key(key) -> str:
    """Collapse a key to its comparison form: NFKC + lowercase, drop spaces/
    underscores, and strip a single trailing plural 's'. Makes
    `color family`==`color_family`==`colorfamily` and `flavors`==`flavor`."""
    k = unicodedata.normalize("NFKC", str(key)).lower()
    k = re.sub(r"[_\s]+", "", k)
    if len(k) > 3 and k.endswith("s"):
        k = k[:-1]
    return k


def _value_tokens(value) -> list:
    return _WORD_RE.findall(normalize_attr_value(value))


def _is_size_key(nk: str) -> bool:
    """Only the canonical clothing/apparel size slot. Keys like `ring_size`,
    `screen_size`, `package_size` intentionally excluded — letter codes there
    mean ring letters / inches / package classes, not garment sizes."""
    return nk == "size"


def _is_compat_key(nk: str) -> bool:
    """Compatibility-family keys — where a leading `for `/`fits ` phrase is
    boilerplate rather than semantic content."""
    return nk in _COMPAT_KEYS


def _expand_size_tokens(tokens: list) -> list:
    return [_SIZE_TOKEN_EXPANSION.get(t, t) for t in tokens]


def _is_token_subsequence(sub: list, seq: list) -> bool:
    """True if every token in `sub` appears in `seq` as a whole token, in
    order. Whole-token boundaries mean `8gb` is NOT a subsequence of the
    single token `128gb` — avoids numeric/unit collisions that raw substring
    matching would create."""
    if not sub:
        return False
    it = iter(seq)
    return all(tok in it for tok in sub)


def _build_attr_index(kv_pairs):
    """From an iterable of (key, value) product tuples build:
    - val_keys: normalized value -> set of normalized keys holding it
    - key_tokens: normalized key -> list of token-lists of its values

    For compat-family keys, both the raw and compat-prefix-stripped variants
    are indexed under the same key. This keeps the reward-vs-product match
    symmetric on prefix — a reward carrying `for X` matches a product `X`
    (and vice versa) without special-casing at the call site.
    """
    val_keys = {}
    key_tokens = {}
    for k, v in kv_pairs:
        nk = normalize_attr_key(k)
        nv = normalize_attr_value(v)
        toks = _value_tokens(v)
        val_keys.setdefault(nv, set()).add(nk)
        key_tokens.setdefault(nk, []).append(toks)
        if _is_compat_key(nk):
            stripped_nv = _strip_compat_prefix(nv)
            if stripped_nv != nv:
                val_keys.setdefault(stripped_nv, set()).add(nk)
            stripped_toks = _strip_compat_prefix_tokens(toks)
            if stripped_toks != toks:
                key_tokens[nk].append(stripped_toks)
    return val_keys, key_tokens


# Curated cross-key aliases: pairs of attribute keys that are semantically the
# same slot, so the SAME value found under either key satisfies the constraint
# (e.g. a `color_family` requirement met by the product's `color`). Hand-vetted:
# a dynamic corpus-co-occurrence approach over-matched because the catalog is
# dirty — unrelated keys share junk values (e.g. `voltage`/`color`
# both carry "4 modes 12v"), and no threshold separates real aliases from
# coincidence. So this is an explicit allowlist. Pairs are stored normalized.
_CROSS_KEY_ALIAS_PAIRS = [
    ("color", "color_family"),
    ("colour", "color_family"),  # British spelling
    ("color_classification", "color_family"),
    ("color_classification", "sort by color"),
    ("color_family", "warna"),  # 'warna' == color (id/ms)
    ("plate_size", "size"),
    ("cord_length", "length"),
    ("cookware_material", "material"),
    ("finish_lipstick", "lips makeup finish"),
    ("colored_gem_type", "main_stone"),
    ("model", "compatibility_by_model"),
    ("compatibility", "compatibility_by_model"),
    ("concern_oral_care", "oral care benefits"),
    ("flavor", "scent"),
    ("leather_material", "material"),
    ("leather_material", "material_filter"),
    # Category-specific key spelling drift observed in miner-audit cases.
    # Values overlap only in-category; distinctiveness stays intact.
    ("curtain_type", "type_curtain"),
]
_CROSS_KEY_ALIASES = frozenset(
    frozenset((normalize_attr_key(a), normalize_attr_key(b)))
    for a, b in _CROSS_KEY_ALIAS_PAIRS
)


def _attr_constraint_hit(reward_key, reward_value, val_keys, key_tokens) -> bool:
    """Does the product satisfy a single reward (key, value) constraint?

    1. Same-key exact match after value + key normalization. Compat-family
       keys have both raw and prefix-stripped variants indexed (see
       `_build_attr_index`), so `for X` on either side matches plain `X`.
    2. Same-key token-subsequence: reward value's tokens appear in one of the
       product's values under that (normalized) key, in order. For the exact
       `size` key, size letter codes (m/l/xl) expand to long forms — but only
       when both sides look like clothing-size values (`_looks_like_clothing_size`),
       so dimensional / unit content (`l x w h`, `5 l`, `s hook`) never expands.
    3. Cross-key alias: the same value (exact, after normalization) sits under a
       curated sibling key that is semantically the same slot (`_CROSS_KEY_ALIASES`).
    """
    nk = normalize_attr_key(reward_key)
    is_compat = _is_compat_key(nk)
    is_size = _is_size_key(nk)

    nv = normalize_attr_value(reward_value)
    if is_compat:
        nv = _strip_compat_prefix(nv)
    if nk in val_keys.get(nv, ()):
        return True

    rt = _WORD_RE.findall(nv) if is_compat else _value_tokens(reward_value)
    rt_size_expandable = is_size and _looks_like_clothing_size(rt)
    if rt_size_expandable:
        rt_matched = _expand_size_tokens(rt)
    else:
        rt_matched = rt

    if rt_matched:
        for ptoks in key_tokens.get(nk, ()):
            if rt_size_expandable and _looks_like_clothing_size(ptoks):
                pt = _expand_size_tokens(ptoks)
            else:
                pt = ptoks
            if _is_token_subsequence(rt_matched, pt):
                return True

    for pk in val_keys.get(nv, ()):
        if pk != nk and frozenset((nk, pk)) in _CROSS_KEY_ALIASES:
            return True
    return False


def _get_sentence_model():
    """Lazy-load sentence model (requires sentence-transformers).

    Returns None if sentence-transformers is not installed.
    """
    global _sentence_model, _sentence_model_unavailable
    if _sentence_model_unavailable:
        return None
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME)
        except ImportError:
            _sentence_model_unavailable = True
            return None
    return _sentence_model


def _get_shadow_model():
    """Lazy-load the shadow sentence model named by SHADOW_SCORE_MODEL."""
    global _shadow_model, _shadow_unavailable
    if _shadow_unavailable:
        return None
    name = os.environ.get(SHADOW_MODEL_ENV)
    if not name:
        return None
    if _shadow_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _shadow_model = SentenceTransformer(name)
        except (ImportError, OSError, ValueError) as exc:
            _logger.warning("Shadow scorer disabled — failed to load %s: %s", name, exc)
            _shadow_unavailable = True
            return None
    return _shadow_model


def _log_shadow_sim(product_title: str, gt_title: str, canonical_sim: float) -> None:
    """Encode the same title pair with the shadow model and log both sims.

    No-op when ``SHADOW_SCORE_MODEL`` is unset or the shadow model failed to
    load. One JSON line per pair; grep CW Logs Insights with ``shadow_sim``
    and apply any threshold offline to estimate the per-row effect of a
    swap. ORO-1474.
    """
    shadow = _get_shadow_model()
    if shadow is None:
        return
    try:
        product_emb = shadow.encode([product_title])
        gt_emb = shadow.encode([gt_title])
        shadow_sim = float(shadow.similarity(product_emb, gt_emb)[0][0])
    except Exception as exc:  # noqa: BLE001
        _logger.warning("Shadow sim failed for product=%r gt=%r: %s", product_title[:60], gt_title[:60], exc)
        return
    _logger.info(
        "shadow_sim %s",
        json.dumps(
            {
                "canonical_sim": canonical_sim,
                "shadow_sim": shadow_sim,
                "canonical_model": SENTENCE_MODEL_NAME,
                "shadow_model": os.environ.get(SHADOW_MODEL_ENV, ""),
                "product_title": product_title,
                "gt_title": gt_title,
            }
        ),
    )


def batch_encode_titles(titles: list[str]) -> list:
    """Batch encode multiple titles in one model.encode() call."""
    model = _get_sentence_model()
    if model is None or not titles:
        return []
    return list(model.encode(titles))


def ground_truth_reward(product: dict, reward: dict) -> float:
    if product["product_id"] == reward["product_id"]:
        return 1
    return 0


def rule_score_reward(product: dict, reward: dict, product_title_emb=None) -> tuple[float, Counter, Counter]:
    total_count = 0
    hit_count = 0
    total_counter = Counter()
    hit_counter = Counter()

    is_ground_truth = ground_truth_reward(product, reward) == 1

    # title — use precomputed embeddings if available, otherwise encode both
    if is_ground_truth and "title" in reward:
        # GT match: same product = same title, skip expensive embedding computation
        n_titles = len(reward["title"])
        total_count += n_titles
        total_counter["title"] += n_titles
        hit_count += n_titles
        hit_counter["title"] += n_titles
    elif "title" in reward:
        model = _get_sentence_model()
        precomputed = reward.get("_title_embeddings", {})
        # Precomputed embeddings shipped with the problem suite may have been
        # generated by a different model. Discard any whose dimensionality
        # does not match the current model — mixing embedders is meaningless
        # and would raise on similarity().
        expected_dim = model.get_sentence_embedding_dimension() if model is not None else None
        if model is not None or precomputed:
            # Use pre-encoded embedding if provided, otherwise encode now
            if product_title_emb is not None:
                product_emb = [product_title_emb]
            elif model is not None:
                product_emb = model.encode([product["title"]])
            else:
                product_emb = None
            if product_emb is not None:
                for title in reward["title"]:
                    cached = precomputed.get(title)
                    if cached is not None and expected_dim is not None and len(cached) == expected_dim:
                        gt_emb = [cached]
                    elif model is not None:
                        gt_emb = model.encode([title])
                    else:
                        continue
                    sim = model.similarity(product_emb, gt_emb)[0][0]
                    total_count += 1
                    total_counter["title"] += 1
                    if sim >= TITLE_SIM_THRESHOLD:
                        hit_count += 1
                        hit_counter["title"] += 1
                    _log_shadow_sim(product["title"], title, float(sim))
    # price
    if "price" in reward:
        price = product["price"]
        for price_range in reward["price"]:
            for mode, (lower_bound, upper_bound) in price_range.items():
                total_count += 1
                total_counter["price"] += 1
                if mode == "less than" and price <= upper_bound:
                    hit_count += 1
                    hit_counter["price"] += 1
                elif mode == "greater than" and price >= lower_bound:
                    hit_count += 1
                    hit_counter["price"] += 1
                elif mode == "between" and lower_bound <= price <= upper_bound:
                    hit_count += 1
                    hit_counter["price"] += 1
    # service
    if "service" in reward:
        for serv in reward["service"]:
            total_count += 1
            total_counter["service"] += 1
            if serv in product["service"]:
                hit_count += 1
                hit_counter["service"] += 1
    # sku options & attributes — normalized + same-key token-subset matching
    # (see _attr_constraint_hit). Attributes are always available; sku options
    # are variant-exclusive, so score against each option (plus an empty
    # baseline for attribute-only products) and keep the best-matching variant.
    attr_pairs = [
        (k, v)
        for k, vs in (product.get("attributes") or {}).items()
        for v in vs
    ]
    product_options = [[]]
    if product.get("sku_options"):
        for option in product["sku_options"].values():
            product_options.append(list(option.items()))

    max_total = 0
    max_hit = 0
    for option_pairs in product_options:
        val_keys, key_tokens = _build_attr_index(attr_pairs + option_pairs)
        cur_total = 0
        cur_hit = 0
        if "sku_options" in reward:
            for option in reward["sku_options"]:
                for k, v in option.items():
                    cur_total += 1
                    if _attr_constraint_hit(k, v, val_keys, key_tokens):
                        cur_hit += 1
        if "attributes" in reward:
            for attr in reward["attributes"]:
                for k, vs in attr.items():
                    for v in vs:
                        cur_total += 1
                        if _attr_constraint_hit(k, v, val_keys, key_tokens):
                            cur_hit += 1
        max_total = cur_total if cur_total > max_total else max_total
        max_hit = cur_hit if cur_hit > max_hit else max_hit
    total_count += max_total
    total_counter["sku & attrs"] += max_total
    hit_count += max_hit
    hit_counter["sku & attrs"] += max_hit

    if is_ground_truth:
        # Ground truth match — perfect score, but counters reflect actual field matches
        return 1, total_counter, hit_counter
    if total_count == 0:
        # No scoreable constraints — return 0 (conservative: no free credit)
        return 0, total_counter, hit_counter
    return hit_count / total_count, total_counter, hit_counter


def length_reward(output: list[dict]) -> float:
    if not output:
        return 0

    final_message = output[-1]["completion"]["message"]
    if (
        not final_message
        or "tool_call" not in final_message
        or not final_message["tool_call"]
    ):
        return 0

    is_terminated = False
    for commend in final_message["tool_call"]:
        if commend["name"] == "terminate":
            is_terminated = True
    if not is_terminated:
        return 0

    return 1.0 / len(output)
