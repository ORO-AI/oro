"""
Agent for ShoppingBench sandbox execution.

Implements a hybrid code+LLM approach: Python handles mechanical work
(query parsing, searching, tool execution, recommend+terminate) while
the LLM handles evaluation of search results against queries. This
reduces LLM calls from 9+ to 2-3 per problem, avoiding timeouts.
"""

import re
import json
import logging
from typing import Dict, List, Optional
from urllib.parse import quote_plus

from src.agent.agent_interface import (
    create_dialogue_step,
    execute_tool_call,
    Tool,
    generate_tool_call_id,
)
from src.agent.proxy_client import ProxyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_STEPS = 25

_proxy = ProxyClient(timeout=120, max_retries=2)


# ── Query parser (inlined for single-file submission) ─────────────────────────

_FILLER_WORDS = frozenset([
    "show", "me", "find", "looking", "for", "a", "an", "the", "i", "im",
    "i'm", "want", "need", "search", "get", "please", "can", "you", "help",
    "that", "which", "with", "and", "or", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "must", "of", "in",
    "on", "at", "to", "from", "by", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "but", "if", "not",
    "no", "nor", "only", "also", "very", "too", "so", "just", "than",
    "then", "now", "here", "there", "this", "these", "those", "each",
    "every", "all", "any", "both", "few", "more", "most", "other", "some",
    "such", "it", "its", "my", "our", "their", "your", "his", "her", "up",
    "out", "about", "over", "under", "again", "further", "once",
    "php", "pesos", "peso", "priced", "cost", "costs", "costing",
    "price", "pricing", "options", "option", "available", "category",
    "specifically", "focus", "please", "compatible",
    "like", "such", "comes", "come", "offers", "offering", "offer",
    "sells", "selling", "sell", "shop", "store", "shops", "stores",
    "item", "items", "product", "products",
    "first", "second", "third", "lastly", "next", "also",
    "run", "runs", "running", "made", "makes", "make",
    "ranging", "range", "ranges",
    "called", "named", "models", "model", "type", "types",
    "belongs", "suitable", "searching", "pieces", "piece",
])
_PRICE_WORDS = frozenset([
    "priced", "price", "pricing", "cost", "costs", "costing",
    "above", "below", "over", "under", "more", "less", "than",
    "between", "from", "to", "ranging", "range", "php", "pesos", "peso", "budget",
])
_SERVICE_WORDS = frozenset([
    "lazmall", "lazflash", "free", "shipping", "cash", "delivery",
    "cod", "authenticity", "authentic", "official", "guaranteed",
    "guarantee", "returns", "perks", "flash", "deal", "deals", "limited-time", "big",
])


def _is_year_range(a, b):
    try:
        va, vb = int(float(a)), int(float(b))
        return 1900 <= va <= 2100 and 1900 <= vb <= 2100
    except ValueError:
        return False


def _looks_like_year(s):
    try:
        v = int(float(s))
        return 1900 <= v <= 2100
    except ValueError:
        return False


def _clean_num(s):
    f = float(s)
    return str(int(f)) if f == int(f) else s


def extract_price_range(text):
    t = text.lower()
    m = re.search(r"(?:priced?|pricing|costs?|ranging)\s+from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)", t)
    if m and not _is_year_range(m.group(1), m.group(2)):
        return f"{_clean_num(m.group(1))}-{_clean_num(m.group(2))}"
    m = re.search(r"from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s+(?:php|pesos?)", t)
    if m:
        return f"{_clean_num(m.group(1))}-{_clean_num(m.group(2))}"
    m = re.search(r"between\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)", t)
    if m and not _is_year_range(m.group(1), m.group(2)):
        return f"{_clean_num(m.group(1))}-{_clean_num(m.group(2))}"
    m = re.search(r"(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s+(?:php|pesos?)", t)
    if m:
        return f"{_clean_num(m.group(1))}-{_clean_num(m.group(2))}"
    m = re.search(r"price\s+ranging\s+from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)", t)
    if m:
        return f"{_clean_num(m.group(1))}-{_clean_num(m.group(2))}"
    m = re.search(r"(?:priced?\s+|costs?\s+)?(?:above|over|more\s+than|exceeds?|greater\s+than)\s+(\d+(?:\.\d+)?)\s*(?:php|pesos?)?", t)
    if m and not _looks_like_year(m.group(1)):
        return f"{_clean_num(m.group(1))}-"
    m = re.search(r"costs?\s+(?:over|more\s+than|above)\s+(\d+(?:\.\d+)?)", t)
    if m:
        return f"{_clean_num(m.group(1))}-"
    m = re.search(r"priced\s+(?:over|above)\s+(\d+(?:\.\d+)?)", t)
    if m:
        return f"{_clean_num(m.group(1))}-"
    m = re.search(r"price\s+(?:above|over)\s+(\d+(?:\.\d+)?)", t)
    if m:
        return f"{_clean_num(m.group(1))}-"
    m = re.search(r"(?:below|under|less\s+than|cheaper\s+than)\s+(\d+(?:\.\d+)?)", t)
    if m:
        return f"0-{_clean_num(m.group(1))}"
    return None


def extract_service_filters(text):
    t = text.lower()
    services = []
    if any(kw in t for kw in ["lazmall", "100% authentic", "authenticity", "guaranteed authentic"]):
        services.append("official")
    if "free shipping" in t:
        services.append("freeShipping")
    if "cash on delivery" in t or "cod" in t.split():
        services.append("COD")
    if any(kw in t for kw in ["lazflash", "flash deal", "flash sale"]):
        services.append("flashsale")
    return ",".join(services) if services else None


def detect_task_type(query):
    q = query.lower()
    if "voucher" in q or "budget" in q:
        return "voucher"
    if "shop" in q and any(kw in q for kw in ["both", "these", "offering", "sells", "offers"]):
        return "shop"
    if re.search(r"shops?\s+(?:offering|that\s+offer|selling)", q):
        return "shop"
    has_ord = bool(re.search(r"\b(?:first|second|third|lastly)\b", q))
    if has_ord and "budget" in q:
        return "voucher"
    if has_ord:
        return "shop"
    return "product"


def extract_voucher_params(query):
    q = query.lower()
    if "voucher" not in q and "budget" not in q:
        return None
    params = {}
    m = re.search(r"budget\s+(?:is\s+(?:only\s+)?|of\s+)[`]?(\d+(?:\.\d+)?)[`]?", q)
    if m:
        params["budget"] = float(m.group(1))
    m = re.search(r"total\s+price\s+.*?exceeds?\s+[`]?(\d+(?:\.\d+)?)[`]?", q)
    if m:
        params["threshold"] = float(m.group(1))
    m = re.search(r"percentage\s+discount\s+of\s+[`]?(\d+(?:\.\d+)?)%?[`]?", q)
    if m:
        params["voucher_type"] = "percentage"
        params["discount_value"] = float(m.group(1))
    else:
        m = re.search(r"fixed\s+discount\s+of\s+[`]?(\d+(?:\.\d+)?)[`]?", q)
        if m:
            params["voucher_type"] = "fixed"
            params["discount_value"] = float(m.group(1))
    m = re.search(r"cap\s+of\s+[`]?(\d+(?:\.\d+)?)[`]?", q)
    if m:
        params["cap"] = float(m.group(1))
    return params if params else None


def _remove_price_phrases(text):
    c = re.sub(r"(?:priced?|costs?|costing|pricing)\s+(?:above|over|below|under|from|more\s+than|less\s+than|ranging\s+from)\s+[\d,.]+\s*(?:to\s+[\d,.]+\s*)?(?:php|pesos?)?", " ", text, flags=re.IGNORECASE)
    c = re.sub(r"(?:above|over|below|under)\s+[\d,.]+\s*(?:php|pesos?)?", " ", c, flags=re.IGNORECASE)
    c = re.sub(r"from\s+[\d,.]+\s+to\s+[\d,.]+\s*(?:php|pesos?)?", " ", c, flags=re.IGNORECASE)
    c = re.sub(r"between\s+[\d,.]+\s+(?:and|to)\s+[\d,.]+", " ", c, flags=re.IGNORECASE)
    c = re.sub(r"\b\d+\s+to\s+\d+\s*(?:php|pesos?)\b", " ", c, flags=re.IGNORECASE)
    c = re.sub(r"price\s+ranging\s+from\s+[\d,.]+\s+to\s+[\d,.]+\s*(?:php|pesos?)?", " ", c, flags=re.IGNORECASE)
    c = re.sub(r",?\s+and\s+(?:is\s+)?priced\s+\w+\s+[\d,.]+\s*(?:php|pesos?)?", " ", c, flags=re.IGNORECASE)
    c = re.sub(r"with\s+a\s+price\s+(?:above|over|below|under|ranging\s+from)\s+[\d,.]+(?:\s+to\s+[\d,.]+)?\s*(?:php|pesos?)?", " ", c, flags=re.IGNORECASE)
    return c


def _remove_service_phrases(text):
    c = re.sub(r"(?:with\s+)?(?:LazMall|LazFlash)\s+(?:perks?\s+)?(?:such\s+as\s+)?[^,.]*", " ", text, flags=re.IGNORECASE)
    c = re.sub(r"(?:available\s+)?with\s+(?:a\s+)?(?:big\s+)?(?:limited[- ]time\s+)?(?:LazFlash|LazMall)\s+(?:deal|perks?|authenticity)?[^,.]*", " ", c, flags=re.IGNORECASE)
    c = re.sub(r"(?:free\s+shipping|cash\s+on\s+delivery|cod)\s*(?:options?)?", " ", c, flags=re.IGNORECASE)
    c = re.sub(r"with\s+(?:free\s+shipping|cash\s+on\s+delivery|cod)(?:\s+and\s+(?:free\s+shipping|cash\s+on\s+delivery|cod))*\s*(?:options?)?", " ", c, flags=re.IGNORECASE)
    c = re.sub(r"(?:LazMall|LazFlash)\s+(?:authenticity|fast\s+delivery)[^,.]*", " ", c, flags=re.IGNORECASE)
    return c


def extract_search_keywords(text, max_words=4):
    cleaned = _remove_price_phrases(text)
    cleaned = _remove_service_phrases(cleaned)
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
    cleaned = re.sub(r"\b\d{4,}[-]\d+[-]\d+\b", " ", cleaned)
    cleaned = re.sub(r"\b\d{5,}\b", " ", cleaned)
    words = re.findall(r"[A-Za-z]+(?:\'[a-z]+)?", cleaned)
    priority, normal = [], []
    for w in words:
        wl = w.lower()
        if wl in _FILLER_WORDS or wl in _PRICE_WORDS or wl in _SERVICE_WORDS or len(wl) <= 1:
            continue
        if w[0].isupper() and wl not in _FILLER_WORDS:
            if wl not in [pw.lower() for pw in priority]:
                priority.append(w)
        else:
            if wl not in [nw.lower() for nw in normal]:
                normal.append(w)
    kws = []
    for w in priority:
        if len(kws) >= max_words:
            break
        kws.append(w.lower())
    for w in normal:
        if len(kws) >= max_words:
            break
        if w.lower() not in kws:
            kws.append(w.lower())
    if not kws:
        for w in words:
            if len(w) > 2 and w.lower() not in _FILLER_WORDS:
                kws.append(w.lower())
                if len(kws) >= max_words:
                    break
    return " ".join(kws[:max_words])


def _strip_voucher_section(query):
    parts = re.split(r"\n\s*My budget", query, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[0].strip()
    parts = re.split(r"\n\s*\d+\.\s+The voucher", query, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[0].strip()
    return query


def _is_voucher_only(text):
    t = text.lower().strip()
    if t.startswith("my budget"):
        return True
    if re.match(r"^\d+\.\s+(the\s+voucher|it\s+is\s+valid|it\s+provides)", t):
        return True
    if "voucher" in t and len(t) < 200 and "product" not in t:
        return True
    return False


def _is_preamble(text, full_query):
    t = text.lower().strip().rstrip(".:,")
    if len(text) > len(full_query) * 0.8:
        return False
    if not bool(re.search(r"\b(?:first|second|third|lastly)\b", full_query.lower())):
        return False
    if len(t) < 120 and any(t.startswith(p) for p in [
        "looking for", "i'm looking", "im looking", "i am looking",
        "show me", "find me", "search for", "find shops", "i want", "i need",
    ]):
        return True
    if text.strip().endswith(":"):
        return True
    return False


def _build_spec(text):
    return {
        "description": text.strip().rstrip("."),
        "price": extract_price_range(text),
        "service": extract_service_filters(text),
        "keywords": extract_search_keywords(text),
    }


def _split_on_ordinals(query):
    pattern = r"(?:^|[.:]\s*|\n\s*|,\s+)\s*(?:First|Second|Third|Lastly|Next|Also(?:\s+searching)?),?\s+"
    parts = re.split(pattern, query, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p and p.strip()]


def _split_on_connectors(query):
    product_text = _strip_voucher_section(query)
    parts = re.split(r",?\s+and\s+also\s+", product_text, flags=re.IGNORECASE)
    if len(parts) > 1:
        return [p.strip() for p in parts if p.strip()]
    task = detect_task_type(query)
    if task in ("shop", "voucher"):
        stripped = re.sub(r"^.*?(?:offering|that\s+offers?|selling|that\s+sells?)\s+", "", product_text, count=1, flags=re.IGNORECASE)
        if stripped != product_text:
            comma_and_parts = re.split(r",\s+and\s+", stripped, flags=re.IGNORECASE)
            all_parts = []
            for part in comma_and_parts:
                all_parts.extend(re.split(r",\s+", part))
            all_parts = [p.strip() for p in all_parts if p.strip()]
            if len(all_parts) > 1:
                return all_parts
    lines = [l.strip() for l in product_text.split("\n") if l.strip()]
    product_lines = [l for l in lines if not _is_voucher_only(l)]
    if len(product_lines) > 1:
        return product_lines
    return [product_text]


def extract_product_specs(query):
    segments = _split_on_ordinals(query)
    if len(segments) <= 1:
        segments = _split_on_connectors(query)
    if len(segments) <= 1:
        task = detect_task_type(query)
        if task == "product":
            return []
        product_text = _strip_voucher_section(query)
        if product_text.strip():
            return [_build_spec(product_text)]
        return []
    specs = []
    for idx, seg in enumerate(segments):
        seg = seg.strip()
        if not seg:
            continue
        if _is_voucher_only(seg):
            continue
        if idx == 0 and _is_preamble(seg, query):
            continue
        specs.append(_build_spec(seg))
    return specs


def parse_query(query):
    task = detect_task_type(query)
    product_specs = extract_product_specs(query)
    if task == "product":
        price = extract_price_range(query)
        service = extract_service_filters(query)
        keywords = extract_search_keywords(query)
    else:
        product_text = _strip_voucher_section(query)
        price = extract_price_range(product_text)
        service = extract_service_filters(product_text)
        keywords = extract_search_keywords(product_text)
    voucher_params = extract_voucher_params(query) if task == "voucher" else None
    return {
        "task": task, "keywords": keywords, "price": price, "service": service,
        "product_specs": product_specs, "voucher_params": voucher_params, "raw_query": query,
    }


# ── Tool definitions ─────────────────────────────────────────────────────────


@Tool
def find_product(
    q: str,
    page: int = 1,
    shop_id: Optional[str] = None,
    price: Optional[str] = None,
    sort: Optional[str] = None,
    service: Optional[str] = None,
) -> List[Dict]:
    """
    Search for products and return up to 10 products per page. Use this tool to find products matching the user's needs.

    Args:
        q: Search query for products, e.g. "nike shoes" or "backpack for college student"
        page: Page number for pagination (1-5), use to browse more results
        shop_id: Filter results to products from a specific shop
        price: Price range filter, e.g. "0-100", "100-1000", "1000-" (open-ended)
        sort: Sort method - "priceasc" (price low to high), "pricedesc" (price high to low), "order" (by sales volume descending), "default" (relevance ranking)
        service: Comma-separated service filters - "official" (LazMall: 100% authenticity guarantee, 15-day returns), "freeShipping" (free shipping), "COD" (cash on delivery), "flashsale" (LazFlash: limited-time promotions), "default" (no filter)

    Returns:
        List of product dicts with product_id, shop_id, title, price, service, sold_count
    """
    q_encoded = quote_plus(q)
    params = {
        "q": q_encoded,
        "page": page,
        "shop_id": shop_id,
        "price": price,
        "sort": sort,
        "service": service,
    }
    if params.get("sort") == "default":
        params.pop("sort")
    if params.get("service") == "default":
        params.pop("service")
    elif params.get("service") and "default" in params["service"]:
        params["service"] = ",".join(
            x for x in params["service"].split(",") if x != "default"
        )
    result = _proxy.get("/search/find_product", params)
    result = result if result is not None else []
    # Auto-retry with broader search when within-shop search returns empty
    if shop_id and not result:
        # Retry 1: drop service filter
        if service:
            retry_params = dict(params)
            retry_params.pop("service", None)
            result = _proxy.get("/search/find_product", retry_params)
            result = result if result is not None else []
        # Retry 2: use shorter query (first 2 words)
        if not result:
            short_q = " ".join(q.split()[:2])
            if short_q != q:
                retry_params = dict(params)
                retry_params["q"] = quote_plus(short_q)
                retry_params.pop("service", None)
                result = _proxy.get("/search/find_product", retry_params)
                result = result if result is not None else []
    return result


@Tool
def view_product_information(product_ids: str) -> List[Dict]:
    """
    Get detailed product information for given product IDs.

    Args:
        product_ids: Comma-separated list of product IDs

    Returns:
        List of product information dicts with full details (title, price, attributes, etc.)
    """
    params = {"product_ids": product_ids}
    result = _proxy.get("/search/view_product_information", params)
    return result if result is not None else []


@Tool
def recommend_product(product_ids: str) -> str:
    """
    Recommend products to the user. You can use this tool only once.

    Args:
        product_ids: Comma-separated product IDs. For a single product match, provide one ID. For multiple products, provide all IDs in the order the user requested them. For products from the same shop, provide all product IDs from that shop in the user-specified order.

    Returns:
        Confirmation message
    """
    return f"Having recommended the products to the user: {product_ids}."


@Tool
def terminate(status: str = "success") -> str:
    """
    End the dialogue when the task is complete or you cannot proceed further.

    Args:
        status: Task outcome - "success" if products were recommended, "failure" if unable to find matching products

    Returns:
        Termination confirmation message
    """
    return f"The interaction has been completed with status: {status}"


# ── Helper tools (client-side, no backend changes needed) ────────────────────


@Tool
def check_product_match(product_id: str, requirements: str) -> Dict:
    """
    Check if a product matches specific attribute requirements. Returns a detailed match report.
    Use this BEFORE recommend_product to verify the product is correct.

    Args:
        product_id: Single product ID to check
        requirements: JSON string of required attributes, e.g. '{"brand": "yamaha", "color": "black", "material": "plastic", "size": "large"}'

    Returns:
        Dict with match result: {matched: bool, matches: [...], mismatches: [...], product_summary: {...}}
    """
    # Fetch full product info
    params = {"product_ids": product_id}
    info_list = _proxy.get("/search/view_product_information", params)
    if not info_list:
        return {
            "matched": False,
            "error": "Product not found",
            "matches": [],
            "mismatches": list(json.loads(requirements).keys()),
        }

    info = info_list[0] if isinstance(info_list, list) else info_list
    attrs = info.get("attributes", {})
    sku_opts = info.get("sku_options", [])

    try:
        reqs = (
            json.loads(requirements) if isinstance(requirements, str) else requirements
        )
    except json.JSONDecodeError:
        return {"matched": False, "error": "Invalid requirements JSON"}

    # Build a flat searchable text from all product fields
    all_values = []
    for v in attrs.values():
        if isinstance(v, list):
            all_values.extend(str(x).lower() for x in v)
        else:
            all_values.append(str(v).lower())
    for opt in sku_opts:
        if isinstance(opt, dict):
            for v in opt.values():
                if isinstance(v, list):
                    all_values.extend(str(x).lower() for x in v)
                else:
                    all_values.append(str(v).lower())
    searchable = " ||| ".join(all_values)
    # Also include description
    desc = (
        info.get("short_description", "") + " " + info.get("description", "")
    ).lower()

    matches = []
    mismatches = []
    for key, required_val in reqs.items():
        req_lower = str(required_val).lower()
        # Check attributes directly
        found = False
        # Direct attribute key match
        if key.lower() in {k.lower() for k in attrs}:
            attr_val = next(v for k, v in attrs.items() if k.lower() == key.lower())
            attr_str = (
                str(attr_val).lower()
                if not isinstance(attr_val, list)
                else " ".join(str(x).lower() for x in attr_val)
            )
            if req_lower in attr_str:
                found = True
        # Fuzzy match across all values
        if not found and req_lower in searchable:
            found = True
        # Check description as fallback
        if not found and req_lower in desc:
            found = True

        if found:
            matches.append(key)
        else:
            mismatches.append(key)

    return {
        "matched": len(mismatches) == 0,
        "matches": matches,
        "mismatches": mismatches,
        "product_summary": {
            "product_id": info.get("product_id", product_id),
            "attributes": {k: v for k, v in list(attrs.items())[:10]},
        },
    }


@Tool
def find_products_in_same_shop(product_queries: str) -> Dict:
    """
    Find multiple products that are ALL available from the SAME shop.
    Automatically searches across multiple shops. Use this for shop and voucher tasks.

    Args:
        product_queries: JSON array of product search specs, e.g. '[{"q": "foam roller", "price": "0-500"}, {"q": "yoga mat", "price": "0-300"}]'. Each item can have: q (required), price (optional), service (optional).

    Returns:
        Dict with: {found: bool, shop_id: str, products: [{product_id, title, price, shop_id}, ...], shops_tried: int}. Products are returned in the SAME ORDER as the input queries.
    """
    try:
        specs = (
            json.loads(product_queries)
            if isinstance(product_queries, str)
            else product_queries
        )
    except json.JSONDecodeError:
        return {"found": False, "error": "Invalid JSON for product_queries"}

    if not specs or not isinstance(specs, list):
        return {
            "found": False,
            "error": "product_queries must be a non-empty JSON array",
        }

    # Step 1: Search for the first product to get candidate shops
    first = specs[0]
    first_q = quote_plus(first.get("q", ""))
    first_params = {"q": first_q, "page": 1}
    if first.get("price"):
        first_params["price"] = first["price"]
    if first.get("service"):
        first_params["service"] = first["service"]

    first_results = _proxy.get("/search/find_product", first_params)
    if not first_results:
        return {
            "found": False,
            "error": f"No results for first product: {first.get('q')}",
            "shops_tried": 0,
        }

    # Collect unique shop_ids from first product results
    candidate_shops = []
    seen_shops = set()
    for p in first_results:
        sid = str(p.get("shop_id", ""))
        if sid and sid not in seen_shops:
            candidate_shops.append({"shop_id": sid, "first_product": p})
            seen_shops.add(sid)

    # Step 2: For each candidate shop, try to find ALL remaining products
    for shop_info in candidate_shops[:10]:  # Try up to 10 shops
        shop_id = shop_info["shop_id"]
        found_products = [shop_info["first_product"]]  # First product already matched
        all_found = True

        for spec in specs[1:]:
            q = spec.get("q", "")
            q_encoded = quote_plus(q)
            params = {"q": q_encoded, "page": 1, "shop_id": shop_id}
            if spec.get("price"):
                params["price"] = spec["price"]

            results = _proxy.get("/search/find_product", params)
            results = results if results is not None else []

            # Auto-retry: drop service, then shorten query
            if not results:
                params.pop("service", None)
                results = _proxy.get("/search/find_product", params)
                results = results if results is not None else []
            if not results:
                short_q = " ".join(q.split()[:2])
                if short_q != q:
                    params["q"] = quote_plus(short_q)
                    results = _proxy.get("/search/find_product", params)
                    results = results if results is not None else []
            if not results:
                # Try single word
                single_q = q.split()[0] if q.split() else q
                if single_q != short_q:
                    params["q"] = quote_plus(single_q)
                    results = _proxy.get("/search/find_product", params)
                    results = results if results is not None else []

            if results:
                found_products.append(results[0])  # Best match from this shop
            else:
                all_found = False
                break

        if all_found:
            return {
                "found": True,
                "shop_id": shop_id,
                "products": [
                    {
                        "product_id": p.get("product_id"),
                        "title": p.get("title", ""),
                        "price": p.get("price"),
                        "shop_id": p.get("shop_id"),
                    }
                    for p in found_products
                ],
                "shops_tried": candidate_shops.index(shop_info) + 1,
            }

    return {
        "found": False,
        "error": f"Could not find all {len(specs)} products in any single shop",
        "shops_tried": min(len(candidate_shops), 10),
    }


@Tool
def calculate_voucher(
    product_prices: str,
    voucher_type: str,
    discount_value: float,
    threshold: float,
    budget: float,
    cap: float = 0,
) -> Dict:
    """
    Calculate the final price after applying a voucher discount. Use this for voucher tasks to verify budget.

    Args:
        product_prices: Comma-separated product prices, e.g. "100,50,75"
        voucher_type: "fixed" for fixed discount, "percentage" for percentage discount
        discount_value: The discount amount (e.g. 18 for fixed, 42 for 42% percentage)
        threshold: Minimum total price for voucher to apply
        budget: Maximum budget the user has
        cap: Maximum discount amount for percentage vouchers (0 = no cap)

    Returns:
        Dict with: {total_before, discount_amount, total_after, within_budget, voucher_applied}
    """
    try:
        prices = [float(p.strip()) for p in str(product_prices).split(",")]
    except ValueError:
        return {"error": "Invalid product_prices format. Use comma-separated numbers."}

    total = sum(prices)
    discount = 0.0
    voucher_applied = False

    if total >= threshold:
        voucher_applied = True
        if voucher_type == "fixed":
            discount = discount_value
        elif voucher_type == "percentage":
            discount = total * (discount_value / 100.0)
            if cap > 0:
                discount = min(discount, cap)

    total_after = total - discount

    return {
        "prices": prices,
        "total_before": round(total, 2),
        "discount_amount": round(discount, 2),
        "total_after": round(total_after, 2),
        "within_budget": total_after <= budget,
        "voucher_applied": voucher_applied,
        "budget": budget,
    }


# ── LLM inference ────────────────────────────────────────────────────────────


def inference(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
) -> Dict:
    """Make LLM inference request via proxy to Chutes API (OpenAI-compatible)."""
    request_data = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "stream": False,
    }
    result = _proxy.post("/inference/chat/completions", json_data=request_data)
    if result and "choices" in result and len(result["choices"]) > 0:
        message = result["choices"][0].get("message", {})
        return {
            "content": message.get("content", ""),
            "reasoning_content": message.get("reasoning_content", ""),
            "tool_calls": message.get("tool_calls"),
        }
    return {"content": "", "reasoning_content": "", "tool_calls": None}


# ── Evaluator prompt (short, JSON-only) ──────────────────────────────────────

EVALUATOR_PROMPT = """You are a product matching evaluator. Given search results and a user query, pick the BEST matching product(s).

Respond with ONLY a JSON object:
{"action": "recommend", "product_ids": ["id1"], "reason": "brief reason"}
OR if results are clearly wrong category:
{"action": "refine", "new_keywords": ["term1", "term2"], "reason": "why"}

CRITICAL RULES:
- For SINGLE product queries: return EXACTLY ONE product_id in the array
- For MULTI product queries: return one product_id PER requested item, IN ORDER
- Match brand names EXACTLY (case-insensitive). If query says "rrj brand", only pick products with "rrj" in the title/attributes
- Verify price is within range. If query says "above 30", product price must be > 30
- Check attributes (color, size, material) from the detailed info section
- Pick the product whose title and attributes most closely match ALL query terms
- Prefer RECOMMEND over REFINE. Only refine if zero results match the category"""


# ── Helper functions ─────────────────────────────────────────────────────────


def _evaluate_results(query: str, results_summary: str, model: str, task_type: str = "product", num_products: int = 1) -> Optional[dict]:
    """Call LLM to evaluate search results. Returns parsed JSON or None."""
    task_hint = f"\nTask type: {task_type}. Return exactly {num_products} product_id(s)."
    messages = [
        {"role": "system", "content": EVALUATOR_PROMPT},
        {"role": "user", "content": f"Query: {query}{task_hint}\n\nSearch Results:\n{results_summary}"},
    ]
    llm_result = inference(model=model, messages=messages, temperature=0.0)
    content = llm_result.get("content", "")

    # Try to parse JSON from response
    # Handle cases where LLM wraps JSON in markdown code blocks
    json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None  # Caller handles fallback


def _format_results_for_llm(products: list, product_details: list = None) -> str:
    """Format search results into a concise summary for the LLM."""
    lines = []
    for i, p in enumerate(products[:10]):
        pid = p.get("product_id", "?")
        title = p.get("title", "?")[:80]
        price = p.get("price", "?")
        shop = p.get("shop_id", "?")
        lines.append(f"{i+1}. [{pid}] {title} — {price} PHP (shop:{shop})")

    if product_details:
        lines.append("\nDetailed info for top candidates:")
        for d in product_details[:5]:
            pid = d.get("product_id", "?")
            attrs = d.get("attributes", {})
            # Show key attributes concisely
            attr_str = ", ".join(f"{k}={v}" for k, v in list(attrs.items())[:6])
            lines.append(f"  [{pid}] {attr_str}")

    return "\n".join(lines)


def _extract_product_ids(results: list) -> List[str]:
    """Extract product IDs from a list of search result dicts."""
    ids = []
    for p in results:
        pid = str(p.get("product_id", ""))
        if pid:
            ids.append(pid)
    return ids


# ── Task-specific runner functions ───────────────────────────────────────────


def _run_product_task(query: str, parsed: Dict, model: str) -> List[Dict]:
    """
    Handle single-product search tasks.

    Flow:
    1. [CODE] find_product(keywords, price) -> results
    2. [CODE] view_product_information(top 5) -> details
    3. [LLM] evaluate results -> recommend or refine
    4. [CODE] If refine: retry search, view details, LLM evaluate again
    5. [CODE] recommend_product + terminate
    """
    steps = []
    step_num = 0

    keywords = parsed["keywords"]
    price = parsed.get("price")
    service = parsed.get("service")

    # ── Step 1: Search pages 1-3 for broader coverage ──────────────────
    step_num += 1
    search_params = {"q": keywords}
    if price:
        search_params["price"] = price
    if service:
        search_params["service"] = service

    all_results = []
    all_search_results = []
    for page in range(1, 4):
        page_params = dict(search_params)
        page_params["page"] = page
        sr = execute_tool_call("find_product", page_params)
        all_search_results.append(sr)
        page_results = sr["result"]
        if page_results and isinstance(page_results, list):
            all_results.extend(page_results)
        if not page_results:
            break  # No more pages

    results = all_results

    steps.append(create_dialogue_step(
        think=f"Searching for products with keywords: {keywords}, price: {price}, service: {service} (pages 1-3, {len(results)} results)",
        tool_results=all_search_results,
        response="",
        query=query,
        step=step_num,
    ))

    if not results:
        # Try broader search without price/service filters
        step_num += 1
        broader_params = {"q": keywords}
        search_result = execute_tool_call("find_product", broader_params)
        results = search_result["result"] or []
        steps.append(create_dialogue_step(
            think="No results with filters, trying broader search.",
            tool_results=[search_result],
            response="",
            query=query,
            step=step_num,
        ))

    if not results:
        # Last resort: recommend nothing but still call recommend+terminate
        step_num += 1
        rec_result = execute_tool_call("recommend_product", {"product_ids": "0"})
        term_result = execute_tool_call("terminate", {"status": "failure"})
        steps.append(create_dialogue_step(
            think="No products found at all. Recommending placeholder.",
            tool_results=[rec_result, term_result],
            response="Could not find matching products.",
            query=query,
            step=step_num,
        ))
        return steps

    # ── Step 2: View product details for top candidates ──────────────────
    # Deduplicate by product_id and take top 8
    step_num += 1
    seen_ids = set()
    unique_results = []
    for r in results:
        pid = str(r.get("product_id", ""))
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            unique_results.append(r)
    results = unique_results

    top_ids = _extract_product_ids(results[:8])
    view_result = execute_tool_call("view_product_information", {"product_ids": ",".join(top_ids)})
    details = view_result["result"]

    steps.append(create_dialogue_step(
        think=f"Viewing details for top {len(top_ids)} candidates from {len(results)} total: {top_ids}",
        tool_results=[view_result],
        response="",
        query=query,
        step=step_num,
    ))

    # ── Step 3: LLM evaluates results ────────────────────────────────────
    step_num += 1
    results_summary = _format_results_for_llm(results, details if isinstance(details, list) else None)
    evaluation = _evaluate_results(query, results_summary, model, task_type="product", num_products=1)

    if evaluation and evaluation.get("action") == "refine":
        # LLM wants to try different keywords
        new_keywords = evaluation.get("new_keywords", [])
        refine_reason = evaluation.get("reason", "Results did not match query")

        if new_keywords:
            refine_q = " ".join(new_keywords) if isinstance(new_keywords, list) else str(new_keywords)

            steps.append(create_dialogue_step(
                think=f"LLM suggests refining search: {refine_reason}. New keywords: {refine_q}",
                tool_results=[],
                response="",
                query=query,
                step=step_num,
            ))

            # ── Step 4: Retry search with refined keywords ───────────────
            step_num += 1
            retry_params = {"q": refine_q}
            if price:
                retry_params["price"] = price
            retry_result = execute_tool_call("find_product", retry_params)
            retry_results = retry_result["result"]

            steps.append(create_dialogue_step(
                think=f"Retrying search with refined keywords: {refine_q}",
                tool_results=[retry_result],
                response="",
                query=query,
                step=step_num,
            ))

            if retry_results:
                # View details for retry results
                step_num += 1
                retry_ids = _extract_product_ids(retry_results[:5])
                retry_view = execute_tool_call("view_product_information", {"product_ids": ",".join(retry_ids)})
                retry_details = retry_view["result"]

                steps.append(create_dialogue_step(
                    think=f"Viewing details for refined results: {retry_ids}",
                    tool_results=[retry_view],
                    response="",
                    query=query,
                    step=step_num,
                ))

                # ── Step 5: LLM evaluates refined results ────────────────
                step_num += 1
                retry_summary = _format_results_for_llm(
                    retry_results, retry_details if isinstance(retry_details, list) else None
                )
                evaluation = _evaluate_results(query, retry_summary, model, task_type="product", num_products=1)

                # Use retry results if evaluation found something
                if evaluation and evaluation.get("action") == "recommend":
                    rec_ids = evaluation.get("product_ids", [])
                    if rec_ids:
                        product_ids_str = ",".join(str(pid) for pid in rec_ids)
                        rec_result = execute_tool_call("recommend_product", {"product_ids": product_ids_str})
                        term_result = execute_tool_call("terminate", {"status": "success"})
                        steps.append(create_dialogue_step(
                            think=f"LLM recommends from refined results: {rec_ids}. Reason: {evaluation.get('reason', '')}",
                            tool_results=[rec_result, term_result],
                            response=f"Recommended products: {product_ids_str}",
                            query=query,
                            step=step_num,
                        ))
                        return steps

                # Fallback: use first from retry results
                fallback_id = str(retry_results[0].get("product_id", results[0].get("product_id", "0")))
                rec_result = execute_tool_call("recommend_product", {"product_ids": fallback_id})
                term_result = execute_tool_call("terminate", {"status": "success"})
                steps.append(create_dialogue_step(
                    think=f"LLM evaluation unclear after refine. Falling back to best candidate: {fallback_id}",
                    tool_results=[rec_result, term_result],
                    response=f"Recommended product: {fallback_id}",
                    query=query,
                    step=step_num,
                ))
                return steps
        else:
            # No new keywords suggested, fall through to recommend from original results
            steps.append(create_dialogue_step(
                think=f"LLM suggested refine but gave no new keywords. Using best from original results.",
                tool_results=[],
                response="",
                query=query,
                step=step_num,
            ))

    # ── Recommend from evaluation or fallback ────────────────────────────
    step_num += 1
    if evaluation and evaluation.get("action") == "recommend":
        rec_ids = evaluation.get("product_ids", [])
        if rec_ids:
            product_ids_str = ",".join(str(pid) for pid in rec_ids)
            rec_result = execute_tool_call("recommend_product", {"product_ids": product_ids_str})
            term_result = execute_tool_call("terminate", {"status": "success"})
            steps.append(create_dialogue_step(
                think=f"LLM recommends: {rec_ids}. Reason: {evaluation.get('reason', '')}",
                tool_results=[rec_result, term_result],
                response=f"Recommended products: {product_ids_str}",
                query=query,
                step=step_num,
            ))
            return steps

    # Fallback: recommend first result
    fallback_id = str(results[0].get("product_id", "0"))
    rec_result = execute_tool_call("recommend_product", {"product_ids": fallback_id})
    term_result = execute_tool_call("terminate", {"status": "success"})
    steps.append(create_dialogue_step(
        think=f"No clear LLM recommendation. Falling back to first search result: {fallback_id}",
        tool_results=[rec_result, term_result],
        response=f"Recommended product: {fallback_id}",
        query=query,
        step=step_num,
    ))
    return steps


def _run_shop_task(query: str, parsed: Dict, model: str) -> List[Dict]:
    """
    Handle same-shop multi-product search tasks.

    Flow:
    1. [CODE] Build search specs from parsed product_specs
    2. [CODE] find_products_in_same_shop(specs) -> results
    3. [LLM] Evaluate results -> recommend or refine
    4. [CODE] If refine: retry with different keywords
    5. [CODE] recommend_product + terminate
    """
    steps = []
    step_num = 0

    product_specs = parsed.get("product_specs", [])

    # Build specs for find_products_in_same_shop
    shop_specs = []
    for spec in product_specs:
        s = {"q": spec.get("keywords", "")}
        if spec.get("price"):
            s["price"] = spec["price"]
        if spec.get("service"):
            s["service"] = spec["service"]
        shop_specs.append(s)

    # If no product specs were extracted, fall back to keywords
    if not shop_specs:
        shop_specs = [{"q": parsed["keywords"]}]
        if parsed.get("price"):
            shop_specs[0]["price"] = parsed["price"]

    # ── Step 1: Find products in same shop ───────────────────────────────
    step_num += 1
    specs_json = json.dumps(shop_specs)
    shop_result = execute_tool_call("find_products_in_same_shop", {"product_queries": specs_json})
    shop_data = shop_result["result"]

    steps.append(create_dialogue_step(
        think=f"Searching for {len(shop_specs)} products in the same shop: {[s.get('q') for s in shop_specs]}",
        tool_results=[shop_result],
        response="",
        query=query,
        step=step_num,
    ))

    if isinstance(shop_data, dict) and shop_data.get("found"):
        # Found products in same shop
        products = shop_data.get("products", [])
        product_ids = [str(p.get("product_id", "")) for p in products]

        # ── Step 2: View product details ─────────────────────────────────
        step_num += 1
        view_result = execute_tool_call("view_product_information", {"product_ids": ",".join(product_ids)})
        details = view_result["result"]

        steps.append(create_dialogue_step(
            think=f"Found all products in shop {shop_data.get('shop_id')}. Viewing details for: {product_ids}",
            tool_results=[view_result],
            response="",
            query=query,
            step=step_num,
        ))

        # ── Step 3: LLM evaluates ───────────────────────────────────────
        step_num += 1
        results_summary = _format_results_for_llm(products, details if isinstance(details, list) else None)
        num_prods = len(product_ids)
        evaluation = _evaluate_results(query, results_summary, model, task_type="shop", num_products=num_prods)

        if evaluation and evaluation.get("action") == "recommend":
            rec_ids = evaluation.get("product_ids", [])
            if rec_ids:
                product_ids_str = ",".join(str(pid) for pid in rec_ids)
            else:
                # LLM said recommend but no IDs - use the found products in order
                product_ids_str = ",".join(product_ids)
        else:
            # Fallback or refine: just use the found products in order
            product_ids_str = ",".join(product_ids)

        rec_result = execute_tool_call("recommend_product", {"product_ids": product_ids_str})
        term_result = execute_tool_call("terminate", {"status": "success"})
        steps.append(create_dialogue_step(
            think=f"Recommending products from shop: {product_ids_str}",
            tool_results=[rec_result, term_result],
            response=f"Recommended products from same shop: {product_ids_str}",
            query=query,
            step=step_num,
        ))
        return steps

    # ── Shop search failed: try with simplified keywords ─────────────────
    step_num += 1
    # Simplify each spec's keywords (take fewer words)
    simplified_specs = []
    for spec in shop_specs:
        q = spec.get("q", "")
        words = q.split()
        # Use first 2 words only
        simplified_q = " ".join(words[:2]) if len(words) > 2 else q
        s = {"q": simplified_q}
        if spec.get("price"):
            s["price"] = spec["price"]
        simplified_specs.append(s)

    retry_json = json.dumps(simplified_specs)
    retry_result = execute_tool_call("find_products_in_same_shop", {"product_queries": retry_json})
    retry_data = retry_result["result"]

    steps.append(create_dialogue_step(
        think=f"Initial shop search failed. Retrying with simplified keywords: {[s.get('q') for s in simplified_specs]}",
        tool_results=[retry_result],
        response="",
        query=query,
        step=step_num,
    ))

    if isinstance(retry_data, dict) and retry_data.get("found"):
        products = retry_data.get("products", [])
        product_ids = [str(p.get("product_id", "")) for p in products]
        product_ids_str = ",".join(product_ids)

        step_num += 1
        rec_result = execute_tool_call("recommend_product", {"product_ids": product_ids_str})
        term_result = execute_tool_call("terminate", {"status": "success"})
        steps.append(create_dialogue_step(
            think=f"Found products on retry in shop {retry_data.get('shop_id')}: {product_ids_str}",
            tool_results=[rec_result, term_result],
            response=f"Recommended products from same shop: {product_ids_str}",
            query=query,
            step=step_num,
        ))
        return steps

    # ── All shop searches failed: do individual searches and recommend best ──
    step_num += 1
    all_product_ids = []
    all_tool_results = []

    for spec in shop_specs:
        search_params = {"q": spec.get("q", "")}
        if spec.get("price"):
            search_params["price"] = spec["price"]
        if spec.get("service"):
            search_params["service"] = spec["service"]

        sr = execute_tool_call("find_product", search_params)
        all_tool_results.append(sr)
        sr_results = sr["result"]
        if sr_results and isinstance(sr_results, list) and len(sr_results) > 0:
            all_product_ids.append(str(sr_results[0].get("product_id", "0")))
        else:
            all_product_ids.append("0")

    if all_product_ids:
        product_ids_str = ",".join(all_product_ids)
    else:
        product_ids_str = "0"

    rec_result = execute_tool_call("recommend_product", {"product_ids": product_ids_str})
    term_result = execute_tool_call("terminate", {"status": "success"})
    all_tool_results.extend([rec_result, term_result])

    steps.append(create_dialogue_step(
        think=f"Could not find all products in same shop. Falling back to individual best matches: {product_ids_str}",
        tool_results=all_tool_results,
        response=f"Recommended products: {product_ids_str}",
        query=query,
        step=step_num,
    ))
    return steps


def _run_voucher_task(query: str, parsed: Dict, model: str) -> List[Dict]:
    """
    Handle voucher/budget multi-product tasks.

    Flow:
    1. [CODE] Build search specs from parsed product_specs
    2. [CODE] find_products_in_same_shop(specs) -> results
    3. [CODE] calculate_voucher with found prices
    4. [LLM] Evaluate results + voucher calculation
    5. [CODE] recommend_product + terminate
    """
    steps = []
    step_num = 0

    product_specs = parsed.get("product_specs", [])
    voucher_params = parsed.get("voucher_params", {}) or {}

    # Build specs for find_products_in_same_shop
    shop_specs = []
    for spec in product_specs:
        s = {"q": spec.get("keywords", "")}
        if spec.get("price"):
            s["price"] = spec["price"]
        if spec.get("service"):
            s["service"] = spec["service"]
        shop_specs.append(s)

    # If no product specs, fall back to keywords
    if not shop_specs:
        shop_specs = [{"q": parsed["keywords"]}]
        if parsed.get("price"):
            shop_specs[0]["price"] = parsed["price"]

    # ── Step 1: Find products in same shop ───────────────────────────────
    step_num += 1
    specs_json = json.dumps(shop_specs)
    shop_result = execute_tool_call("find_products_in_same_shop", {"product_queries": specs_json})
    shop_data = shop_result["result"]

    steps.append(create_dialogue_step(
        think=f"Searching for {len(shop_specs)} products in the same shop for voucher task: {[s.get('q') for s in shop_specs]}",
        tool_results=[shop_result],
        response="",
        query=query,
        step=step_num,
    ))

    found_products = None
    product_ids = []

    if isinstance(shop_data, dict) and shop_data.get("found"):
        found_products = shop_data.get("products", [])
        product_ids = [str(p.get("product_id", "")) for p in found_products]
    else:
        # Retry with simplified keywords
        step_num += 1
        simplified_specs = []
        for spec in shop_specs:
            q = spec.get("q", "")
            words = q.split()
            simplified_q = " ".join(words[:2]) if len(words) > 2 else q
            s = {"q": simplified_q}
            if spec.get("price"):
                s["price"] = spec["price"]
            simplified_specs.append(s)

        retry_json = json.dumps(simplified_specs)
        retry_result = execute_tool_call("find_products_in_same_shop", {"product_queries": retry_json})
        retry_data = retry_result["result"]

        steps.append(create_dialogue_step(
            think=f"Initial search failed. Retrying with simplified keywords: {[s.get('q') for s in simplified_specs]}",
            tool_results=[retry_result],
            response="",
            query=query,
            step=step_num,
        ))

        if isinstance(retry_data, dict) and retry_data.get("found"):
            found_products = retry_data.get("products", [])
            product_ids = [str(p.get("product_id", "")) for p in found_products]
        else:
            # Fallback: individual searches
            step_num += 1
            all_tool_results = []
            for spec in shop_specs:
                search_params = {"q": spec.get("q", "")}
                if spec.get("price"):
                    search_params["price"] = spec["price"]
                sr = execute_tool_call("find_product", search_params)
                all_tool_results.append(sr)
                sr_results = sr["result"]
                if sr_results and isinstance(sr_results, list) and len(sr_results) > 0:
                    product_ids.append(str(sr_results[0].get("product_id", "0")))
                    if found_products is None:
                        found_products = []
                    found_products.append(sr_results[0])
                else:
                    product_ids.append("0")

            steps.append(create_dialogue_step(
                think=f"Could not find all products in same shop. Using individual best matches.",
                tool_results=all_tool_results,
                response="",
                query=query,
                step=step_num,
            ))

    # ── Step 2: Calculate voucher if we have voucher params and products ──
    if voucher_params and found_products:
        step_num += 1
        # Extract prices from found products
        prices = []
        for p in found_products:
            price_val = p.get("price", 0)
            try:
                prices.append(float(price_val))
            except (ValueError, TypeError):
                prices.append(0.0)

        voucher_call_params = {
            "product_prices": ",".join(str(p) for p in prices),
            "voucher_type": voucher_params.get("voucher_type", "fixed"),
            "discount_value": voucher_params.get("discount_value", 0),
            "threshold": voucher_params.get("threshold", 0),
            "budget": voucher_params.get("budget", 0),
        }
        if voucher_params.get("cap"):
            voucher_call_params["cap"] = voucher_params["cap"]

        voucher_result = execute_tool_call("calculate_voucher", voucher_call_params)

        steps.append(create_dialogue_step(
            think=f"Calculating voucher for prices {prices} with params: {voucher_params}",
            tool_results=[voucher_result],
            response="",
            query=query,
            step=step_num,
        ))

    # ── Step 3: View product details ─────────────────────────────────────
    if product_ids and product_ids[0] != "0":
        step_num += 1
        valid_ids = [pid for pid in product_ids if pid != "0"]
        view_result = execute_tool_call("view_product_information", {"product_ids": ",".join(valid_ids)})
        details = view_result["result"]

        steps.append(create_dialogue_step(
            think=f"Viewing details for found products: {valid_ids}",
            tool_results=[view_result],
            response="",
            query=query,
            step=step_num,
        ))

        # ── Step 4: LLM evaluates ───────────────────────────────────────
        step_num += 1
        products_for_summary = found_products if found_products else []
        results_summary = _format_results_for_llm(
            products_for_summary, details if isinstance(details, list) else None
        )
        num_prods = len(product_ids)
        evaluation = _evaluate_results(query, results_summary, model, task_type="voucher", num_products=num_prods)

        if evaluation and evaluation.get("action") == "recommend":
            rec_ids = evaluation.get("product_ids", [])
            if rec_ids:
                product_ids_str = ",".join(str(pid) for pid in rec_ids)
            else:
                product_ids_str = ",".join(product_ids)
        else:
            product_ids_str = ",".join(product_ids)

        rec_result = execute_tool_call("recommend_product", {"product_ids": product_ids_str})
        term_result = execute_tool_call("terminate", {"status": "success"})
        steps.append(create_dialogue_step(
            think=f"Recommending products for voucher task: {product_ids_str}",
            tool_results=[rec_result, term_result],
            response=f"Recommended products: {product_ids_str}",
            query=query,
            step=step_num,
        ))
        return steps

    # ── Fallback: recommend whatever we have ─────────────────────────────
    step_num += 1
    product_ids_str = ",".join(product_ids) if product_ids else "0"
    rec_result = execute_tool_call("recommend_product", {"product_ids": product_ids_str})
    term_result = execute_tool_call("terminate", {"status": "success"})
    steps.append(create_dialogue_step(
        think=f"Fallback recommendation for voucher task: {product_ids_str}",
        tool_results=[rec_result, term_result],
        response=f"Recommended products: {product_ids_str}",
        query=query,
        step=step_num,
    ))
    return steps


# ── Main agent entry point ───────────────────────────────────────────────────


def agent_main(problem_data: Dict) -> List[Dict]:
    """
    Hybrid code+LLM agent entry point.

    Uses Python code for mechanical work (parsing, searching, recommending)
    and LLM only for evaluating search results (2-3 calls instead of 9+).

    Args:
        problem_data: Dictionary with 'query' key (reward is NOT included)

    Returns:
        List of dialogue steps in the format expected by the evaluation framework.
    """
    query = problem_data.get("query", "")
    model = "Qwen/Qwen3-32B-TEE"

    logger.info(f"[Hybrid] Processing query: {query}")

    parsed = parse_query(query)
    task = parsed["task"]

    logger.info(f"[Hybrid] Task: {task}, Keywords: {parsed['keywords']}, Price: {parsed.get('price')}")

    if task == "product":
        steps = _run_product_task(query, parsed, model)
    elif task == "shop":
        steps = _run_shop_task(query, parsed, model)
    else:
        steps = _run_voucher_task(query, parsed, model)

    logger.info(f"[Hybrid] Completed with {len(steps)} steps")
    return steps
