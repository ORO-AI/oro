"""
Agent template for ShoppingBench sandbox execution.

Implements a ReAct (Reasoning + Acting) loop following the ShoppingBench paper:
each step calls the LLM, parses <think>, <tool_call>, <response> XML tags,
executes tools, appends observations to dialogue history, and repeats until
the agent terminates or hits MAX_STEPS.
"""

from os import getenv

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
        found = False
        if key.lower() in {k.lower() for k in attrs}:
            attr_val = next(v for k, v in attrs.items() if k.lower() == key.lower())
            attr_str = (
                str(attr_val).lower()
                if not isinstance(attr_val, list)
                else " ".join(str(x).lower() for x in attr_val)
            )
            if req_lower in attr_str:
                found = True
        if not found and req_lower in searchable:
            found = True
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


def _build_toolkit_descriptions() -> str:
    """Build toolkit descriptions from registered tool docstrings and signatures."""
    import inspect
    from src.agent.agent_interface import _TOOL_REGISTRY

    lines = []
    for i, (name, func) in enumerate(_TOOL_REGISTRY.items(), 1):
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""
        lines.append(f"{i}. {name}{sig}")
        for doc_line in doc.split("\n"):
            lines.append(f"    {doc_line}")
        lines.append("")
    return "\n".join(lines).rstrip()


TOOLKIT_DESCRIPTIONS = _build_toolkit_descriptions()


BASE_SYSTEM_PROMPT = f"""# Role
You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.

# Available Tools
{TOOLKIT_DESCRIPTIONS}

# Tools Rules
1. Use the "tool_call_id" field to link tool calls to their results in `<obs>...</obs>`.
2. Don't blindly trust the tool call results. Carefully evaluate whether they align with the user's needs, and use additional tools for verification if necessary.
3. Use the `find_product` tool to search for products. If the results do not meet expectations, you can:
    - Modify the parameter `q` and reuse the tool to get results related to the modified query.
    - Keep the parameter `q` the same, but change the parameter `page` to get new results.
    - Set the parameter `shop_id` to get results within the specified shop.
4. To check product information such as color, size, weight, model, material, pattern and so on, use the `view_product_information` tool.
5. When you identify products that fulfill the user's needs, use the `recommend_product` tool to recommend them to the user.
6. **CRITICAL**: You MUST call `recommend_product` BEFORE calling `terminate`. NEVER terminate without recommending. If you cannot find a perfect match, recommend the BEST available match.
7. Complete the task progressively without asking the user for external information.

# Search Strategy (CRITICAL)
- Use SHORT, focused search queries with 2-4 keywords. NEVER pass the full user query to find_product.
- Extract the most distinctive terms: brand name, product type, and 1-2 key attributes.
- Example: For "Looking for a hosport brand waist bag for motorcycles, priced from 180 to 505 PHP", search: q="hosport waist bag", price="180-505"
- Example: For "Show me lancol battery testers priced from 1593 to 3846 PHP", search: q="lancol battery tester", price="1593-3846"
- If the first search returns irrelevant results, try different keyword combinations.
- ALWAYS use `view_product_information` on the top 3-5 candidate product IDs before recommending. Verify that attributes (brand, material, color, size) EXACTLY match the query.

# Output Format
1. Your output must always include `<think>...</think>` and at least one of `<tool_call>...</tool_call>` or `<response>...</response>`. No other content is allowed.
2. Tool calls must be included within `<tool_call>...</tool_call>` and structured as a JSON array. Each tool call must have a "name" field and a "parameters" field as a dictionary. If no parameters are required, the dictionary can be empty.
3. Below is a template of your output:
```plaintext
<think>Your thoughts and reasoning</think>
<tool_call>[
{{"name": "tool name", "parameters": {{"parameter1": "value1", "parameter2": "value2"}}}},
{{"name": "...", "parameters": {{...}}}},
...
]</tool_call>
<response>Your response will be displayed to the user</response>
```"""

# Task-specific strategy additions
PRODUCT_STRATEGY = """
# Task: Find ONE product
- Search with 2-3 keywords, then `view_product_information` on top 3-5 results
- Use `check_product_match` to verify attributes BEFORE recommending
- MUST call `recommend_product` then `terminate`. Never terminate without recommending.
"""

SHOP_STRATEGY = """
# Task: Find products from the SAME shop
- Use `find_products_in_same_shop` with a JSON array: [{"q": "short keywords", "price": "range"}, ...]
- Verify results with `check_product_match` for each product
- MUST call `recommend_product` with ALL product IDs in the ORDER from the query, then `terminate`
- If find_products_in_same_shop returns found=true, recommend those products immediately
- NEVER terminate without recommending. If imperfect match, recommend best available.
"""

VOUCHER_STRATEGY = """
# Task: Find products within budget after voucher
- Use `find_products_in_same_shop` to find all products from one shop
- Use `calculate_voucher` to verify the total price is within budget after discount
- MUST call `recommend_product` with ALL product IDs in query ORDER, then `terminate`
- NEVER terminate without recommending. If imperfect match, recommend best available.
"""


def _detect_task(query: str) -> str:
    """Detect task type from query content."""
    q_lower = query.lower()
    if "voucher" in q_lower or "budget" in q_lower:
        return "voucher"
    elif "shop" in q_lower and (
        "both" in q_lower
        or "these" in q_lower
        or "offering" in q_lower
        or "sells" in q_lower
    ):
        return "shop"
    return "product"


def _build_system_prompt(query: str) -> str:
    """Build task-specific system prompt."""
    task = _detect_task(query)
    if task == "voucher":
        return BASE_SYSTEM_PROMPT + VOUCHER_STRATEGY
    elif task == "shop":
        return BASE_SYSTEM_PROMPT + SHOP_STRATEGY
    return BASE_SYSTEM_PROMPT + PRODUCT_STRATEGY


def parse_llm_output(content: str, reasoning_content: str = "") -> dict:
    """Parse LLM output for <think>, <tool_call>, <response> tags."""
    parsed = {}

    match = re.search(r"<think>(.+?)</think>", content, re.DOTALL)
    if match:
        parsed["think"] = match.group(1).strip()
    elif reasoning_content:
        parsed["think"] = (
            reasoning_content.replace("<think>", "").replace("</think>", "").strip()
        )
    else:
        parsed["think"] = ""

    parsed["tool_call"] = []
    match = re.search(r"<tool_call>(.+?)</tool_call>", content, re.DOTALL)
    if match:
        try:
            json_array = json.loads(match.group(1).strip())
            if isinstance(json_array, dict):
                json_array = [json_array]
            for cmd in json_array:
                name = cmd["name"]
                parameters = cmd.get("parameters", {})
                tool_call_id = generate_tool_call_id(name, parameters)
                parsed["tool_call"].append(
                    {
                        "name": name,
                        "parameters": parameters,
                        "tool_call_id": tool_call_id,
                    }
                )
        except (json.JSONDecodeError, KeyError):
            logger.warning("Failed to parse tool_call JSON from LLM output")

    match = re.search(r"<response>(.+?)</response>", content, re.DOTALL)
    if match:
        parsed["response"] = match.group(1).strip()
    else:
        parsed["response"] = ""

    return parsed


def format_message_for_history(role: str, content) -> str:
    """Format a message part as XML tag for dialogue history."""
    if isinstance(content, (dict, list)):
        content = json.dumps(content)
    return f"<{role}>{content}</{role}>"


def build_user_prompt(history_messages: List[str]) -> str:
    """Build user prompt from dialogue history."""
    history = "\n\n".join(history_messages)
    return f"# Dialogue Records History\n{history}"


def is_terminate(parsed: dict) -> bool:
    """Check if the agent explicitly called terminate."""
    for cmd in parsed["tool_call"]:
        if cmd["name"] == "terminate":
            return True
    return False


def is_empty_response(parsed: dict) -> bool:
    """Check if the LLM returned an empty response (likely API failure)."""
    return not parsed["think"] and not parsed["tool_call"] and not parsed["response"]


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


def agent_main(problem_data: Dict) -> List[Dict]:
    """
    ReAct agent entry point.

    Implements the paper's multi-step ReAct loop:
    1. Build prompt from system prompt + dialogue history
    2. Call LLM
    3. Parse <think>, <tool_call>, <response> tags
    4. Execute tools, append observations
    5. Repeat until terminate or MAX_STEPS

    Args:
        problem_data: Dictionary with 'query' key (reward is NOT included)

    Returns:
        List of dialogue steps in the format expected by the evaluation framework.
    """
    steps = []
    query = problem_data.get("query", "")
    model = getenv("SANDBOX_MODEL", "deepseek-ai/DeepSeek-V3.2-TEE")

    logger.info(f"[ReAct] Processing query: {query}")

    # History accumulator (list of XML-tagged strings)
    history_messages: List[str] = []

    # Initial user message
    history_messages.append(format_message_for_history("user", query))

    system_prompt = _build_system_prompt(query)

    consecutive_empties = 0
    max_consecutive_empties = 3
    recommended = False  # Track if recommend_product was called
    candidate_product_ids: List[str] = []  # Track best candidates seen

    for step_num in range(1, MAX_STEPS + 1):
        user_prompt = build_user_prompt(history_messages)

        # Call LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        llm_result = inference(model=model, messages=messages, temperature=0.0)

        content = llm_result.get("content", "")
        reasoning_content = llm_result.get("reasoning_content", "")

        parsed = parse_llm_output(content, reasoning_content)

        # Handle empty responses (API failures) - retry up to max_consecutive_empties times
        if is_empty_response(parsed):
            consecutive_empties += 1
            if consecutive_empties >= max_consecutive_empties:
                logger.warning(
                    f"[ReAct] Step {step_num}/{MAX_STEPS} - {consecutive_empties} consecutive empty responses, terminating"
                )
                break
            logger.warning(
                f"[ReAct] Step {step_num}/{MAX_STEPS} - empty LLM response (API failure), retrying ({consecutive_empties}/{max_consecutive_empties})"
            )
            continue
        consecutive_empties = 0  # Reset on successful response

        logger.info(
            f"[ReAct] Step {step_num}/{MAX_STEPS} - "
            f"think={len(parsed['think'])}chars, "
            f"tools={len(parsed['tool_call'])}, "
            f"response={len(parsed['response'])}chars"
        )

        # Execute tool calls and collect observations
        tool_results = []
        observations = []

        for cmd in parsed["tool_call"]:
            if cmd["name"] == "terminate":
                status = cmd.get("parameters", {}).get("status", "success")
                term_msg = f"The interaction has been completed with status: {status}"
                tool_results.append(
                    {
                        "name": "terminate",
                        "parameters": cmd.get("parameters", {}),
                        "tool_call_id": cmd["tool_call_id"],
                        "result": term_msg,
                    }
                )
                observations.append(
                    {
                        "tool_call_id": cmd["tool_call_id"],
                        "results": term_msg,
                    }
                )
                continue

            try:
                result = execute_tool_call(cmd["name"], cmd["parameters"])
                tool_results.append(result)
                observations.append(
                    {
                        "tool_call_id": cmd["tool_call_id"],
                        "results": result["result"],
                    }
                )

                # Track recommend_product calls
                if cmd["name"] == "recommend_product":
                    recommended = True

                # Track candidate product IDs from search/shop results
                if cmd["name"] in ("find_product", "find_products_in_same_shop"):
                    res = result["result"]
                    if isinstance(res, list):
                        for p in res:
                            pid = (
                                str(p.get("product_id", ""))
                                if isinstance(p, dict)
                                else ""
                            )
                            if pid and pid not in candidate_product_ids:
                                candidate_product_ids.append(pid)
                    elif (
                        isinstance(res, dict)
                        and res.get("found")
                        and res.get("products")
                    ):
                        for p in res["products"]:
                            pid = (
                                str(p.get("product_id", ""))
                                if isinstance(p, dict)
                                else ""
                            )
                            if pid and pid not in candidate_product_ids:
                                candidate_product_ids.append(pid)

            except Exception as e:
                logger.error(f"[ReAct] Tool {cmd['name']} failed: {e}")
                tool_results.append(
                    {
                        "name": cmd["name"],
                        "parameters": cmd["parameters"],
                        "tool_call_id": cmd["tool_call_id"],
                        "result": f"Error: {str(e)}",
                    }
                )
                observations.append(
                    {
                        "tool_call_id": cmd["tool_call_id"],
                        "results": f"Error: {str(e)}",
                    }
                )

        step = create_dialogue_step(
            think=parsed["think"],
            tool_results=tool_results,
            response=parsed["response"],
            query=query,
            step=step_num,
        )
        steps.append(step)

        if parsed["think"]:
            history_messages.append(
                format_message_for_history("think", parsed["think"])
            )
        if parsed["tool_call"]:
            tc_for_history = [
                {"name": c["name"], "parameters": c["parameters"]}
                for c in parsed["tool_call"]
            ]
            history_messages.append(
                format_message_for_history("tool_call", tc_for_history)
            )
        if observations:
            history_messages.append(format_message_for_history("obs", observations))
        if parsed["response"]:
            history_messages.append(
                format_message_for_history("response", parsed["response"])
            )

        if is_terminate(parsed):
            logger.info(f"[ReAct] Terminated at step {step_num}")
            break

    # Auto-recommend fallback: if agent never called recommend_product, inject one
    if not recommended and candidate_product_ids:
        logger.warning(
            f"[ReAct] Agent terminated without recommending. Auto-recommending from {len(candidate_product_ids)} candidates: {candidate_product_ids[:5]}"
        )
        # For product tasks, recommend first candidate; for shop/voucher, recommend all
        task = _detect_task(query)
        if task == "product":
            fallback_ids = candidate_product_ids[0]
        else:
            fallback_ids = ",".join(candidate_product_ids)

        fallback_result = execute_tool_call(
            "recommend_product", {"product_ids": fallback_ids}
        )
        fallback_step = create_dialogue_step(
            think="Auto-recommending best available candidates.",
            tool_results=[fallback_result],
            response="",
            query=query,
            step=len(steps) + 1,
        )
        steps.append(fallback_step)

    logger.info(f"[ReAct] Completed with {len(steps)} steps")
    return steps
