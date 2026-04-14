from collections import Counter

_sentence_model = None
_sentence_model_unavailable = False


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

            _sentence_model = SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B",
                model_kwargs={"torch_dtype": "float32"},
            )
        except ImportError:
            _sentence_model_unavailable = True
            return None
    return _sentence_model


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
                    if title in precomputed:
                        gt_emb = [precomputed[title]]
                    elif model is not None:
                        gt_emb = model.encode([title])
                    else:
                        continue
                    sim = model.similarity(product_emb, gt_emb)[0][0]
                    total_count += 1
                    total_counter["title"] += 1
                    if sim >= 0.7:
                        hit_count += 1
                        hit_counter["title"] += 1
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
    # flat sku options
    sku_flattens = [set()]
    if "sku_options" in product and product["sku_options"]:
        for option in product["sku_options"].values():
            flatten = set()
            for k, v in option.items():
                flatten.add((k, v))
            sku_flattens.append(flatten)
    # flat attributes
    attr_flatten = set()
    if "attributes" in product and product["attributes"]:
        for k, vs in product["attributes"].items():
            for v in vs:
                attr_flatten.add((k, v))
    # sku options & attributes
    max_total = 0
    max_hit = 0
    for sku_flatten in sku_flattens:
        cur_total = 0
        cur_hit = 0
        if "sku_options" in reward:
            for option in reward["sku_options"]:
                for k, v in option.items():
                    cur_total += 1
                    if (k, v) in sku_flatten or (k, v) in attr_flatten:
                        cur_hit += 1
        if "attributes" in reward:
            for attr in reward["attributes"]:
                for k, vs in attr.items():
                    for v in vs:
                        cur_total += 1
                        if (k, v) in sku_flatten or (k, v) in attr_flatten:
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


def web_rule_score_reward(product: dict, reward: dict) -> tuple[float, float]:
    if ground_truth_reward(product, reward) == 1:
        return 1, 1

    kw = reward["key_attribute"]
    title_score = 1 if kw.lower() in product["title"].lower() else 0
    kw_score = 1 if kw.lower() in product["description"].lower() else 0
    kw_score = max(title_score, kw_score)

    # title
    model = _get_sentence_model()
    if "title" in reward and model is not None:
        title = reward["title"]
        sentences = [product["title"], title]
        embeddings = model.encode(sentences)
        similarities = model.similarity(embeddings, embeddings)
        sim = similarities[0][1]
        title_score = 1 if sim >= 0.8 else 0

    return kw_score, title_score


def web_response_score_reward(response: str, kw: str) -> float:
    score = 1 if kw.lower() in response.lower() else 0
    return score


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
