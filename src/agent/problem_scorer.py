"""ProblemScorer - Score individual problems independently.

This module provides the ProblemScorer class which scores shopping benchmark
problems one at a time, enabling partial results and real-time progress reporting.

Uses HTTP calls to the search-server for product lookups, eliminating the need
for local Java/Pyserini installation.
"""

import os
import logging

from collections import defaultdict
from typing import Optional

import requests

from src.agent.rewards.orm import batch_encode_titles, ground_truth_reward, rule_score_reward, length_reward
from src.agent.rewards.prm import format_reward
from src.agent.util.message import Message, OUTPUT_ROLES


# Search server URL - configurable via environment variable
# Local dev: http://localhost:5632
# Docker/sandbox: http://search-server:5632
SEARCH_SERVER_URL = os.getenv("SEARCH_SERVER_URL", "http://localhost:5632")

# Cache for product lookups to avoid repeated HTTP calls
_product_cache: dict[str, Optional[dict]] = {}


def clear_product_cache() -> None:
    """Clear the product cache between evaluation runs to prevent unbounded growth."""
    _product_cache.clear()


def get_product(product_id: str) -> Optional[dict]:
    """Fetch full product document from search-server.

    Args:
        product_id: Product ID to look up

    Returns:
        Full product dict or None if not found
    """
    if not product_id:
        return None

    # Check cache first
    if product_id in _product_cache:
        return _product_cache[product_id]

    try:
        resp = requests.get(
            f"{SEARCH_SERVER_URL}/get_product_raw",
            params={"product_ids": product_id},
            timeout=10,
        )
        if resp.ok:
            products = resp.json()
            if products and len(products) > 0:
                _product_cache[product_id] = products[0]
                return products[0]
        _product_cache[product_id] = None
        return None
    except requests.RequestException as e:
        logging.warning(f"Failed to fetch product {product_id}: {e}")
        return None


FIELDS = ["title", "price", "service", "sku & attrs"]

VALID_TASKS = ["product", "shop", "voucher"]

# Multiplier applied to per-step format score when extra_info.timestamp is missing.
# 1.0 = no penalty, 0.0 = full penalty. Tunable at launch.
TIMESTAMP_MISSING_PENALTY = 0.5


class ProblemScorer:
    """Scores individual shopping benchmark problems independently.

    This class enables per-problem scoring without requiring the entire test
    suite to complete. It supports product, shop, and voucher tasks.

    Attributes:
        task: Task type ("product", "shop", or "voucher")
        rewards: Dictionary mapping queries to ground truth rewards
        vouchers: Dictionary mapping queries to voucher constraints
    """

    def __init__(self, task: str, rewards: dict, vouchers: dict):
        """Initialize the problem scorer.

        Args:
            task: Task type ("product", "shop", or "voucher")
            rewards: Dictionary mapping queries to ground truth rewards
            vouchers: Dictionary mapping queries to voucher constraints

        Raises:
            ValueError: If task is not a valid task type
        """
        if task not in VALID_TASKS:
            raise ValueError(f"Invalid task: {task}. Must be one of {VALID_TASKS}")

        self.task = task
        self.rewards = rewards
        self.vouchers = vouchers

    def score_problem(
        self,
        query: str,
        output: list[dict],
        model: str = "default",
        mode: str = "think",
    ) -> Optional[dict]:
        """Score a single problem independently.

        Args:
            query: The problem query/identifier
            output: The rollout output for this problem
            model: Model name (used to determine if format scoring should be skipped)
            mode: Reasoning mode ("think" or "no think")

        Returns:
            Dictionary containing scores for this problem, or None if reward not found:
            - length: Length reward score
            - format: Format reward score
            - gt: Ground truth match score
            - rule: Rule-based score
            - product: Product found score
            - Additional task-specific scores (shop, budget, field scores)
        """
        # Check if we have reward data for this query
        if query not in self.rewards:
            return None

        reward = self.rewards[query]
        voucher = self.vouchers.get(query)

        score = defaultdict(float)

        # Length score
        length_score = length_reward(output)
        score["length"] = length_score

        # Format score (includes timestamp presence penalty)
        format_score = 0
        if model != "human":
            for step in output:
                try:
                    message = Message.from_dict(step["completion"]["message"])
                    completion = message.to_string(OUTPUT_ROLES)
                    step_format = (
                        format_reward(completion)
                        if mode == "think"
                        else format_reward(completion, ["tool_call"])
                    )
                    # Penalize steps missing a valid timestamp in extra_info
                    ts = step.get("extra_info", {}).get("timestamp")
                    if not isinstance(ts, (int, float)) or ts <= 0:
                        step_format *= TIMESTAMP_MISSING_PENALTY
                    format_score += step_format
                except (KeyError, TypeError, AttributeError) as e:
                    logging.warning(
                        "Malformed output step during format scoring: %s", e
                    )
                    continue
        format_score = format_score / len(output) if output else 0
        score["format"] = format_score

        # Task-specific evaluation
        if self.task == "product":
            self._eval_product(score, output, reward)
        elif self.task == "shop":
            self._eval_shop(score, output, reward)
        elif self.task == "voucher":
            self._eval_voucher(score, output, reward, voucher)

        return dict(score)

    def _extract_recommended_product(self, output: list[dict]) -> list[str]:
        """Extract deduplicated recommended product IDs from output.

        Returns:
            List of unique, whitespace-stripped product IDs (order preserved)
        """
        product_ids = ""
        if not output:
            return []

        for step in output:
            try:
                message = step["completion"]["message"]
                if message and "tool_call" in message and message["tool_call"]:
                    for command in message["tool_call"]:
                        if command["name"] == "recommend_product":
                            product_ids = command["parameters"].get("product_ids", "")
            except (KeyError, TypeError, AttributeError) as e:
                logging.warning(
                    "Malformed output step during product extraction: %s", e
                )
                continue

        if not isinstance(product_ids, str):
            return []

        # Strip whitespace and deduplicate while preserving order
        seen = set()
        result = []
        for pid in product_ids.split(","):
            pid = pid.strip()
            if pid and pid not in seen:
                seen.add(pid)
                result.append(pid)
        return result

    def _set_eval_score(self, product: dict, score: dict, reward: dict, product_title_emb=None) -> float:
        """Update score dict with product evaluation metrics.

        Args:
            product: Product data from search index
            score: Score dictionary to update
            reward: Ground truth reward data
            product_title_emb: Pre-encoded product title embedding (optional)

        Returns:
            The rule score for this product (0.0 to 1.0)
        """
        score["product"] += 1

        score["gt"] += ground_truth_reward(product, reward)

        rule_score, total_counter, hit_counter = rule_score_reward(product, reward, product_title_emb=product_title_emb)
        score["rule"] += rule_score

        for field in FIELDS:
            score[field] += (
                hit_counter.get(field, 0) / total_counter.get(field, 0)
                if total_counter.get(field, 0) > 0
                else 0
            )

        return rule_score

    def _eval_product(self, score: dict, output: list[dict], reward: dict):
        """Evaluate product task (single product recommendation).

        Args:
            score: Score dictionary to update
            output: Rollout output
            reward: Ground truth reward
        """
        product_id_list = self._extract_recommended_product(output)
        if not product_id_list:
            return
        product_id = product_id_list[0]

        product = get_product(product_id)
        if not product:
            return

        self._set_eval_score(product, score, reward)

    def _fetch_and_encode(self, output, reward):
        """Fetch products and batch encode non-GT titles for multi-product tasks.

        Returns (products, emb_map) where emb_map maps index → pre-encoded title embedding.
        """
        product_id_list = self._extract_recommended_product(output)
        products = [
            get_product(product_id_list[i]) if i < len(product_id_list) else None
            for i in range(len(reward))
        ]

        non_gt_indices = []
        non_gt_titles = []
        for i, (product, sub_reward) in enumerate(zip(products, reward)):
            if product is not None and ground_truth_reward(product, sub_reward) != 1:
                non_gt_indices.append(i)
                non_gt_titles.append(product["title"])

        embeddings = batch_encode_titles(non_gt_titles)
        emb_map = dict(zip(non_gt_indices, embeddings))
        return products, emb_map

    def _score_multi_product(self, score, products, reward, emb_map):
        """Score multiple products, returns (num_hits, shop_ids, total_price)."""
        num_hits = 0
        total_price = 0.0
        shop_ids = set()
        for i, sub_reward in enumerate(reward):
            product = products[i] if i < len(products) else None
            if not product:
                continue
            rule_score = self._set_eval_score(
                product, score, sub_reward, product_title_emb=emb_map.get(i)
            )
            if rule_score > 0:
                num_hits += 1
                total_price += product["price"]
                shop_ids.add(product["shop_id"])
        return num_hits, shop_ids, total_price

    def _normalize_multi_product_score(self, score, reward):
        """Normalize accumulated scores by number of reward items."""
        if len(reward) > 0:
            score["product"] /= len(reward)
            score["gt"] /= len(reward)
            score["rule"] /= len(reward)
            for field in FIELDS:
                score[field] /= len(reward)

    def _eval_shop(self, score: dict, output: list[dict], reward: list[dict]):
        """Evaluate shop task (multiple products from same shop)."""
        products, emb_map = self._fetch_and_encode(output, reward)
        num_hits, shop_ids, _ = self._score_multi_product(score, products, reward, emb_map)
        self._normalize_multi_product_score(score, reward)
        score["shop"] = 1 if num_hits == len(reward) and len(shop_ids) == 1 else 0

    def _eval_voucher(
        self, score: dict, output: list[dict], reward: list[dict], voucher: dict
    ):
        """Evaluate voucher task (budget constraint)."""
        products, emb_map = self._fetch_and_encode(output, reward)
        num_hits, shop_ids, total_price = self._score_multi_product(
            score, products, reward, emb_map
        )

        budget_match = 0
        if num_hits == len(reward):
            if total_price <= voucher["budget"]:
                budget_match = 1
            elif voucher["voucher_type"] == "platform" or (
                voucher["voucher_type"] == "shop" and len(shop_ids) == 1
            ):
                if total_price >= voucher["threshold"]:
                    if voucher["discount_type"] == "fixed":
                        total_price_after_discount = total_price - voucher["face_value"]
                    elif voucher["discount_type"] == "percentage":
                        total_price_after_discount = max(
                            total_price * (1 - voucher["discount"]),
                            total_price - voucher["cap"],
                        )
                    else:
                        raise Exception(
                            f"Invalid voucher discount type: {voucher['discount_type']}"
                        )
                    budget_match = (
                        1 if total_price_after_discount <= voucher["budget"] else 0
                    )

        self._normalize_multi_product_score(score, reward)
        score["budget"] = budget_match
