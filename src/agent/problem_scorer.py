"""ProblemScorer - Score individual problems independently.

This module provides the ProblemScorer class which scores shopping benchmark
problems one at a time, enabling partial results and real-time progress reporting.

Uses HTTP calls to the search-server for product lookups, eliminating the need
for local Java/Pyserini installation.
"""

import os
import logging

try:
    import ujson as json
except ImportError:
    import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import requests

from rewards.orm import ground_truth_reward, rule_score_reward, length_reward
from rewards.prm import format_reward
from util.message import Message, OUTPUT_ROLES


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

    def _set_eval_score(self, product: dict, score: dict, reward: dict) -> float:
        """Update score dict with product evaluation metrics.

        Args:
            product: Product data from search index
            score: Score dictionary to update
            reward: Ground truth reward data

        Returns:
            The rule score for this product (0.0 to 1.0)
        """
        score["product"] += 1

        score["gt"] += ground_truth_reward(product, reward)

        rule_score, total_counter, hit_counter = rule_score_reward(product, reward)
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

    def _eval_shop(self, score: dict, output: list[dict], reward: list[dict]):
        """Evaluate shop task (multiple products from same shop).

        Args:
            score: Score dictionary to update
            output: Rollout output
            reward: List of ground truth rewards (one per product)
        """
        num_hits = 0
        shop_ids = set()
        product_id_list = self._extract_recommended_product(output)

        for i, sub_reward in enumerate(reward):
            if i >= len(product_id_list):
                continue
            product_id = product_id_list[i]

            product = get_product(product_id)
            if not product:
                continue

            rule_score = self._set_eval_score(product, score, sub_reward)
            if rule_score > 0:
                num_hits += 1
                shop_ids.add(product["shop_id"])

        if len(reward) > 0:
            score["product"] /= len(reward)
            score["gt"] /= len(reward)
            score["rule"] /= len(reward)
            for field in FIELDS:
                score[field] /= len(reward)
        score["shop"] = 1 if num_hits == len(reward) and len(shop_ids) == 1 else 0

    def _eval_voucher(
        self, score: dict, output: list[dict], reward: list[dict], voucher: dict
    ):
        """Evaluate voucher task (budget constraint).

        Args:
            score: Score dictionary to update
            output: Rollout output
            reward: List of ground truth rewards
            voucher: Voucher constraints (budget, discount, etc.)
        """
        num_hits = 0
        total_price = 0
        shop_ids = set()
        product_id_list = self._extract_recommended_product(output)

        for i, sub_reward in enumerate(reward):
            if i >= len(product_id_list):
                continue
            product_id = product_id_list[i]

            product = get_product(product_id)
            if not product:
                continue

            rule_score = self._set_eval_score(product, score, sub_reward)
            if rule_score > 0:
                num_hits += 1
                total_price += product["price"]
                shop_ids.add(product["shop_id"])

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

        if len(reward) > 0:
            score["product"] /= len(reward)
            score["gt"] /= len(reward)
            score["rule"] /= len(reward)
            for field in FIELDS:
                score[field] /= len(reward)
        score["budget"] = budget_match

    def write_progress(self, problem_id: str, score: Optional[dict], output_file: Path):
        """Write problem score to progress file in JSONL format.

        This writes a single line to the output file that ProgressReporter
        can parse and send to the Backend API.

        Args:
            problem_id: Unique identifier for this problem
            score: Score dictionary from score_problem(), or None if scoring failed
            output_file: Path to output JSONL file
        """
        # Determine status from score
        if score is None:
            status = "FAILED"
        elif self.task == "product":
            status = "SUCCESS" if score.get("rule", 0) >= 1 else "FAILED"
        elif self.task == "shop":
            status = (
                "SUCCESS"
                if score.get("rule", 0) >= 1 and score.get("shop", 0) >= 1
                else "FAILED"
            )
        elif self.task == "voucher":
            status = (
                "SUCCESS"
                if score.get("rule", 0) >= 1 and score.get("budget", 0) >= 1
                else "FAILED"
            )
        else:
            status = "FAILED"

        # Build progress entry
        progress_entry = {"problem_id": problem_id, "status": status, "score": score}

        # Append to file (create if doesn't exist)
        with open(output_file, "a") as f:
            f.write(json.dumps(progress_entry) + "\n")
