#!/usr/bin/env python3
"""
Standalone local evaluation script for ShoppingBench.

Evaluates sandbox output against ground truth without requiring Java/Pyserini.
Computes GT (ground truth) scores by comparing recommended product IDs against
the expected product IDs in the reward data.

Usage:
    python scripts/eval_local.py
    python scripts/eval_local.py --sandbox logs/sandbox_output_local-test.jsonl --ground-truth data/test_subset.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def extract_recommend_product(output: list[dict]) -> str:
    """Extract recommended product IDs from the last recommend_product tool call.

    Mirrors the logic in src/agent/run_evaluate.py:extract_recommed_product().
    Iterates through all steps and returns the product_ids from the LAST
    recommend_product call found.
    """
    product_ids = ""
    if not output:
        return product_ids

    for step in output:
        message = step["completion"]["message"]
        if message and "tool_call" in message and message["tool_call"]:
            for command in message["tool_call"]:
                if command["name"] == "recommend_product":
                    product_ids = command["parameters"].get("product_ids", "")
    if not isinstance(product_ids, str):
        return ""
    return product_ids


def eval_product_gt(output: list[dict], reward: dict) -> dict:
    """Evaluate a Product problem: check if first recommended ID matches reward.

    For Product problems, the reward is a single dict with a "product_id" field.
    The agent should recommend exactly one product, and the first ID in the
    comma-separated list must match.
    """
    product_ids_str = extract_recommend_product(output)
    recommended_ids = [pid.strip() for pid in product_ids_str.split(",") if pid.strip()]
    first_id = recommended_ids[0] if recommended_ids else ""

    expected_id = reward["product_id"]
    gt = 1.0 if first_id == expected_id else 0.0

    return {
        "recommended_ids": recommended_ids,
        "expected_ids": [expected_id],
        "gt": gt,
    }


def eval_shop_gt(output: list[dict], reward: list[dict]) -> dict:
    """Evaluate a Shop problem: positional matching of product IDs.

    For Shop problems, the reward is a list of dicts, each with a "product_id".
    The recommended IDs are positionally matched: product_id_list[i] must match
    reward[i]["product_id"]. GT score is the fraction of positional matches.
    """
    product_ids_str = extract_recommend_product(output)
    recommended_ids = [pid.strip() for pid in product_ids_str.split(",") if pid.strip()]

    expected_ids = [r["product_id"] for r in reward]
    num_matches = 0

    for i, expected in enumerate(expected_ids):
        if i < len(recommended_ids) and recommended_ids[i] == expected:
            num_matches += 1

    gt = num_matches / len(expected_ids) if expected_ids else 0.0

    # Shop score: all must match AND all from same shop (we cannot verify shop
    # without index data, so we just report the GT fraction)
    all_match = num_matches == len(expected_ids)

    return {
        "recommended_ids": recommended_ids,
        "expected_ids": expected_ids,
        "gt": gt,
        "all_positions_match": all_match,
    }


def eval_voucher_gt(output: list[dict], reward: list[dict], voucher: dict) -> dict:
    """Evaluate a Voucher problem: positional matching plus budget info.

    Same positional matching as Shop problems. Budget check requires knowing
    actual prices from the product index, which we do not have access to here.
    We report the GT positional match score and note the budget constraint.
    """
    product_ids_str = extract_recommend_product(output)
    recommended_ids = [pid.strip() for pid in product_ids_str.split(",") if pid.strip()]

    expected_ids = [r["product_id"] for r in reward]
    num_matches = 0

    for i, expected in enumerate(expected_ids):
        if i < len(recommended_ids) and recommended_ids[i] == expected:
            num_matches += 1

    gt = num_matches / len(expected_ids) if expected_ids else 0.0
    all_match = num_matches == len(expected_ids)

    return {
        "recommended_ids": recommended_ids,
        "expected_ids": expected_ids,
        "gt": gt,
        "all_positions_match": all_match,
        "budget": voucher.get("budget"),
        "price_after_voucher": voucher.get("price_after_voucher"),
        "budget_note": "Budget check requires product index (not available locally)",
    }


def load_sandbox_output(path: Path) -> dict:
    """Load sandbox output, keyed by query string."""
    outputs = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            query = data[0]["extra_info"]["query"]
            outputs[query] = data
    return outputs


def load_ground_truth(path: Path) -> list[dict]:
    """Load ground truth problems from the synthesized test file."""
    problems = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))
    return problems


def format_ids(ids: list[str], max_display: int = 5) -> str:
    """Format a list of IDs for display, truncating if too many."""
    if len(ids) <= max_display:
        return ", ".join(ids) if ids else "(none)"
    return ", ".join(ids[:max_display]) + f" ... (+{len(ids) - max_display} more)"


def main():
    parser = argparse.ArgumentParser(
        description="Local evaluation of ShoppingBench sandbox output (GT scoring only)"
    )
    parser.add_argument(
        "--sandbox",
        type=Path,
        default=Path("logs/sandbox_output_local-test.jsonl"),
        help="Path to sandbox output JSONL file",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/test_subset.jsonl"),
        help="Path to ground truth JSONL file",
    )
    args = parser.parse_args()

    # Resolve paths relative to the project root (parent of scripts/)
    project_root = Path(__file__).resolve().parent.parent
    sandbox_path = args.sandbox if args.sandbox.is_absolute() else project_root / args.sandbox
    gt_path = args.ground_truth if args.ground_truth.is_absolute() else project_root / args.ground_truth

    if not sandbox_path.exists():
        print(f"Error: Sandbox output file not found: {sandbox_path}", file=sys.stderr)
        sys.exit(1)
    if not gt_path.exists():
        print(f"Error: Ground truth file not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    # Load data
    sandbox_outputs = load_sandbox_output(sandbox_path)
    gt_problems = load_ground_truth(gt_path)

    print(f"Loaded {len(sandbox_outputs)} sandbox outputs from: {sandbox_path}")
    print(f"Loaded {len(gt_problems)} ground truth problems from: {gt_path}")
    print()

    # Evaluate each problem
    results = []
    category_scores = defaultdict(list)
    difficulty_scores = defaultdict(list)

    for problem in gt_problems:
        query = problem["query"]
        category = problem["category"]
        difficulty = problem["difficulty"]
        title = problem["title"]
        reward = problem["reward"]
        voucher = problem.get("voucher")

        output = sandbox_outputs.get(query)

        if output is None:
            result = {
                "title": title,
                "category": category,
                "difficulty": difficulty,
                "status": "MISSING",
                "recommended_ids": [],
                "expected_ids": [],
                "gt": 0.0,
            }
            if category == "Product":
                result["expected_ids"] = [reward["product_id"]]
            else:
                result["expected_ids"] = [r["product_id"] for r in reward]
        else:
            task = category.lower()
            if task == "product":
                eval_result = eval_product_gt(output, reward)
            elif task == "shop":
                eval_result = eval_shop_gt(output, reward)
            elif task == "voucher":
                eval_result = eval_voucher_gt(output, reward, voucher)
            else:
                print(f"  Warning: Unknown category '{category}', skipping", file=sys.stderr)
                continue

            result = {
                "title": title,
                "category": category,
                "difficulty": difficulty,
                "status": "EVALUATED",
                **eval_result,
            }

        results.append(result)
        category_scores[category].append(result["gt"])
        difficulty_scores[difficulty].append(result["gt"])

    # Print per-problem results
    print("=" * 80)
    print("PER-PROBLEM RESULTS")
    print("=" * 80)

    for r in results:
        gt_str = f"{r['gt']:.2f}"
        status_marker = "MISS" if r["status"] == "MISSING" else ("PASS" if r["gt"] >= 1.0 else "FAIL")

        print(f"\n[{status_marker}] {r['title']}")
        print(f"  Category:    {r['category']} | Difficulty: {r['difficulty']}")
        print(f"  Expected:    {format_ids(r['expected_ids'])}")
        print(f"  Recommended: {format_ids(r['recommended_ids'])}")
        print(f"  GT Score:    {gt_str}")

        if "all_positions_match" in r:
            print(f"  All Match:   {'Yes' if r['all_positions_match'] else 'No'}")
        if "budget_note" in r:
            budget = r.get("budget")
            price_after = r.get("price_after_voucher")
            print(f"  Budget:      {budget} (expected price after voucher: {price_after})")
            print(f"  Note:        {r['budget_note']}")

    # Print aggregate scores
    print()
    print("=" * 80)
    print("AGGREGATE SCORES")
    print("=" * 80)

    total_gt_scores = [r["gt"] for r in results]
    evaluated_gt_scores = [r["gt"] for r in results if r["status"] == "EVALUATED"]

    num_total = len(results)
    num_evaluated = len(evaluated_gt_scores)
    num_missing = num_total - num_evaluated

    print(f"\nTotal problems:     {num_total}")
    print(f"Evaluated:          {num_evaluated}")
    print(f"Missing output:     {num_missing}")

    if num_evaluated > 0:
        avg_gt_evaluated = sum(evaluated_gt_scores) / num_evaluated
        gt_pass_evaluated = sum(1 for s in evaluated_gt_scores if s >= 1.0)
        print("\nGT Score (evaluated only):")
        print(f"  Average:          {avg_gt_evaluated:.3f}")
        print(f"  Pass rate:        {gt_pass_evaluated}/{num_evaluated} ({gt_pass_evaluated/num_evaluated:.1%})")

    if num_total > 0:
        avg_gt_all = sum(total_gt_scores) / num_total
        gt_pass_all = sum(1 for s in total_gt_scores if s >= 1.0)
        print("\nGT Score (all problems, missing=0):")
        print(f"  Average:          {avg_gt_all:.3f}")
        print(f"  Pass rate:        {gt_pass_all}/{num_total} ({gt_pass_all/num_total:.1%})")

    # By category
    print("\nBy Category:")
    for cat in ["Product", "Shop", "Voucher"]:
        scores = category_scores.get(cat, [])
        if scores:
            avg = sum(scores) / len(scores)
            passed = sum(1 for s in scores if s >= 1.0)
            print(f"  {cat:10s}  avg={avg:.3f}  pass={passed}/{len(scores)}")

    # By difficulty
    print("\nBy Difficulty:")
    for diff in ["easy", "medium", "hard"]:
        scores = difficulty_scores.get(diff, [])
        if scores:
            avg = sum(scores) / len(scores)
            passed = sum(1 for s in scores if s >= 1.0)
            print(f"  {diff:10s}  avg={avg:.3f}  pass={passed}/{len(scores)}")

    print()


if __name__ == "__main__":
    main()
