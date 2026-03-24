#!/usr/bin/env python3
"""Select a balanced set of 90 problems (30 per category) for the launch suite.

Reads the 3 ShoppingBench JSONL source files, classifies each problem by
difficulty using heuristics, then randomly selects 10 easy + 10 medium + 10 hard
from each category. Outputs a single JSON array file compatible with
load_problem_suite_from_s3().

Optionally excludes problems that a baseline agent solved in one-shot evaluation
(--exclude-report), replacing them with other problems from the same difficulty tier.

Difficulty heuristics:
  Product - reward constraint count: 2-4 easy, 5-6 medium, 7+ hard
  Shop    - items per query: 2 easy, 3 medium, 4 hard
  Voucher - composite score (item_count + shop_voucher + pct_discount):
            1-2 easy, 3-4 medium, 5-6 hard
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SOURCES = {
    "Product": DATA_DIR / "synthesize_product_test.jsonl",
    "Shop": DATA_DIR / "synthesize_shop_test.jsonl",
    "Voucher": DATA_DIR / "synthesize_voucher_test.jsonl",
}

TARGET_PER_TIER = 10  # 10 easy + 10 medium + 10 hard = 30 per category


# ---------------------------------------------------------------------------
# Difficulty classifiers
# ---------------------------------------------------------------------------

def _count_product_constraints(reward: dict) -> int:
    """Count constraint fields in a Product reward dict."""
    count = 0
    if "product_id" in reward:
        count += 1
    if "title" in reward:
        count += 1
    if "sku_options" in reward:
        count += len(reward["sku_options"])  # list of dicts, each key is a constraint
    if "attributes" in reward:
        count += len(reward["attributes"])
    if "price" in reward:
        count += len(reward["price"])
    if "service" in reward:
        count += len(reward["service"]) if isinstance(reward["service"], list) else 1
    return count


def classify_product(problem: dict) -> str:
    n = _count_product_constraints(problem["reward"])
    if n <= 4:
        return "easy"
    elif n <= 6:
        return "medium"
    else:
        return "hard"


def classify_shop(problem: dict) -> str:
    n = len(problem["reward"])  # reward is a list of product dicts
    if n <= 2:
        return "easy"
    elif n == 3:
        return "medium"
    else:
        return "hard"


def classify_voucher(problem: dict) -> str:
    reward = problem["reward"]
    voucher = problem["voucher"]
    score = len(reward)
    if voucher.get("voucher_type") == "shop":
        score += 1
    if voucher.get("discount_type") == "percentage":
        score += 1
    if score <= 2:
        return "easy"
    elif score <= 4:
        return "medium"
    else:
        return "hard"


CLASSIFIERS = {
    "Product": classify_product,
    "Shop": classify_shop,
    "Voucher": classify_voucher,
}


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def load_excluded_queries(report_path: Path) -> set[str]:
    """Load solvable query strings from an evaluation report to exclude."""
    report = json.loads(report_path.read_text())
    excluded: set[str] = set()
    for data in report["categories"].values():
        for q in data.get("solvable_queries", []):
            excluded.add(q)
    return excluded


def select_problems(
    seed: int = 42, excluded_queries: set[str] | None = None
) -> list[dict]:
    rng = random.Random(seed)
    selected = []
    excluded_queries = excluded_queries or set()

    for category, source_path in SOURCES.items():
        if not source_path.exists():
            print(f"ERROR: Source file not found: {source_path}", file=sys.stderr)
            sys.exit(1)

        problems = load_jsonl(source_path)
        classifier = CLASSIFIERS[category]

        # Bucket by difficulty, excluding solvable problems
        buckets: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
        n_excluded = 0
        for p in problems:
            if p["query"] in excluded_queries:
                n_excluded += 1
                continue
            tier = classifier(p)
            buckets[tier].append(p)

        suffix = f" ({n_excluded} excluded as too easy)" if n_excluded else ""
        print(f"{category}: {len(problems)} total{suffix} — "
              f"easy={len(buckets['easy'])}, "
              f"medium={len(buckets['medium'])}, "
              f"hard={len(buckets['hard'])}")

        for tier in ("easy", "medium", "hard"):
            pool = buckets[tier]
            if len(pool) < TARGET_PER_TIER:
                print(f"  WARNING: {category}/{tier} has only {len(pool)} problems "
                      f"(need {TARGET_PER_TIER}), taking all", file=sys.stderr)
                picks = pool
            else:
                picks = rng.sample(pool, TARGET_PER_TIER)

            for i, p in enumerate(picks, start=1):
                p["category"] = category
                p["difficulty"] = tier
                p["title"] = f"{category} {tier.title()} {i}"
            selected.extend(picks)

    return selected


def validate(problems: list[dict]) -> bool:
    """Validate against load_problem_suite_from_s3 requirements."""
    required = ["category", "query", "reward"]
    ok = True
    for i, p in enumerate(problems):
        missing = [f for f in required if f not in p]
        if missing:
            print(f"VALIDATION FAIL: problem {i} missing {missing}", file=sys.stderr)
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description="Select launch problem suite")
    parser.add_argument("-o", "--output", required=True,
                        help="Output JSON file path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--exclude-report", type=Path, default=None,
                        help="Eval report JSON — exclude solvable queries as too easy")
    args = parser.parse_args()

    excluded = set()
    if args.exclude_report:
        excluded = load_excluded_queries(args.exclude_report)
        print(f"Excluding {len(excluded)} solvable queries from {args.exclude_report}\n")

    problems = select_problems(seed=args.seed, excluded_queries=excluded)

    if not validate(problems):
        sys.exit(1)

    # Summary
    from collections import Counter
    cat_counts = Counter(p["category"] for p in problems)
    print(f"\nTotal: {len(problems)} problems")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(problems, f, indent=2)

    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
