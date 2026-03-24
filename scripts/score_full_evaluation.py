#!/usr/bin/env python3
"""Score sandbox outputs from a full evaluation run and generate a solvability report.

Reads the dialogue JSONL files produced by run_full_evaluation.sh, scores each
problem using ProblemScorer (via search-server HTTP API), classifies difficulty,
and outputs a JSON report identifying which problems the example agent can solve.

Usage:
    source .venv/bin/activate
    PYTHONPATH=src/agent python scripts/score_full_evaluation.py -o reports/full_eval_report.json

Prerequisites:
    - search-server must be running: docker compose up -d search-server
    - SEARCH_SERVER_URL defaults to http://localhost:5632
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add src/agent to path so ProblemScorer imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "agent"))

from problem_scorer import ProblemScorer  # noqa: E402


# ---------------------------------------------------------------------------
# Category configs
# ---------------------------------------------------------------------------

CATEGORIES = {
    "product": {
        "source_file": "data/synthesize_product_test.jsonl",
        "output_file": "logs/full_eval_product.jsonl",
        "task": "product",
    },
    "shop": {
        "source_file": "data/synthesize_shop_test.jsonl",
        "output_file": "logs/full_eval_shop.jsonl",
        "task": "shop",
    },
    "voucher": {
        "source_file": "data/synthesize_voucher_test.jsonl",
        "output_file": "logs/full_eval_voucher.jsonl",
        "task": "voucher",
    },
}


# ---------------------------------------------------------------------------
# Difficulty classifiers (from select_launch_problems.py on kri-121 branch)
# ---------------------------------------------------------------------------

def _count_product_constraints(reward: dict) -> int:
    count = 0
    if "product_id" in reward:
        count += 1
    if "title" in reward:
        count += 1
    if "sku_options" in reward:
        count += len(reward["sku_options"])
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
    return "hard"


def classify_shop(problem: dict) -> str:
    n = len(problem["reward"])
    if n <= 2:
        return "easy"
    elif n == 3:
        return "medium"
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
    return "hard"


CLASSIFIERS = {
    "product": classify_product,
    "shop": classify_shop,
    "voucher": classify_voucher,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_source_problems(path: str) -> list[dict]:
    """Load source JSONL file with query, reward, and optional voucher fields."""
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def load_sandbox_output(path: str) -> dict[str, list[dict]]:
    """Load sandbox dialogue JSONL. Returns {query: dialogue_steps}."""
    outputs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dialogue = json.loads(line)
            if not dialogue or not isinstance(dialogue, list):
                continue
            # Extract query from first step's extra_info
            try:
                query = dialogue[0]["extra_info"]["query"]
                outputs[query] = dialogue
            except (KeyError, IndexError, TypeError):
                continue
    return outputs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_category(category: str, config: dict, project_dir: Path) -> dict:
    """Score all problems in a category and return results."""
    source_path = project_dir / config["source_file"]
    output_path = project_dir / config["output_file"]
    task = config["task"]

    print(f"\n--- Scoring {category} ---")

    # Load source problems
    problems = load_source_problems(str(source_path))
    print(f"  Source problems: {len(problems)}")

    # Build reward/voucher dicts
    rewards = {}
    vouchers = {}
    for p in problems:
        q = p["query"]
        rewards[q] = p["reward"]
        if "voucher" in p:
            vouchers[q] = p["voucher"]

    # Load sandbox output
    if not output_path.exists():
        print(f"  WARNING: Output file not found: {output_path}")
        print(f"  Marking all {len(problems)} problems as unsolved.")
        outputs = {}
    else:
        outputs = load_sandbox_output(str(output_path))
        print(f"  Sandbox outputs: {len(outputs)}")

    # Create scorer
    scorer = ProblemScorer(task=task, rewards=rewards, vouchers=vouchers)
    classifier = CLASSIFIERS[category]

    # Score each problem
    results = []
    solvable_queries = []
    missing_count = 0
    error_count = 0

    for i, problem in enumerate(problems):
        query = problem["query"]
        difficulty = classifier(problem)

        if query not in outputs:
            # Agent didn't produce output for this problem
            missing_count += 1
            results.append({
                "query": query,
                "difficulty": difficulty,
                "solvable": False,
                "gt": 0,
                "status": "missing",
            })
            continue

        dialogue = outputs[query]
        try:
            score = scorer.score_problem(query=query, output=dialogue)
        except Exception as e:
            error_count += 1
            results.append({
                "query": query,
                "difficulty": difficulty,
                "solvable": False,
                "gt": 0,
                "status": "error",
                "error": str(e),
            })
            continue

        if score is None:
            error_count += 1
            results.append({
                "query": query,
                "difficulty": difficulty,
                "solvable": False,
                "gt": 0,
                "status": "no_reward",
            })
            continue

        gt = score.get("gt", 0)
        solvable = gt > 0

        if solvable:
            solvable_queries.append(query)

        results.append({
            "query": query,
            "difficulty": difficulty,
            "solvable": solvable,
            "gt": gt,
            "rule": score.get("rule", 0),
            "score": score,
            "status": "scored",
        })

        # Progress indicator every 50 problems
        if (i + 1) % 50 == 0:
            print(f"  Scored {i + 1}/{len(problems)}...")

    print(f"  Done: {len(problems)} total, {missing_count} missing, {error_count} errors")

    # Aggregate stats
    total = len(problems)
    solvable_count = len(solvable_queries)

    by_difficulty = {}
    for tier in ("easy", "medium", "hard"):
        tier_results = [r for r in results if r["difficulty"] == tier]
        tier_solvable = [r for r in tier_results if r["solvable"]]
        tier_total = len(tier_results)
        by_difficulty[tier] = {
            "total": tier_total,
            "solvable": len(tier_solvable),
            "solvable_rate": len(tier_solvable) / tier_total if tier_total > 0 else 0,
        }

    return {
        "summary": {
            "total": total,
            "solvable": solvable_count,
            "solvable_rate": solvable_count / total if total > 0 else 0,
            "missing": missing_count,
            "errors": error_count,
        },
        "by_difficulty": by_difficulty,
        "solvable_queries": solvable_queries,
        "per_problem": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score full evaluation run and generate solvability report"
    )
    parser.add_argument(
        "-o", "--output",
        default="reports/full_eval_report.json",
        help="Output JSON report path (default: reports/full_eval_report.json)",
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help="ShoppingBench project root (default: auto-detect from script location)",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir) if args.project_dir else Path(__file__).resolve().parent.parent
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_dir / output_path

    print(f"Project dir: {project_dir}")
    print(f"Output: {output_path}")

    start_time = time.time()

    report = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "grand_total": {"total": 0, "solvable": 0, "solvable_rate": 0},
        "categories": {},
    }

    grand_total = 0
    grand_solvable = 0

    for category, config in CATEGORIES.items():
        cat_result = score_category(category, config, project_dir)
        report["categories"][category] = {
            "summary": cat_result["summary"],
            "by_difficulty": cat_result["by_difficulty"],
            "solvable_queries": cat_result["solvable_queries"],
        }
        grand_total += cat_result["summary"]["total"]
        grand_solvable += cat_result["summary"]["solvable"]

    report["grand_total"] = {
        "total": grand_total,
        "solvable": grand_solvable,
        "solvable_rate": grand_solvable / grand_total if grand_total > 0 else 0,
    }

    duration = time.time() - start_time

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary to stdout
    print(f"\n{'=' * 60}")
    print("  FULL EVALUATION REPORT")
    print(f"{'=' * 60}")
    print(f"  Timestamp: {report['timestamp']}")
    print(f"  Duration:  {duration:.1f}s")
    print("")
    print(f"  {'Category':<12} {'Total':>6} {'Solvable':>10} {'Rate':>8}")
    print(f"  {'-' * 40}")

    for cat_name, cat_data in report["categories"].items():
        s = cat_data["summary"]
        print(f"  {cat_name:<12} {s['total']:>6} {s['solvable']:>10} {s['solvable_rate']:>7.1%}")

    gt = report["grand_total"]
    print(f"  {'-' * 40}")
    print(f"  {'TOTAL':<12} {gt['total']:>6} {gt['solvable']:>10} {gt['solvable_rate']:>7.1%}")
    print()

    # Difficulty breakdown
    print("  Difficulty Breakdown:")
    print(f"  {'Category':<12} {'Easy':>12} {'Medium':>12} {'Hard':>12}")
    print(f"  {'-' * 52}")
    for cat_name, cat_data in report["categories"].items():
        bd = cat_data["by_difficulty"]
        easy = f"{bd['easy']['solvable']}/{bd['easy']['total']}"
        med = f"{bd['medium']['solvable']}/{bd['medium']['total']}"
        hard = f"{bd['hard']['solvable']}/{bd['hard']['total']}"
        print(f"  {cat_name:<12} {easy:>12} {med:>12} {hard:>12}")

    print(f"\n  Report written to: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
