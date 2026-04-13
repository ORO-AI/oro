"""Local agent testing — run an agent against test problems and print score.

Usage (via Docker Compose):
    docker compose run test --agent-file my_agent.py

Usage (direct):
    python -m subnet.test_runner --agent-file my_agent.py
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Search server URL for ProblemScorer product lookups — must be set BEFORE
# importing ProblemScorer, which reads SEARCH_SERVER_URL at module level.
SEARCH_SERVER_URL = os.environ.get("SEARCH_SERVER_URL", "http://search-server:5632")
os.environ.setdefault("SEARCH_SERVER_URL", SEARCH_SERVER_URL)

# ProblemScorer uses HTTP calls to search-server — no pyserini/Java needed
sys.path.insert(0, "/app/src/agent")
from problem_scorer import ProblemScorer  # noqa: E402
from scoring import is_problem_successful, compute_aggregate  # noqa: E402

from subnet.sandbox import (  # noqa: E402
    SANDBOX_IMAGE,
    HOST_PROJECT_DIR,
    host_path,
    load_problems,
    build_sandbox_command,
    attach_title_embeddings,
)

# Default test problem file — problem suite v1 (90 problems)
DEFAULT_PROBLEM_FILE = "data/suites/problem_suite_v3.json"


def _write_jsonl(problems: list[dict], output_path: Path) -> None:
    """Write problems as JSONL (sandbox expects this format)."""
    with open(output_path, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")


def _score_output(output_file: Path, problems: list[dict]) -> float:
    """Score agent output using ProblemScorer (HTTP-based, no pyserini).

    Returns:
        Score (0.0 - 1.0), or -1.0 on failure.
    """

    # Build rewards and vouchers lookup, grouped by task
    rewards = {}
    vouchers = {}
    task_for_query = {}
    for p in problems:
        query = p.get("query", "")
        reward = p.get("reward")
        category = p.get("category", "Product").lower()
        if query and reward:
            # Attach precomputed title embeddings to the reward dict(s)
            attach_title_embeddings(reward, p.get("reward_title_embeddings"))
            rewards[query] = reward
            task_for_query[query] = category
            voucher = p.get("voucher")
            if voucher:
                vouchers[query] = voucher

    # Create scorers per task type
    scorers = {}
    for task in set(task_for_query.values()):
        task_rewards = {q: r for q, r in rewards.items() if task_for_query[q] == task}
        task_vouchers = {
            q: v for q, v in vouchers.items() if task_for_query.get(q) == task
        }
        scorers[task] = ProblemScorer(
            task=task, rewards=task_rewards, vouchers=task_vouchers
        )

    scores = []
    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            output = json.loads(line)
            if not output:
                continue
            query = output[0].get("extra_info", {}).get("query", "")
            if query not in rewards:
                continue

            task = task_for_query[query]
            score = scorers[task].score_problem(query=query, output=output)
            scores.append({"score_dict": score, "category": task, "query": query})

            gt = score.get("gt", 0)
            rule = score.get("rule", 0)
            status = "PASS" if is_problem_successful(score, task) else "FAIL"
            short_query = query[:55] + "..." if len(query) > 55 else query
            print(f"  [{status}] {task:8s} gt={gt:.2f} rule={rule:.2f} | {short_query}")

    if not scores:
        print("Warning: No problems scored", file=sys.stderr)
        return -1.0

    agg = compute_aggregate(scores, total_problems=len(problems))

    # Per-category breakdown
    categories = sorted(set(s["category"] for s in scores))
    if len(categories) > 1:
        print()
        for cat in categories:
            cat_scores = [s for s in scores if s["category"] == cat]
            cat_pass = sum(1 for s in cat_scores if is_problem_successful(s["score_dict"], cat))
            print(f"  {cat:8s}: {cat_pass}/{len(cat_scores)} passed ({len(cat_scores)} problems)")

    print()
    print(f"Ground truth rate: {agg['ground_truth_rate']:.3f}")
    print(f"Success rate:      {agg['success_rate']:.3f}")
    print(f"Format score:      {agg['format_score']:.3f}")
    print(f"Field matching:    {agg['field_matching']:.3f}")

    return agg["success_rate"]


def run_test(
    agent_file: str,
    problem_file: str = DEFAULT_PROBLEM_FILE,
    max_workers: int = 3,
    timeout: int = 1800,
) -> float:
    """Run agent against test problems and return score.

    Args:
        agent_file: Path to the agent Python file.
        problem_file: Path to the problem file (JSON array or JSONL).
        max_workers: Maximum parallel workers for sandbox execution.
        timeout: Timeout in seconds for sandbox execution.

    Returns:
        Score (0.0 - 1.0), or -1.0 on failure.
    """
    agent_path = Path(agent_file)
    if not agent_path.exists():
        print(f"Error: Agent file not found: {agent_path}", file=sys.stderr)
        return -1.0

    # Set up output paths
    logs_dir = Path("/app/logs") if HOST_PROJECT_DIR else Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_id = "local-test"
    output_file = logs_dir / f"sandbox_output_{eval_id}.jsonl"

    # Clean up any previous test output
    if output_file.exists():
        output_file.unlink()

    # Resolve paths for Docker volume mounts
    host_agent = host_path(str(agent_path.resolve()))
    host_logs = host_path(str(logs_dir.resolve()))

    # Problem file handling
    problem_path = Path(problem_file)
    if not problem_path.is_absolute():
        for prefix in [Path("/workspace"), Path("/app"), Path(".")]:
            candidate = prefix / problem_file
            if candidate.exists():
                problem_path = candidate
                break

    if not problem_path.exists():
        print(f"Error: Problem file not found: {problem_file}", file=sys.stderr)
        return -1.0

    problems = load_problems(problem_path)
    if not problems:
        print(f"Error: No problems loaded from {problem_file}", file=sys.stderr)
        return -1.0

    # Convert to JSONL for sandbox (it expects JSONL format)
    sandbox_problems = logs_dir / "test_problems.jsonl"
    _write_jsonl(problems, sandbox_problems)
    host_problem = host_path(str(sandbox_problems.resolve()))

    print(f"Agent:    {agent_path.name}")
    print(f"Problems: {problem_path.name} ({len(problems)} problems)")
    print(f"Image:    {SANDBOX_IMAGE}")
    print()
    print("Starting sandbox...")

    # Build docker run command
    chutes_key = os.environ.get("CHUTES_API_KEY")
    cmd = build_sandbox_command(
        agent_host_path=host_agent,
        logs_host_path=host_logs,
        problem_file_arg="/tmp/test_problems.jsonl",
        output_path=f"/app/logs/sandbox_output_{eval_id}.jsonl",
        extra_volumes=[(host_problem, "/tmp/test_problems.jsonl")],
        max_workers=max_workers,
        chutes_access_token=chutes_key,
    )

    try:
        result = subprocess.run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"Error: Sandbox timed out ({timeout}s limit)", file=sys.stderr)
        return -1.0
    except Exception as e:
        print(f"Error running sandbox: {e}", file=sys.stderr)
        return -1.0

    if not output_file.exists():
        if result.returncode != 0:
            print(f"Error: Sandbox exited with code {result.returncode}", file=sys.stderr)
        print("Error: No output file produced", file=sys.stderr)
        return -1.0

    if result.returncode != 0:
        print(f"Warning: Sandbox exited with code {result.returncode}, scoring partial results", file=sys.stderr)

    # Score using ProblemScorer (HTTP calls to search-server, no pyserini)
    print()
    print("Scoring results...")
    return _score_output(output_file, problems)


def main():
    parser = argparse.ArgumentParser(
        prog="test_runner",
        description="Test an agent locally against benchmark problems",
    )
    parser.add_argument(
        "--agent-file",
        required=True,
        help="Path to the agent Python file to test",
    )
    parser.add_argument(
        "--problem-file",
        default=DEFAULT_PROBLEM_FILE,
        help=f"Path to problem file, JSON or JSONL (default: {DEFAULT_PROBLEM_FILE})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum parallel workers for sandbox execution (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout in seconds for sandbox execution (default: 1800)",
    )

    args = parser.parse_args()
    score = run_test(args.agent_file, args.problem_file, args.max_workers, args.timeout)

    if score < 0:
        print("\nTest FAILED")
        sys.exit(1)

    print()
    print(f"Score: {score:.4f}")
    print(f"Result: {'PASS' if score > 0 else 'FAIL'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
