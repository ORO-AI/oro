#!/usr/bin/env python3
"""
Main entry point for sandbox execution.

This script loads problems from a JSONL file, executes the agent against them
in parallel, and outputs results in a format compatible with evaluation.
"""

import argparse
import json
import logging
import os
import sys

from src.agent.sandbox_config import SandboxConfig
from src.agent.sandbox_executor import (
    load_problems,
    execute_problems_parallel,
    format_results,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Execute agent against ShoppingBench problems in sandbox"
    )
    parser.add_argument(
        "--problem-file",
        type=str,
        default="data/synthesize_product_test.jsonl",
        help="Path to JSONL file containing problems",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout per problem in seconds (default: 300)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (overrides other arguments)",
    )
    parser.add_argument(
        "--agent-file",
        type=str,
        default=None,
        help="Path to agent file to use (default: use built-in src.agent.agent)",
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        config = SandboxConfig(**config_dict)
    else:
        config = SandboxConfig(
            problem_file=args.problem_file,
            max_workers=args.max_workers,
            timeout_per_problem=args.timeout,
            output_file=args.output,
            agent_file=args.agent_file,
        )

    try:
        config.validate()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    os.environ["SANDBOX_PROXY_URL"] = config.sandbox_proxy_url
    if config.output_file:
        os.environ["SANDBOX_OUTPUT_FILE"] = config.output_file

    logger.info(f"Loading problems from {config.problem_file}")
    problems = load_problems(config.problem_file)

    if not problems:
        logger.error("No problems loaded")
        sys.exit(1)

    # Execute problems (scoring will be done by validator, not sandbox)
    logger.info(f"Executing {len(problems)} problems...")
    results = execute_problems_parallel(
        problems=problems,
        max_workers=config.max_workers,
        timeout_per_problem=config.timeout_per_problem,
        agent_file=config.agent_file,
        output_file=config.output_file,
    )

    if config.output_file:
        # Results already written incrementally by execute_problems_parallel
        logger.info(f"Dialogue output written to {config.output_file}")
        logger.info("To evaluate, run: python -m src.agent.run_evaluate <config.json>")
    else:
        # If no output file, write to stdout in JSONL format
        for result in results:
            if result.success and isinstance(result.result, list):
                print(json.dumps(result.result))

    # Also format and display summary
    formatted_results = format_results(results)
    summary_json = json.dumps(
        formatted_results["summary"], indent=2, ensure_ascii=False
    )
    logger.info(f"Execution summary:\n{summary_json}")

    # Log failures but exit cleanly — timeouts and agent errors are expected
    # outcomes, not infrastructure failures. The validator scores partial results.
    failed = formatted_results["summary"]["failed"]
    total = formatted_results["summary"]["total"]
    if failed > 0:
        logger.warning(f"{failed}/{total} problems failed (timeouts or agent errors)")
    else:
        logger.info("All problems executed successfully")


if __name__ == "__main__":
    main()
