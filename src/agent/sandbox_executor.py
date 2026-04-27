"""Sandbox executor for running agents against problems in parallel."""

import importlib.util
import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
    as_completed,
)
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

from src.agent.sandbox_status import SandboxProblemStatus

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing an agent against a single problem."""

    query: str
    success: bool
    result: Optional[Dict] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    problem_id: Optional[str] = None
    inference_failure_count: int = 0
    inference_total: int = 0
    proxy_calls: Optional[List[Dict]] = None
    # FAILED is the safe default — every successful exit path sets SUCCESS,
    # every timeout path sets TIMED_OUT explicitly. Only the unexpected
    # construction omits status.
    status: SandboxProblemStatus = field(default=SandboxProblemStatus.FAILED)


def load_problems(problem_file: str) -> List[Dict]:
    """
    Load problems from a JSONL file.

    Args:
        problem_file: Path to JSONL file containing problems

    Returns:
        List of problem dictionaries (with reward removed)
    """
    problems = []
    try:
        with open(problem_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    problem = json.loads(line)
                    if "query" not in problem:
                        logger.warning(
                            f"Skipping line {line_num}: missing 'query' field"
                        )
                        continue
                    # Remove reward - agent should not see the expected answer
                    if "reward" in problem:
                        problem = {k: v for k, v in problem.items() if k != "reward"}
                    problems.append(problem)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse line {line_num} in {problem_file}: {e}"
                    )
                    continue
    except FileNotFoundError:
        logger.error(f"Problem file not found: {problem_file}")
        return []
    except Exception as e:
        logger.error(f"Error loading problems from {problem_file}: {e}")
        return []

    logger.info(f"Loaded {len(problems)} problems from {problem_file}")
    return problems


def load_agent_from_file(file_path: str) -> Callable:
    """
    Load agent_main function from a Python file.

    Args:
        file_path: Path to the agent Python file

    Returns:
        The agent_main function from the loaded module

    Raises:
        FileNotFoundError: If the file doesn't exist
        ImportError: If the module can't be loaded or agent_main is missing
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Agent file not found: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"Agent file path is not a file: {file_path}")

    module_name = f"user_agent_{abs(hash(file_path))}"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec from file: {file_path}")

    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass + PEP 563 annotations can resolve
    # string types via sys.modules[cls.__module__] during class definition.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    if not hasattr(module, "agent_main"):
        raise ImportError(f"Module {file_path} does not define 'agent_main' function")

    agent_main = getattr(module, "agent_main")
    if not callable(agent_main):
        raise ImportError(f"'agent_main' in {file_path} is not callable")

    logger.info(f"Succesfully loaded agent from {file_path}")
    return agent_main


def _load_agent(agent_file: Optional[str] = None) -> Callable:
    """Load agent_main function from file or default module.

    Args:
        agent_file: Optional path to agent file. If None, uses default src.agent.agent.

    Returns:
        The agent_main callable.
    """
    if agent_file:
        return load_agent_from_file(agent_file)
    from src.agent.agent import agent_main

    return agent_main


def _read_inference_stats(path: str, problem_id: str) -> tuple:
    """Read inference stats for a problem from the shared JSONL file.

    Multiple problems append to the same file (one line per inference call,
    cumulative counts). Returns the last matching entry's (failure_count, total).
    """
    last_failed, last_total = 0, 0
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if str(entry.get("problem_id")) == str(problem_id):
                    last_failed = entry.get("inference_failed", 0)
                    last_total = entry.get("inference_total", 0)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return last_failed, last_total


def _read_request_log(path: str) -> List[Dict]:
    """Read all proxy call entries from the JSONL sidecar file."""
    entries = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return entries


def _run_in_process(
    problem: Dict,
    agent_file: Optional[str],
    result_queue: multiprocessing.Queue,
    stats_file: Optional[str] = None,
    request_log_file: Optional[str] = None,
) -> None:
    """Target function executed in a child process.

    Loads the agent, runs it against *problem*, and puts the outcome into
    *result_queue*.  Any exception is caught and sent back as an error string
    so the parent never blocks on a dead queue.
    """
    if stats_file:
        os.environ["INFERENCE_STATS_FILE"] = stats_file
    if request_log_file:
        os.environ["REQUEST_LOG_FILE"] = request_log_file
    os.environ["PROBLEM_DATA"] = json.dumps(problem)
    try:
        agent_fn = _load_agent(agent_file)
        result = agent_fn(problem)
        result_queue.put(("success", result))
    except Exception as e:
        result_queue.put(("error", str(e)))


def execute_single_problem(
    problem: Dict,
    timeout: float = 300.0,
    agent_file: Optional[str] = None,
) -> ExecutionResult:
    """Execute agent_main() for a single problem with timeout.

    Spawns the agent in a child **process** so that a timed-out execution can
    be reliably terminated via SIGTERM/SIGKILL (unlike threads which continue
    running after ``future.result(timeout=...)`` raises).

    Args:
        problem: Problem dictionary with 'query' key (reward removed).
        timeout: Maximum execution time in seconds.
        agent_file: Path to agent file (loaded in child process). If None,
            uses default ``src.agent.agent``.

    Returns:
        ExecutionResult with execution outcome.
    """
    query = problem.get("query", "")
    problem_id = problem.get("problem_id") or problem.get("id")
    start_time = time.time()

    # All processes append to a shared JSONL file alongside output.jsonl.
    # Each line includes problem_id so the reader can match.
    output_file = os.environ.get("SANDBOX_OUTPUT_FILE", "")
    output_dir = os.path.dirname(output_file) if output_file else "/tmp"
    stats_file = os.path.join(output_dir, "inference_stats.jsonl")
    request_log_file = os.path.join(output_dir, f"request_log_{problem_id}.jsonl")
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_in_process,
        args=(problem, agent_file, result_queue, stats_file, request_log_file),
    )
    process.start()
    process.join(timeout=timeout)
    execution_time = time.time() - start_time

    timed_out = process.is_alive()
    if timed_out:
        logger.warning(
            f"Problem '{query}' exceeded timeout of {timeout}s, terminating process"
        )
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()

    inf_failures, inf_total = _read_inference_stats(stats_file, str(problem_id))
    proxy_calls = _read_request_log(request_log_file)

    if timed_out:
        result_queue.close()
        return ExecutionResult(
            query=query,
            success=False,
            error=f"Execution exceeded timeout of {timeout}s",
            execution_time=execution_time,
            problem_id=problem_id,
            inference_failure_count=inf_failures,
            inference_total=inf_total,
            proxy_calls=proxy_calls or None,
            status=SandboxProblemStatus.TIMED_OUT,
        )

    try:
        if not result_queue.empty():
            status, data = result_queue.get_nowait()
            if status == "success":
                return ExecutionResult(
                    query=query,
                    success=True,
                    result=data,
                    execution_time=execution_time,
                    problem_id=problem_id,
                    inference_failure_count=inf_failures,
                    inference_total=inf_total,
                    proxy_calls=proxy_calls or None,
                    status=SandboxProblemStatus.SUCCESS,
                )
            return ExecutionResult(
                query=query,
                success=False,
                error=data,
                execution_time=execution_time,
                problem_id=problem_id,
                inference_failure_count=inf_failures,
                inference_total=inf_total,
                proxy_calls=proxy_calls or None,
                status=SandboxProblemStatus.FAILED,
            )
        return ExecutionResult(
            query=query,
            success=False,
            error=f"Process exited with code {process.exitcode} but produced no result",
            execution_time=execution_time,
            problem_id=problem_id,
            inference_failure_count=inf_failures,
            inference_total=inf_total,
            proxy_calls=proxy_calls or None,
            status=SandboxProblemStatus.FAILED,
        )
    finally:
        result_queue.close()


def _classify_error_type(
    error_message: Optional[str], status: SandboxProblemStatus
) -> str:
    """Best-effort error type classification from message + status.

    No traceback — sandbox stderr already captured in the logs bundle.
    """
    if status == SandboxProblemStatus.TIMED_OUT:
        return "TimeoutError"
    if not error_message:
        return "UnknownError"
    # Cheap heuristic: error_message often starts with the exception class name
    head = error_message.split(":", 1)[0].strip()
    return head if head.isidentifier() else "RuntimeError"


def _format_single_result(result: ExecutionResult) -> str:
    """Format an ExecutionResult as one JSONL envelope line.

    Always returns a non-empty line — failures emit with dialogue=null so the
    validator's ProgressReporter sees every problem outcome through one channel.
    """
    dialogue: Optional[List[Dict]] = None
    if result.success and isinstance(result.result, list):
        dialogue = result.result
        # Decorate dialogue steps as before — preserves backward compat for
        # downstream consumers that look at extra_info.
        for i, step in enumerate(dialogue):
            step.setdefault("extra_info", {})
            step["extra_info"].setdefault("step", i + 1)
            step["extra_info"].setdefault("query", result.query)
            step["extra_info"].setdefault("timestamp", int(time.time() * 1000))
            if result.problem_id:
                step["extra_info"].setdefault("problem_id", result.problem_id)
        # Stamp per-problem execution time onto the first step only — consumers
        # that look at dialogue[0].extra_info still find it there.
        if dialogue:
            dialogue[0]["extra_info"].setdefault("execution_time", result.execution_time)
        # Distribute proxy calls across steps by timestamp approximation.
        # Calls before the first step go to step 0; calls between step N and
        # N+1 go to step N.
        if result.proxy_calls and dialogue:
            step_timestamps = [
                s["extra_info"].get("timestamp", 0) for s in dialogue
            ]
            # Per-attempt entries (kind="attempt") are diagnostic only — keep
            # them out of the trajectory the judge sees.
            summary_calls = [
                c for c in result.proxy_calls if c.get("kind") != "attempt"
            ]
            buckets: Dict[int, List[Dict]] = {}
            for call in summary_calls:
                call_ts = call.get("timestamp", 0)
                target = 0
                for i, st in enumerate(step_timestamps):
                    if st <= call_ts:
                        target = i
                    else:
                        break
                buckets.setdefault(target, []).append(call)
            for idx, calls in buckets.items():
                dialogue[idx]["extra_info"]["proxy_calls"] = calls

    error_obj: Optional[Dict] = None
    if not result.success and result.error:
        error_obj = {
            "type": _classify_error_type(result.error, result.status),
            "message": result.error,
        }

    envelope = {
        "problem_id": result.problem_id,
        "status": result.status.value,
        "execution_time": result.execution_time,
        "inference_failure_count": result.inference_failure_count or 0,
        "inference_total": result.inference_total or 0,
        "error": error_obj,
        "dialogue": dialogue,
    }
    return json.dumps(envelope)


def execute_problems_parallel(
    problems: List[Dict],
    max_workers: Optional[int] = None,
    timeout_per_problem: float = 300.0,
    agent_file: Optional[str] = None,
    output_file: Optional[str] = None,
) -> List[ExecutionResult]:
    """Execute multiple problems in parallel, writing results incrementally.

    Each problem runs in its own child process (via ``execute_single_problem``)
    so that timeouts reliably kill the work.  A ``ThreadPoolExecutor`` controls
    concurrency — at most *max_workers* processes run simultaneously.

    When *output_file* is provided, each result is appended and flushed
    as its future completes, enabling real-time progress monitoring.
    """
    if not problems:
        logger.warning("No problems to execute")
        return []

    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    if agent_file:
        logger.info(f"Using agent from {agent_file}")
    else:
        logger.info("Using default agent from src.agent.agent")

    logger.info(
        f"Executing {len(problems)} problems with {max_workers} workers, timeout={timeout_per_problem}s"
    )

    fout = None
    if output_file:
        from pathlib import Path

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        fout = open(output_file, "a", encoding="utf-8")

    results: List[ExecutionResult] = []
    start_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_problem = {
                executor.submit(
                    execute_single_problem, problem, timeout_per_problem, agent_file
                ): problem
                for problem in problems
            }

            completed = 0
            for future in as_completed(future_to_problem):
                completed += 1
                try:
                    result = future.result(timeout=timeout_per_problem + 30)
                    results.append(result)
                    status = "OK" if result.success else "FAIL"
                    logger.info(
                        f"[{completed}/{len(problems)}] {status} {result.query[:50]}... "
                        f"({result.execution_time:.2f}s)"
                    )

                    if fout is not None:
                        line = _format_single_result(result)
                        fout.write(line + "\n")
                        fout.flush()

                except FutureTimeoutError:
                    problem = future_to_problem[future]
                    query = problem.get("query", "unknown")
                    problem_id = problem.get("problem_id") or problem.get("id")
                    logger.error(
                        f"Future for problem '{query}' timed out during result retrieval"
                    )
                    results.append(
                        ExecutionResult(
                            query=query,
                            success=False,
                            error="Future timeout during result retrieval",
                            execution_time=timeout_per_problem,
                            problem_id=problem_id,
                            status=SandboxProblemStatus.TIMED_OUT,
                        )
                    )
                except Exception as e:
                    problem = future_to_problem[future]
                    query = problem.get("query", "unknown")
                    problem_id = problem.get("problem_id") or problem.get("id")
                    logger.error(
                        f"Unexpected error retrieving result for '{query}': {e}"
                    )
                    results.append(
                        ExecutionResult(
                            query=query,
                            success=False,
                            error=f"Unexpected error: {str(e)}",
                            execution_time=0.0,
                            problem_id=problem_id,
                            status=SandboxProblemStatus.FAILED,
                        )
                    )
    finally:
        if fout is not None:
            fout.close()

    total_time = time.time() - start_time
    succeeded = sum(1 for r in results if r.success)
    logger.info(
        f"Completed {len(results)} problems in {total_time:.2f}s "
        f"({succeeded} succeeded, {len(results) - succeeded} failed)"
    )

    return results


def format_results(results: List[ExecutionResult]) -> Dict:
    """
    Format execution results for evaluation.

    Args:
        results: List of ExecutionResult objects

    Returns:
        Dictionary with formatted results
    """
    completed = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    total_time = sum(r.execution_time for r in results)
    avg_time = total_time / len(results) if results else 0.0

    return {
        "summary": {
            "total": len(results),
            "completed": len(completed),
            "failed": len(failed),
            "completion_rate": len(completed) / len(results) if results else 0.0,
            "total_execution_time": total_time,
            "average_execution_time": avg_time,
        },
        "results": [
            {
                "query": r.query,
                "success": r.success,
                "result": r.result,
                "error": r.error,
                "execution_time": r.execution_time,
            }
            for r in results
        ],
    }
