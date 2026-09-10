"""Local adapter for the production generated-environment validator path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from subnet import local_report
from subnet.inference import resolve_inference_credentials
from subnet.sandbox import build_sandbox_command, host_path
from subnet.validator.env_pack_loader import (
    LoadedPack,
    PackValidationError,
    load_local_pack,
)
from subnet.validator.episode_emitter import (
    build_episode_artifact,
    serialize_episode_artifact,
)
from subnet.validator.generated_evaluation import aggregate_results, write_problem_file
from subnet.validator.session_registry import SessionRegistry
from subnet.validator.session_service import SessionRuntime, SessionServer

GENERATED_FAMILIES = frozenset(
    {
        "intent_decomposition",
        "retrieval_recall",
        "constraint_satisfaction",
        "preference_reasoning",
        "ranking",
        "recovery",
        "justification",
    }
)
QUALIFYING_TASKS_PER_FAMILY = 5
LOCAL_SUMMARY_SCHEMA = "oro.local_generated_summary.v1"
SANDBOX_RESULT_GRACE_SECONDS = 60.0
MAX_SANDBOX_OUTPUT_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class LocalGeneratedConfig:
    agent_path: Path
    pack_path: Path
    output_root: Path
    inference_access_token: str
    inference_provider: str
    inference_base_url: str
    model: str
    pack_sha256: str | None = None
    problem_count: int | None = None
    seed: int | None = None
    max_workers: int = 7
    timeout: float = 1800.0
    session_host: str = "0.0.0.0"
    session_port: int = 9101


@dataclass(frozen=True)
class LocalGeneratedResult:
    run_id: str
    aggregate_score: float
    results: list[dict[str, Any]]
    artifact_dir: Path
    summary_path: Path
    summary: dict[str, Any]
    report_path: Path | None = None

    @property
    def score(self) -> float:
        """Compatibility alias for callers that display a generic score."""

        return self.aggregate_score


class LocalGeneratedValidatorError(RuntimeError):
    """A local generated run failed outside normal agent task outcomes."""

    def __init__(
        self,
        message: str,
        *,
        classification: str,
        artifact_dir: Path,
        summary_path: Path,
    ) -> None:
        super().__init__(message)
        self.classification = classification
        self.artifact_dir = artifact_dir
        self.summary_path = summary_path


def _sample_problem_ids(pack: LoadedPack, count: int, seed: int | None) -> list[str]:
    """Pick ``count`` task ids at random, spread as evenly across families as possible."""

    rng = random.Random(seed)
    by_family: dict[str, list[int]] = defaultdict(list)
    for index, task in enumerate(pack.task_specs):
        by_family[task.family].append(index)

    families = sorted(by_family)
    rng.shuffle(families)
    for indexes in by_family.values():
        rng.shuffle(indexes)

    chosen: list[int] = []
    deepest = max(len(indexes) for indexes in by_family.values())
    for depth in range(deepest):
        for family in families:
            indexes = by_family[family]
            if depth < len(indexes):
                chosen.append(indexes[depth])
                if len(chosen) == count:
                    return [pack.task_ids[index] for index in sorted(chosen)]
    return [pack.task_ids[index] for index in sorted(chosen)]


def validate_local_pack(
    pack: LoadedPack,
    expected_families: frozenset[str] = GENERATED_FAMILIES,
    tasks_per_family: int = QUALIFYING_TASKS_PER_FAMILY,
    *,
    problem_count: int | None = None,
    seed: int | None = None,
) -> list[str]:
    """Validate the family roster, then choose which problems to run.

    Without ``problem_count`` this selects the qualifying roster: the first
    ``tasks_per_family`` of every family, in archive order. With it, that many
    problems are sampled at random and spread across families, so a short run
    still covers as many of TF1 through TF7 as it has room for.
    """

    if problem_count is None and tasks_per_family <= 0:
        raise ValueError("tasks_per_family must be positive")

    counts = Counter(task.family for task in pack.task_specs)
    missing = sorted(expected_families - set(counts))
    unexpected = sorted(set(counts) - expected_families)
    insufficient = (
        []
        if problem_count is not None
        else sorted(
            f"{family}:{counts[family]}"
            for family in expected_families
            if counts[family] < tasks_per_family
        )
    )
    if missing or unexpected or insufficient:
        details = [f"task_count={len(pack.task_specs)}"]
        if missing:
            details.append(f"missing={','.join(missing)}")
        if unexpected:
            details.append(f"unexpected={','.join(unexpected)}")
        if insufficient:
            details.append(f"insufficient={','.join(insufficient)}")
        raise ValueError("invalid local generated pack: " + "; ".join(details))

    if problem_count is None:
        selected: list[str] = []
        selected_counts: Counter[str] = Counter()
        for task_id, task in zip(pack.task_ids, pack.task_specs, strict=True):
            if selected_counts[task.family] < tasks_per_family:
                selected.append(task_id)
                selected_counts[task.family] += 1
        return selected

    available = len(pack.task_ids)
    if not 1 <= problem_count <= available:
        raise ValueError(f"--problems must be between 1 and {available}")
    return _sample_problem_ids(pack, problem_count, seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sandbox_rows(path: Path) -> list[dict[str, Any]]:
    try:
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent_fd,
            )
        finally:
            os.close(parent_fd)
        with os.fdopen(fd, "rb") as source:
            metadata = os.fstat(source.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError("sandbox output must be a regular file with one link")
            if metadata.st_size > MAX_SANDBOX_OUTPUT_BYTES:
                raise ValueError("sandbox output exceeds size limit")
            content = source.read(MAX_SANDBOX_OUTPUT_BYTES + 1)
        if len(content) > MAX_SANDBOX_OUTPUT_BYTES:
            raise ValueError("sandbox output exceeds size limit")
        rows = [
            json.loads(line)
            for line in content.decode("utf-8").splitlines()
            if line.strip()
        ]
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("sandbox output rows must be objects")
        return rows
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"could not read sandbox output: {type(exc).__name__}"
        ) from exc


def _error_classification(result: dict[str, Any]) -> str | None:
    outcome = str(result.get("outcome") or "")
    if outcome == "completed":
        return None
    if outcome == "verifier_error":
        return "verifier"
    if outcome == "environment_error":
        return "environment"
    if outcome in {"leakage", "exploit"}:
        return "infrastructure"
    return "agent"


def _summary_tasks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = []
    for result in results:
        verdict = result.get("verdict") or {}
        correct = bool(verdict.get("correct"))
        tasks.append(
            {
                "task_id": str(result.get("task_id") or ""),
                "family": result.get("family"),
                "outcome": result.get("outcome"),
                "correct": correct,
                "reward": float(verdict.get("paid_reward") or 0) if correct else 0.0,
                "error_classification": _error_classification(result),
                "error_detail": result.get("error_detail"),
            }
        )
    return tasks


def _write_summary(
    path: Path,
    *,
    run_id: str,
    pack: LoadedPack | None,
    configured_pack_sha256: str | None,
    task_ids: list[str],
    results: list[dict[str, Any]],
    aggregate_score: float | None,
    status: str,
    seed: int | None = None,
    error: LocalGeneratedValidatorError | None = None,
) -> dict[str, Any]:
    task_rows = _summary_tasks(results)
    family_counts: Counter[str] = Counter()
    family_totals: Counter[str] = Counter()
    for task in task_rows:
        family = task["family"]
        if family is not None:
            family = str(family)
            family_counts[family] += 1
            family_totals[family] += float(task["reward"])
    summary = {
        "schema_version": LOCAL_SUMMARY_SCHEMA,
        "run_id": run_id,
        "status": status,
        "pack_sha256": (
            pack.pack_sha256 if pack is not None else configured_pack_sha256
        ),
        "task_count": len(results),
        "task_roster": task_ids,
        "pack_task_count": len(pack.task_ids) if pack is not None else len(results),
        "selection_seed": seed,
        "models": (
            {
                str(role): str(model)
                for role, model in pack.manifest.get("models", {}).items()
            }
            if pack is not None
            else {}
        ),
        "aggregate_score": aggregate_score,
        "tasks": task_rows,
        "family_rewards": {
            family: family_totals[family] / count
            for family, count in family_counts.items()
        },
        "error": (
            None
            if error is None
            else {
                "classification": error.classification,
                "message": str(error),
            }
        ),
    }
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _run_failure_classification(results: list[dict[str, Any]]) -> str:
    outcomes = {str(result.get("outcome") or "") for result in results}
    if "verifier_error" in outcomes:
        return "verifier"
    if "environment_error" in outcomes:
        return "environment"
    return "infrastructure"


def _validate_result_roster(task_ids: list[str], results: list[dict[str, Any]]) -> None:
    expected = Counter(task_ids)
    actual = Counter(str(result.get("task_id") or "") for result in results)
    if actual != expected:
        raise ValueError(
            "local generated run requires one finalized result per pack task"
        )


def _write_episode_results(path: Path, results: list[dict[str, Any]]) -> None:
    artifacts = [
        serialize_episode_artifact(build_episode_artifact(result)) for result in results
    ]
    path.write_bytes(b"\n".join(artifacts) + b"\n")


def _raise_run_error(
    message: str,
    *,
    classification: str,
    artifact_dir: Path,
    summary_path: Path,
) -> None:
    raise LocalGeneratedValidatorError(
        message,
        classification=classification,
        artifact_dir=artifact_dir,
        summary_path=summary_path,
    )


def _classified_error(
    error: Exception,
    *,
    phase: str,
    artifact_dir: Path,
    summary_path: Path,
) -> LocalGeneratedValidatorError:
    if isinstance(error, LocalGeneratedValidatorError):
        return error
    if isinstance(error, PackValidationError):
        classification = "search" if "search" in str(error).lower() else "configuration"
    elif phase == "pack_validation":
        classification = "configuration"
    elif phase == "finalization":
        classification = "verifier"
    else:
        classification = "infrastructure"
    return LocalGeneratedValidatorError(
        str(error) or type(error).__name__,
        classification=classification,
        artifact_dir=artifact_dir,
        summary_path=summary_path,
    )


def _cleanup(
    *,
    container_name: str,
    remove_container: bool,
    server: SessionServer | None,
    runtime: SessionRuntime | None,
    registry: SessionRegistry | None,
    registry_installed: bool,
    pack: LoadedPack | None,
) -> Exception | None:
    """Release local run resources synchronously and return the first error."""

    cleanup_errors: list[Exception] = []
    if remove_container:
        try:
            subprocess.run(
                ["docker", "rm", "--force", container_name],
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            cleanup_errors.append(exc)
    if server is not None:
        try:
            server.stop()
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(exc)
    if registry is not None:
        try:
            if registry_installed and runtime is not None:
                runtime.clear(registry)
            else:
                registry.close()
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(exc)
    elif pack is not None:
        try:
            pack.close()
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(exc)
    return cleanup_errors[0] if cleanup_errors else None


def run_local_generated_validator(
    config: LocalGeneratedConfig,
) -> LocalGeneratedResult:
    """Execute one local EnvPack through the generated validator components."""

    if config.max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if config.timeout <= 0:
        raise ValueError("timeout must be positive")

    run_id = f"local-{uuid4().hex}"
    artifact_dir = config.output_root / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir.chmod(0o755)
    summary_path = artifact_dir / "summary.json"
    summary_path.touch(exist_ok=False)
    sandbox_dir = artifact_dir / "sandbox"
    sandbox_dir.mkdir()
    sandbox_dir.chmod(0o777)
    sandbox_output = sandbox_dir / "sandbox_output.jsonl"
    container_name = f"oro-generated-{run_id}"

    pack: LoadedPack | None = None
    registry: SessionRegistry | None = None
    registry_installed = False
    remove_container = False
    results: list[dict[str, Any]] = []
    task_ids: list[str] = []
    aggregate_score: float | None = None
    completed_result: LocalGeneratedResult | None = None
    caught_error: Exception | None = None
    run_error: LocalGeneratedValidatorError | None = None
    phase = "initialization"
    runtime: SessionRuntime | None = None
    server: SessionServer | None = None
    try:
        _write_summary(
            summary_path,
            run_id=run_id,
            pack=None,
            configured_pack_sha256=config.pack_sha256,
            task_ids=task_ids,
            results=[],
            aggregate_score=None,
            status="running",
        )

        phase = "server_setup"
        runtime = SessionRuntime()
        server = SessionServer(
            runtime,
            host=config.session_host,
            port=config.session_port,
        )

        phase = "pack_load"
        pack = load_local_pack(config.pack_path, config.pack_sha256)

        phase = "pack_validation"
        task_ids = validate_local_pack(
            pack, problem_count=config.problem_count, seed=config.seed
        )

        phase = "session_setup"
        registry = SessionRegistry(
            pack,
            max_workers=config.max_workers,
            inference_access_token=config.inference_access_token,
            simulator_proxy_url="http://127.0.0.1:80",
        )
        agent_identity = _sha256(config.agent_path)
        sessions = []
        for task_id in task_ids:
            sessions.append(
                registry.start(
                    evaluation_run_id=run_id,
                    agent_version_id=agent_identity,
                    task_id=task_id,
                )
            )
        phase = "artifact_write"
        (artifact_dir / "environment_sessions.json").write_text(
            json.dumps(
                {
                    "schema_version": "oro.session_bootstrap.v1",
                    "sessions": sessions,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        problem_file = artifact_dir / "problems.jsonl"
        write_problem_file(problem_file, sessions)

        phase = "server_startup"
        runtime.install(registry)
        registry_installed = True
        server.start(timeout=5.0)

        phase = "sandbox"
        command = build_sandbox_command(
            agent_host_path=host_path(str(config.agent_path.resolve())),
            logs_host_path=host_path(str(artifact_dir.resolve())),
            output_host_path=host_path(str(sandbox_dir.resolve())),
            problem_file_arg="/app/logs/problems.jsonl",
            output_path="/app/output/sandbox_output.jsonl",
            max_workers=config.max_workers,
            timeout=config.timeout,
            inference_access_token=config.inference_access_token,
            inherit_inference_access_token=True,
            inference_provider=config.inference_provider,
            inference_base_url=config.inference_base_url,
            inference_model=config.model,
            inherit_inference_model=True,
            container_name=container_name,
        )
        sandbox_environment = dict(os.environ)
        sandbox_environment.update(
            {
                "INFERENCE_ACCESS_TOKEN": config.inference_access_token,
                "SANDBOX_MODEL": config.model,
            }
        )
        sandbox_failure = None
        sandbox_exception: Exception | None = None
        try:
            try:
                completed = subprocess.run(
                    command,
                    env=sandbox_environment,
                    timeout=config.timeout + SANDBOX_RESULT_GRACE_SECONDS,
                    check=False,
                )
                if completed.returncode != 0:
                    sandbox_failure = f"sandbox exited with code {completed.returncode}"
            except subprocess.TimeoutExpired:
                remove_container = True
                removal = subprocess.run(
                    ["docker", "rm", "--force", container_name],
                    timeout=30,
                    check=False,
                )
                remove_container = removal.returncode != 0
                sandbox_failure = (
                    f"local generated run timed out after {config.timeout:g} seconds"
                )
        except Exception as exc:  # noqa: BLE001
            sandbox_exception = exc
        except BaseException:
            remove_container = True
            raise
        finally:
            phase = "finalization"
            results = registry.finalized_results()
            _write_episode_results(artifact_dir / "episode_results.jsonl", results)

        if sandbox_exception is not None:
            phase = "sandbox"
            raise sandbox_exception

        phase = "result_validation"
        try:
            _validate_result_roster(task_ids, results)
        except ValueError as exc:
            if sandbox_failure:
                _raise_run_error(
                    sandbox_failure,
                    classification="infrastructure",
                    artifact_dir=artifact_dir,
                    summary_path=summary_path,
                )
            _raise_run_error(
                str(exc),
                classification="infrastructure",
                artifact_dir=artifact_dir,
                summary_path=summary_path,
            )

        phase = "aggregation"
        try:
            aggregate_score = aggregate_results(results)
        except ValueError as exc:
            if sandbox_failure:
                _raise_run_error(
                    sandbox_failure,
                    classification="infrastructure",
                    artifact_dir=artifact_dir,
                    summary_path=summary_path,
                )
            _raise_run_error(
                str(exc),
                classification=_run_failure_classification(results),
                artifact_dir=artifact_dir,
                summary_path=summary_path,
            )

        if not sandbox_output.is_file() or (
            sandbox_failure and sandbox_output.stat().st_size == 0
        ):
            _raise_run_error(
                sandbox_failure or "sandbox did not produce output",
                classification="infrastructure",
                artifact_dir=artifact_dir,
                summary_path=summary_path,
            )

        phase = "sandbox_output"
        _read_sandbox_rows(sandbox_output)

        if sandbox_failure:
            _raise_run_error(
                sandbox_failure,
                classification="infrastructure",
                artifact_dir=artifact_dir,
                summary_path=summary_path,
            )

        assert aggregate_score is not None
        phase = "summary"
        summary = _write_summary(
            summary_path,
            run_id=run_id,
            pack=pack,
            configured_pack_sha256=config.pack_sha256,
            task_ids=task_ids,
            results=results,
            aggregate_score=aggregate_score,
            status="completed",
            seed=config.seed,
        )
        report_path = local_report.write_trajectory_report(
            artifact_dir.resolve() / "trajectories.html",
            [build_episode_artifact(result) for result in results],
            run_id=run_id,
        )
        completed_result = LocalGeneratedResult(
            run_id=run_id,
            aggregate_score=aggregate_score,
            results=results,
            artifact_dir=artifact_dir.resolve(),
            summary_path=summary_path.resolve(),
            summary=summary,
            report_path=report_path,
        )
    except Exception as exc:  # noqa: BLE001
        caught_error = exc
        run_error = _classified_error(
            exc,
            phase=phase,
            artifact_dir=artifact_dir,
            summary_path=summary_path,
        )
    finally:
        cleanup_error = _cleanup(
            container_name=container_name,
            remove_container=remove_container,
            server=server,
            runtime=runtime,
            registry=registry,
            registry_installed=registry_installed,
            pack=pack,
        )
    if run_error is None and cleanup_error is not None:
        caught_error = cleanup_error
        run_error = _classified_error(
            cleanup_error,
            phase="cleanup",
            artifact_dir=artifact_dir,
            summary_path=summary_path,
        )

    if run_error is not None:
        try:
            _write_summary(
                summary_path,
                run_id=run_id,
                pack=pack,
                configured_pack_sha256=config.pack_sha256,
                task_ids=task_ids,
                results=results,
                aggregate_score=aggregate_score,
                status="failed",
                seed=config.seed,
                error=run_error,
            )
        except Exception:  # noqa: BLE001, S110
            pass
        if caught_error is run_error:
            raise run_error
        raise run_error from caught_error

    if completed_result is None:
        raise AssertionError("local generated run completed without a result")
    return completed_result


__all__ = [
    "GENERATED_FAMILIES",
    "LOCAL_SUMMARY_SCHEMA",
    "LocalGeneratedConfig",
    "LocalGeneratedResult",
    "LocalGeneratedValidatorError",
    "run_local_generated_validator",
    "validate_local_pack",
]


def parse_config(arguments: list[str] | None = None) -> LocalGeneratedConfig:
    parser = argparse.ArgumentParser(
        description="Run the bundled generated EnvPack locally."
    )
    parser.add_argument("--agent-file", default="src/agent/environment_agent.py")
    parser.add_argument(
        "--problems",
        type=int,
        default=None,
        help=(
            "How many problems to run, sampled at random and spread across the "
            "seven families. Defaults to the full qualifying roster."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reuse a previous run's selection seed to repeat its problems.",
    )
    args = parser.parse_args(arguments)
    if args.problems is not None and args.problems <= 0:
        raise ValueError("--problems must be positive")
    # A fresh sample each run avoids tuning against one lucky subset; the seed is
    # reported so any run can be repeated exactly.
    seed = args.seed
    if args.problems is not None and seed is None:
        seed = random.randrange(2**31)
    agent = Path(args.agent_file).expanduser()
    workspace_agent = Path("/workspace") / agent
    if not agent.is_absolute() and workspace_agent.is_file():
        agent = workspace_agent
    if not agent.is_file():
        raise ValueError(f"agent file does not exist: {args.agent_file}")

    key, provider, base_url = resolve_inference_credentials()
    if not key or not provider or not base_url:
        raise ValueError("set OPENROUTER_API_KEY or CHUTES_API_KEY in .env")
    model = os.environ.get("SANDBOX_MODEL") or "deepseek-ai/DeepSeek-V3.2-TEE"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+-]*", model):
        raise ValueError("SANDBOX_MODEL contains unsupported characters")
    max_workers = int(os.environ.get("LOCAL_MAX_WORKERS") or "7")
    timeout = float(os.environ.get("LOCAL_TIMEOUT") or "1800")
    if max_workers <= 0:
        raise ValueError("LOCAL_MAX_WORKERS must be positive")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("LOCAL_TIMEOUT must be finite and positive")
    root = Path(__file__).resolve().parents[1]
    return LocalGeneratedConfig(
        agent_path=agent.resolve(),
        pack_path=Path(
            os.environ.get("LOCAL_ENV_PACK_PATH")
            or root / "data/local-test/env-pack.tar.gz"
        ),
        output_root=Path(
            os.environ.get("LOCAL_OUTPUT_ROOT") or "logs/environment-runs"
        ),
        inference_access_token=key,
        inference_provider=provider,
        inference_base_url=base_url,
        model=model,
        pack_sha256=os.environ.get("LOCAL_ENV_PACK_SHA256")
        or "9e5d11c6945edc19e06b730afd5681a035f75827933f958e6bfbcc846a28c73a",
        problem_count=args.problems,
        seed=seed,
        max_workers=max_workers,
        timeout=timeout,
        session_host="127.0.0.1",
    )


def main(arguments: list[str] | None = None) -> int:
    try:
        config = parse_config(arguments)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    try:
        result = run_local_generated_validator(config)
    except LocalGeneratedValidatorError as error:
        print(
            f"Error [{error.classification}]: {error}\nSummary: {error.summary_path}",
            file=sys.stderr,
        )
        return 1
    print(
        local_report.render_console_report(
            result.summary,
            artifact_dir=result.artifact_dir,
            report_path=result.report_path,
            provider=config.inference_provider,
            agent_model=config.model,
            color=local_report.stdout_supports_color(),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
