"""Shared Docker sandbox utilities for test_runner and validator.

Centralises sandbox image/network configuration, host-path mapping, problem
loading, and Docker command construction so the two call-sites stay in sync.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple


# Docker configuration — shared between test_runner and validator
SANDBOX_IMAGE = os.environ.get(
    "SANDBOX_IMAGE", "ghcr.io/oro-ai/oro/sandbox:latest"
)
SANDBOX_NETWORK = os.environ.get("SANDBOX_NETWORK", "sandbox-network")
HOST_PROJECT_DIR = os.environ.get("HOST_PROJECT_DIR")


def host_path(path: str, workspace_dir: Optional[str] = None) -> str:
    """Map a container-local path to its host equivalent for Docker volume mounts.

    When running inside a container with the Docker socket mounted, volume mount
    paths must reference *host* paths.  ``HOST_PROJECT_DIR`` provides the
    host-side path to the project root.

    Args:
        path: The container-local path to translate.
        workspace_dir: If provided (validator case), strip this prefix from
            *path* before joining with ``HOST_PROJECT_DIR``.  When ``None``
            (test_runner case), the function strips well-known prefixes
            (``/app/``, ``/workspace/``).

    Returns:
        The translated host path, or *path* unchanged when ``HOST_PROJECT_DIR``
        is not set.
    """
    if not HOST_PROJECT_DIR:
        return path

    if workspace_dir is not None:
        if path.startswith(workspace_dir):
            relative = path[len(workspace_dir) :].lstrip("/")
            return str(Path(HOST_PROJECT_DIR) / relative)
        return path

    # test_runner: strip well-known container prefixes
    if path.startswith("/app/"):
        return str(Path(HOST_PROJECT_DIR) / path[len("/app/") :])
    if path.startswith("/workspace/"):
        return str(Path(HOST_PROJECT_DIR) / path[len("/workspace/") :])
    return path


def load_problems(problem_path: Path) -> list[dict]:
    """Load problems from a JSON array or JSONL file.

    Supports both formats so callers don't need to care which one the file
    uses.  Returns an empty list for empty files.
    """
    with open(problem_path) as f:
        content = f.read().strip()
    if not content:
        return []
    # JSON array format (e.g. problem_suite_v1.json)
    if content.startswith("["):
        return json.loads(content)
    # JSONL format (one JSON object per line)
    problems: list[dict] = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            problems.append(json.loads(line))
    return problems


def build_sandbox_command(
    *,
    agent_host_path: str,
    logs_host_path: str,
    problem_file_arg: str,
    output_path: str,
    image: str = SANDBOX_IMAGE,
    network: str = SANDBOX_NETWORK,
    extra_volumes: Optional[List[Tuple[str, str]]] = None,
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
    chutes_access_token: Optional[str] = None,
    agent_container_path: Optional[str] = None,
) -> list[str]:
    """Build a ``docker run`` command for the sandbox container.

    Args:
        agent_host_path: Host path to the agent Python file.  When the agent
            file lives inside the logs directory (validator case), pass an
            empty string and set *agent_container_path* to the path within the
            already-mounted ``/app/logs`` volume — this avoids a separate file
            bind mount which can fail on Docker Desktop for Mac due to
            filesystem caching delays.
        logs_host_path: Host path to the logs directory.
        problem_file_arg: Container-side path to the problem file (passed as
            ``--problem-file`` to ``run_sandbox``).
        output_path: Container-side path where sandbox writes output JSONL.
        image: Docker image to use.
        network: Docker network to attach to.
        extra_volumes: Optional list of ``(host_path, container_path)`` tuples
            mounted read-only.
        max_workers: If set, passed as ``--max-workers`` to ``run_sandbox``.
        chutes_access_token: If set, injected as ``CHUTES_ACCESS_TOKEN`` env var
            so the agent's inference requests use the miner's Chutes token.
        agent_container_path: If set, use this as the ``--agent-file`` path
            inside the container instead of mounting *agent_host_path* to
            ``/app/user_agent.py``.  Useful when the agent file is already
            accessible via the logs volume mount.

    Returns:
        Complete ``docker run`` command as a list of strings.
    """
    effective_agent_path = agent_container_path or "/app/user_agent.py"

    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        network,
        # Resource limits — prevent runaway miner agents from impacting the host
        "--memory",
        "4g",
        "--memory-swap",
        "4g",
        "--pids-limit",
        "256",
        "--ulimit",
        "nofile=1024:1024",
        "--user",
        "1000:1000",
        # Security hardening — minimize container attack surface
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges=true",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=256m",
        "-e",
        "SANDBOX_PROXY_URL=http://proxy:80",
    ]

    # Only mount agent file separately when it's not already in the logs dir
    if not agent_container_path:
        cmd.extend(["-v", f"{agent_host_path}:/app/user_agent.py:ro"])

    cmd.extend(["-v", f"{logs_host_path}:/app/logs"])

    if chutes_access_token:
        cmd.extend(["-e", f"CHUTES_ACCESS_TOKEN={chutes_access_token}"])

    if extra_volumes:
        for host, container in extra_volumes:
            cmd.extend(["-v", f"{host}:{container}:ro"])

    cmd.extend(
        [
            image,
            "python",
            "-m",
            "src.agent.run_sandbox",
            "--agent-file",
            effective_agent_path,
            "--problem-file",
            problem_file_arg,
            "--output",
            output_path,
        ]
    )

    if max_workers is not None:
        cmd.extend(["--max-workers", str(max_workers)])

    if timeout is not None:
        cmd.extend(["--timeout", str(timeout)])

    return cmd
