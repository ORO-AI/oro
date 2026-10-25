"""Feature-flagged, validator-owned environment startup preflight.

The preflight builds a small neutral world in validator code, signs it with a
key generated for this run only, and drives two scripted policies through the
real sandbox -> proxy -> validator session path with stub model and search
clients (zero inference). It checks both verdicts and that each re-grades from
its recorded readings before the validator begins claiming work.
"""

from __future__ import annotations

import base64
import importlib.metadata
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from oro_env_runtime import PACK_FORMAT, RUNTIME_CONTRACT, canonical
from oro_env_runtime.pack import Pack, load_pack, signed_bytes
from src.agent.sandbox_executor import MAX_SANDBOX_OUTPUT_BYTES, read_regular_file

from . import environment_preflight_agent
from .episode_emitter import replay_matches
from .session_registry import SessionRegistry
from .session_service import SessionRuntime


POSITIVE_POLICY = "positive"
NEGATIVE_POLICY = "negative"
INFERENCE_USD_CAP = "0.00"
_PREFLIGHT_NAMESPACE = UUID("840c8091-3cd2-4dd0-9b7a-c7fd47d51c5c")
_KEY_ID = "validator-preflight"
_REQUEST = "Order one blue mug."
_JUDGE = {"model": "preflight-judge", "prompt": "Answer PASS."}
_CATALOG = [
    {"id": "blue-mug", "title": "blue mug"},
    {"id": "red-mug", "title": "red mug"},
]
_TOOLS = [
    {
        "name": "search",
        "handler": "h_search",
        "description": "Search items by words.",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "additionalProperties": False,
            "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 100}},
        },
    },
    {
        "name": "order",
        "handler": "h_order",
        "description": "Order one of an item.",
        "parameters": {
            "type": "object",
            "required": ["id"],
            "additionalProperties": False,
            "properties": {"id": {"type": "string", "maxLength": 100}},
        },
    },
]
_WORLD_CODE = '''
def start(task):
    return None

def h_search(args):
    found = search("titles", args["query"], {"k": 5})
    ids = [h["id"] for h in found["hits"]] if found["ok"] else []
    return {"results": [{"id": i, "title": catalog_get(i)["title"]} for i in ids]}

def h_order(args):
    if catalog_get(args["id"]) == None:
        return {"error": "no such item"}
    basket_set(args["id"], 1)
    request_order(basket_get())
    return {"ordered": args["id"]}

def call(handler, args):
    if handler == "h_search":
        return h_search(args)
    return h_order(args)

def message(text):
    return ""

def turn():
    return None

def can_finish(order):
    return {"ok": True, "observation": {"status": "ordered"}}

def grade(episode, answer):
    end = episode[-1]
    if end["reason"] != "order":
        return {"score": 0.0, "category": "did not finish"}
    if [line["id"] for line in end["order"]["lines"]] != [answer["id"]]:
        return {"score": 0.0, "category": "request not met"}
    check = model_call(answer["judge"]["model"], answer["judge"]["prompt"], {})
    if check["ok"] and check["text"] == "PASS":
        return {"score": 1.0, "category": None}
    return {"score": 0.5, "category": "process requirement missed"}
'''
POLICY_ACTIONS: dict[str, list[dict[str, Any]]] = {
    policy: [
        {"name": "search", "args": {"query": "blue mug"}},
        {"name": "order", "args": {"id": item}},
    ]
    for policy, item in ((POSITIVE_POLICY, "blue-mug"), (NEGATIVE_POLICY, "red-mug"))
}
EXPECTED_VERDICTS = {
    POSITIVE_POLICY: (1.0, None),
    NEGATIVE_POLICY: (0.0, "request not met"),
}

SandboxRunner = Callable[
    [Path, str, Path],
    tuple[Path | None, dict[str, Any]],
]


class EnvironmentPreflightError(RuntimeError):
    """The feature-flagged environment path failed its startup gate."""


class _StubModels:
    """Answers every completion with PASS; the preflight world never decides."""

    def complete(self, model: str, prompt: str, params: dict, timeout: float) -> str:
        return "PASS"

    def decide(self, model: str, state: object, questions: object, params: dict, timeout: float) -> object:
        raise RuntimeError("the preflight world makes no decision calls")


class _StubSearch:
    """Matches every query word against the bundled catalog titles."""

    def ensure(self, index: dict) -> None:
        return None

    def search(self, index_id: str, query: str, params: dict, timeout: float) -> list[tuple[str, float]]:
        words = query.lower().split()
        return [(item["id"], 1.0) for item in _CATALOG if all(w in item["title"].split() for w in words)]


def build_preflight_pack(root: Path) -> Pack:
    """Write the neutral world at ``root``, sign it with a fresh key and load it."""

    answer = {"id": "blue-mug", "judge": _JUDGE}
    files: dict[str, bytes] = {
        "world/catalog.jsonl": b"".join(canonical.dumps(item) + b"\n" for item in _CATALOG),
        "world/tools.json": canonical.dumps({"tools": _TOOLS}),
        "world/code/main.star": _WORLD_CODE.encode(),
        "world/index/titles.postings": canonical.dumps({item["id"]: item["title"] for item in _CATALOG}),
        "world/index/titles.json": canonical.dumps({"archive": "world/index/titles.postings", "settings": {}}),
    }
    for task_id in POLICY_ACTIONS:
        files[f"tasks/{task_id}.json"] = canonical.dumps(
            {"task_id": task_id, "request_text": _REQUEST, "params": {}, "sha256": canonical.digest({})}
        )
        files[f"answers/{task_id}.json"] = canonical.dumps(
            {"task_id": task_id, "answer": answer, "sha256": canonical.digest(answer)}
        )
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    manifest = {
        "pack_format": PACK_FORMAT,
        "runtime_contract": RUNTIME_CONTRACT,
        "pack_id": "validator-preflight",
        "world_id": "validator-preflight",
        "delivery": {"kind": "practice", "id": "validator-preflight"},
        "sandbox": {
            "language": "starlark",
            "implementation": "starlark-pyo3",
            "version": importlib.metadata.version("starlark-pyo3"),
            "generator": "validator-preflight",
        },
        "limits": {
            "max_steps": 4,
            "max_turns": 10,
            "call_ticks": 500,
            "grade_ticks": 2000,
            "max_callstack": 50,
            "max_arg_bytes": 1024,
        },
        "parts": {rel: canonical.sha256_hex(data) for rel, data in files.items()},
    }
    key = Ed25519PrivateKey.generate()
    manifest["signature"] = {
        "algorithm": "ed25519",
        "key_id": _KEY_ID,
        "value": base64.b64encode(key.sign(signed_bytes(manifest))).decode(),
    }
    (root / "manifest.json").write_bytes(canonical.dumps(manifest))
    return load_pack(root, {_KEY_ID: key.public_key()})


def _write_sandbox_inputs(run_dir: Path, starts: dict[str, dict[str, Any]]) -> tuple[Path, Path]:
    agent_path = run_dir / "agent.py"
    problem_path = run_dir / "problems.jsonl"
    (run_dir / "output.jsonl").unlink(missing_ok=True)
    agent_path.write_text(
        Path(environment_preflight_agent.__file__).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    rows = [
        {
            "problem_id": policy,
            "query": start["policy_view"]["query"],
            "category": "environment_preflight",
            "environment": {
                "binding": start["binding"],
                "policy_view": start["policy_view"],
                # This is validator-owned test input, never miner-supplied work.
                "actions": POLICY_ACTIONS[policy],
            },
        }
        for policy, start in starts.items()
    ]
    problem_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (run_dir / "environment_sessions.json").write_text(
        json.dumps(
            {
                "schema_version": "oro.session_bootstrap.v1",
                "sessions": [
                    {"session_id": start["session_id"], "policy_view": start["policy_view"]}
                    for start in starts.values()
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return agent_path, problem_path


def _read_sandbox_results(output_path: Path | None) -> list[dict[str, Any]]:
    if output_path is None or not output_path.exists():
        return []
    try:
        content = read_regular_file(output_path, MAX_SANDBOX_OUTPUT_BYTES)
    except (OSError, ValueError):
        return []
    return [
        json.loads(line)
        for line in content.decode("utf-8").splitlines()
        if line.strip()
    ]


def _sandbox_passed(rows: list[dict[str, Any]]) -> bool:
    return (
        {row.get("problem_id") for row in rows} == set(POLICY_ACTIONS)
        and all(row.get("status") == "SUCCESS" for row in rows)
        and all(int(row.get("inference_total") or 0) == 0 for row in rows)
    )


def environment_preflight_run_id(key: str) -> str:
    """Return the stable evaluation-shaped ID used for logs and sandbox naming."""

    return str(uuid5(_PREFLIGHT_NAMESPACE, key))


def run_environment_preflight(
    *,
    runtime: SessionRuntime,
    run_dir: Path,
    sandbox_runner: SandboxRunner,
) -> dict[str, Any]:
    """Run the scripted policies on the bundled world through the active runtime."""

    preflight_started = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir.chmod(0o777)
    pack = build_preflight_pack(run_dir / "world")
    run_id = environment_preflight_run_id(pack.manifest_sha256)
    registry = SessionRegistry(
        pack,
        episode_nonces={task_id: canonical.sha256_hex(f"{run_id}\0{task_id}".encode()) for task_id in POLICY_ACTIONS},
        model_client=_StubModels(),
        search_client=_StubSearch(),
    )
    installed = False
    try:
        starts = {policy: registry.start(policy) for policy in POLICY_ACTIONS}
        agent_path, problem_path = _write_sandbox_inputs(run_dir, starts)
        runtime.install(registry)
        installed = True
        sandbox_started = time.perf_counter()
        output_path, sandbox_metadata = sandbox_runner(agent_path, run_id, problem_path)
        sandbox_duration_ms = round((time.perf_counter() - sandbox_started) * 1000.0, 3)
        sandbox_rows = _read_sandbox_results(output_path)
        registry.finish_all()
        records = {record["task_id"]: record for record in registry.terminal_results()}

        policies = []
        for policy in POLICY_ACTIONS:
            record = records.get(policy) or {}
            verdict = record.get("verdict") or {}
            finish = (record.get("ledger") or {}).get("episode", [{}])[-1]
            policies.append(
                {
                    "policy": policy,
                    "score": verdict.get("score"),
                    "category": verdict.get("category"),
                    "finish_reason": finish.get("reason"),
                    "steps": record.get("steps"),
                    "replay_matches": bool(record) and replay_matches(pack, record),
                }
            )
        passed = (
            sandbox_metadata.get("exit_code") == 0
            and _sandbox_passed(sandbox_rows)
            and all(
                (p["score"], p["category"]) == EXPECTED_VERDICTS[p["policy"]]
                and p["finish_reason"] == "order"
                and p["steps"] == len(POLICY_ACTIONS[p["policy"]])
                and p["replay_matches"]
                for p in policies
            )
        )
        summary = {
            "schema_version": "oro.environment_preflight.v2",
            "status": "pass" if passed else "fail",
            "run_id": run_id,
            "runtime_contract": RUNTIME_CONTRACT,
            "inference_usd_cap": INFERENCE_USD_CAP,
            "timing_ms": {
                "sandbox": sandbox_duration_ms,
                "total": round((time.perf_counter() - preflight_started) * 1000.0, 3),
            },
            "sandbox": {
                "exit_code": sandbox_metadata.get("exit_code"),
                "problem_count": len(sandbox_rows),
                "inference_call_count": sum(int(row.get("inference_total") or 0) for row in sandbox_rows),
            },
            "policies": policies,
        }
        (run_dir / "environment_preflight_result.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not passed:
            raise EnvironmentPreflightError(json.dumps(summary, sort_keys=True))
        return summary
    finally:
        if installed:
            runtime.clear(registry)
        else:
            registry.close()


__all__ = [
    "EnvironmentPreflightError",
    "NEGATIVE_POLICY",
    "POSITIVE_POLICY",
    "build_preflight_pack",
    "environment_preflight_run_id",
    "run_environment_preflight",
]
