"""A tiny signed world pack and stub outside services for validator tests.

``build_world_pack(dest)`` writes a signed pack directory (two tasks, three tools,
one search index); ``archive_pack(pack_dir, path)`` tars it with ``manifest.json``
at the root. Packs are signed with ``PRIVATE_KEY``; load them with ``KEYRING``
(or the ``world_keyring`` fixture, which installs it as ``env_pack_loader.KEYRING``).

The world: ``start`` opens with the task's greeting; ``find`` searches the
``title`` index; ``ask`` messages the shopper (a ``complete`` reading); ``buy``
orders one listing and finishes. The grader scores 1.0 if the bought listing is
the task's answer, "request not met" otherwise and "did not finish" without an order.
"""

from __future__ import annotations

import base64
import importlib
import importlib.metadata
import io
import json
import tarfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from oro_env_runtime import RUNTIME_CONTRACT, canonical, pack as runtime_pack

KEY_ID = "validator-test-key"
PRIVATE_KEY = Ed25519PrivateKey.generate()
KEYRING = {KEY_ID: PRIVATE_KEY.public_key()}

CODE = '''
def start(task):
    return {"greeting": task["params"]["greeting"]}

def call(tool, args):
    if tool == "h_find":
        found = search("title", args["query"], {"k": 5})
        return {"ok": found["ok"], "ids": [h["id"] for h in found["hits"]] if found["ok"] else []}
    if tool == "h_ask":
        return {"reply": send_message(args["text"])}
    request_order([{"id": args["id"], "quantity": 1}])
    return {"submitted": args["id"]}

def message(text):
    reading = model_call("shopper-model", "Shopper, answer: " + text, {"temperature": 0})
    return reading["text"] if reading["ok"] else "(no reply)"

def turn():
    pass

def can_finish(order):
    return {"ok": True, "observation": {"status": "ordered"}}

def grade(episode, answer):
    end = episode[-1]
    if end["reason"] != "order":
        return {"score": 0.0, "category": "did not finish"}
    if end["order"]["lines"][0]["id"] == answer["id"]:
        return {"score": 1.0, "category": None}
    return {"score": 0.0, "category": "request not met"}
'''


def _tool(name: str, description: str, prop: str) -> dict:
    return {
        "name": name,
        "description": description,
        "handler": f"h_{name}",
        "parameters": {
            "type": "object",
            "properties": {prop: {"type": "string"}},
            "required": [prop],
            "additionalProperties": False,
        },
    }


TOOLS = [
    _tool("find", "Search listings by title.", "query"),
    _tool("ask", "Message the shopper; the reply is the result.", "text"),
    _tool("buy", "Order one listing and finish.", "id"),
]
CATALOG = [{"id": "a", "title": "red mug"}, {"id": "b", "title": "blue mug"}]
TASKS = {"t1": {"greeting": "hello"}, "t2": {"greeting": "hi there"}}
ANSWERS = {"t1": {"id": "a"}, "t2": {"id": "b"}}
SEARCH_SETTINGS = {"bm25": {"k1": 0.9, "b": 0.4}}


def manifest(*, delivery_id: str, delivery_kind: str, max_steps: int) -> dict:
    return {
        "pack_format": 1,
        "runtime_contract": RUNTIME_CONTRACT,
        "pack_id": "pack-validator-test",
        "world_id": "world-validator-test",
        "delivery": {"kind": delivery_kind, "id": delivery_id},
        "sandbox": {
            "language": "starlark",
            "implementation": "starlark-pyo3",
            "version": importlib.metadata.version("starlark-pyo3"),
            "generator": "hand-written",
        },
        "limits": {
            "max_steps": max_steps,
            "max_turns": 4 * max_steps,
            "call_ticks": 500,
            "grade_ticks": 500,
            "max_callstack": 40,
            "max_arg_bytes": 4096,
        },
    }


def build_world_pack(
    dest: Path,
    *,
    delivery_id: str = "practice-1",
    delivery_kind: str = "practice",
    max_steps: int = 6,
    private_key: Ed25519PrivateKey = PRIVATE_KEY,
    key_id: str = KEY_ID,
) -> Path:
    """Write the test world's signed pack at ``dest/pack``, signed with the public runtime's ``signed_bytes``
    (as ``validator.environment_preflight.build_preflight_pack`` does), and return that directory."""

    files = {
        "world/code/main.star": CODE.encode(),
        "world/tools.json": json.dumps({"tools": TOOLS}).encode(),
        "world/catalog.jsonl": "".join(json.dumps(row) + "\n" for row in CATALOG).encode(),
        "world/index/title.archive": json.dumps({row["id"]: row["title"] for row in CATALOG}).encode(),
        "world/index/title.json": json.dumps(
            {"archive": "world/index/title.archive", "settings": SEARCH_SETTINGS}
        ).encode(),
    }
    for kind, docs, key in (("tasks", TASKS, "params"), ("answers", ANSWERS, "answer")):
        for task_id, body in docs.items():
            doc = {"task_id": task_id, key: body, "sha256": canonical.digest(body)}
            if kind == "tasks":
                doc["request_text"] = f"Buy the item for {task_id}."
            files[f"{kind}/{task_id}.json"] = canonical.dumps(doc)
    out = Path(dest) / "pack"
    for rel, data in files.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(data)
    signed = manifest(delivery_id=delivery_id, delivery_kind=delivery_kind, max_steps=max_steps)
    signed["parts"] = {rel: canonical.sha256_hex(data) for rel, data in files.items()}
    signed["signature"] = {
        "algorithm": "ed25519",
        "key_id": key_id,
        "value": base64.b64encode(private_key.sign(runtime_pack.signed_bytes(signed))).decode(),
    }
    (out / "manifest.json").write_bytes(canonical.dumps(signed))
    return out


def archive_pack(pack_dir: Path, archive: Path, *, extra: dict[str, bytes] | None = None) -> Path:
    """Tar.gz ``pack_dir`` with its files at the archive root; ``extra`` adds raw members."""

    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(Path(pack_dir).rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(pack_dir).as_posix())
        for name, data in (extra or {}).items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return archive


def load_world_pack(dest: Path, **kwargs) -> runtime_pack.Pack:
    return runtime_pack.load_pack(build_world_pack(dest, **kwargs), KEYRING)


class StubModelClient:
    """``complete`` echoes a fixed shopper reply; ``decide`` echoes its questions."""

    def __init__(self, reply: str = "I want the red one.") -> None:
        self.reply = reply
        self.calls: list[tuple] = []

    def complete(self, model, prompt, params, timeout):
        self.calls.append(("complete", model, prompt, params))
        return self.reply

    def decide(self, model, state, questions, params, timeout):
        self.calls.append(("decide", model, state, questions, params))
        return {"answers": questions}


class StubSearchClient:
    """Index archives are JSON ``{id: title}``; a hit matches every query word."""

    def __init__(self) -> None:
        self.titles: dict[str, dict[str, str]] = {}
        self.calls: list[tuple] = []

    def ensure(self, index: dict) -> None:
        self.titles[index["id"]] = json.loads(Path(index["archive"]).read_text())

    def search(self, index_id, query, params, timeout):
        self.calls.append((index_id, query, params))
        words = query.lower().split()
        return [(i, float(len(words))) for i, title in sorted(self.titles[index_id].items())
                if all(w in title.split() for w in words)][: params.get("k", 10)]


@pytest.fixture
def world_keyring(monkeypatch):
    """Trust the test key in the loader (under both import paths) for this test."""
    for name in ("validator.env_pack_loader", "subnet.validator.env_pack_loader"):
        monkeypatch.setattr(importlib.import_module(name), "KEYRING", KEYRING)
    return KEYRING
