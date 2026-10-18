"""download_agent must save the agent source byte-for-byte."""

import importlib.util
from types import SimpleNamespace

from requests.models import Response

from validator import main
from validator.main import Validator

# Non-ASCII string keys: a mis-decoded file turns each into several
# characters and str.maketrans raises at import.
SOURCE = (
    '"""Spine (FRAME→DISCOVER→VERIFY) • notes ↔ harness — draft."""\n'
    'TABLE = str.maketrans({"“": \'"\', "″": \'"\', "’": "\'"})\n'
    "def agent_main(problem_data):\n"
    "    return []\n"
).encode("utf-8")


def test_download_agent_writes_raw_bytes(tmp_path, monkeypatch):
    response = Response()
    response.status_code = 200
    response._content = SOURCE
    # S3 sends no charset; force the kind of wrong guess requests can make.
    response.encoding = "mac_iceland"
    monkeypatch.setattr(main.requests, "get", lambda url, timeout: response)
    validator = Validator.__new__(Validator)
    validator.config = SimpleNamespace(workspace_dir=str(tmp_path))

    agent_path = validator.download_agent("https://example.invalid/agent.py", "run-1")

    assert agent_path.read_bytes() == SOURCE
    spec = importlib.util.spec_from_file_location("downloaded_agent", agent_path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_eval_dir_is_world_writable_and_sticky(tmp_path):
    validator = Validator.__new__(Validator)
    validator.config = SimpleNamespace(workspace_dir=str(tmp_path))

    assert validator._eval_dir("run-1").stat().st_mode & 0o7777 == 0o1777
