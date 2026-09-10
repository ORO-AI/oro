from __future__ import annotations

import hashlib
import json
import tarfile
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest
from oro_env_runtime import (
    ENV_CONTRACT_VERSION,
    EVENT_CONTRACT_VERSION,
    REPLAY_CONTRACT_VERSION,
    RESULT_SCHEMA_VERSION,
    RUNTIME_VERSION,
    TOOL_CONTRACT_VERSION,
    VERIFIER_VERSION,
)

from subnet import local_generated_validator as local

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PACK_SHA256 = (
    "9e5d11c6945edc19e06b730afd5681a035f75827933f958e6bfbcc846a28c73a"
)


@pytest.fixture(autouse=True)
def inference_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in (
        "OPENROUTER_API_KEY",
        "CHUTES_API_KEY",
        "INFERENCE_PROVIDER",
        "SANDBOX_MODEL",
        "LOCAL_ENV_PACK_PATH",
        "LOCAL_ENV_PACK_SHA256",
        "LOCAL_OUTPUT_ROOT",
        "LOCAL_MAX_WORKERS",
        "LOCAL_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)
    # Keep parse_config tests independent of whether this checkout has the
    # LFS pack; test_bundled_pack_matches_released_runtime_contracts covers it.
    monkeypatch.setenv("LOCAL_ENV_PACK_PATH", str(tmp_path / "env-pack.tar.gz"))


@pytest.mark.parametrize(
    ("openrouter", "chutes", "provider", "expected_key", "expected_provider"),
    [
        ("or-test", "", "", "or-test", "openrouter"),
        ("", "ch-test", "", "ch-test", "chutes"),
        ("or-test", "ch-test", "chutes", "ch-test", "chutes"),
        ("or-test", "ch-test", "openrouter", "or-test", "openrouter"),
        ("or-test", "ch-test", "", "or-test", "openrouter"),
    ],
)
def test_existing_credentials_select_one_provider(
    monkeypatch,
    tmp_path,
    openrouter,
    chutes,
    provider,
    expected_key,
    expected_provider,
) -> None:
    agent = tmp_path / "agent.py"
    agent.write_text("raise AssertionError('agent imported outside sandbox')\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", openrouter)
    monkeypatch.setenv("CHUTES_API_KEY", chutes)
    monkeypatch.setenv("INFERENCE_PROVIDER", provider)
    monkeypatch.setenv("SANDBOX_MODEL", "vendor/custom-model")

    config = local.parse_config(["--agent-file", str(agent)])

    assert config.agent_path == agent
    assert config.inference_access_token == expected_key
    assert config.inference_provider == expected_provider
    assert (
        config.inference_base_url
        == {
            "openrouter": "https://openrouter.ai/api/v1",
            "chutes": "https://llm.chutes.ai/v1",
        }[expected_provider]
    )
    assert config.model == "vendor/custom-model"


def test_model_flag_is_not_required(monkeypatch, tmp_path) -> None:
    agent = tmp_path / "agent.py"
    agent.touch()
    monkeypatch.setenv("CHUTES_API_KEY", "ch-test")

    config = local.parse_config(["--agent-file", str(agent)])

    assert config.model == "deepseek-ai/DeepSeek-V3.2-TEE"
    assert config.pack_path.name == "env-pack.tar.gz"
    assert config.pack_sha256 == EXPECTED_PACK_SHA256
    assert config.max_workers == 7


def test_pack_path_defaults_to_the_bundled_archive(monkeypatch, tmp_path) -> None:
    # Every other parse_config test runs against the fixture's temp pack so the
    # suite does not depend on this checkout's LFS state. That hides the default
    # path, so pin it here with the pointer probe stubbed out.
    agent = tmp_path / "agent.py"
    agent.touch()
    monkeypatch.setenv("CHUTES_API_KEY", "ch-test")
    monkeypatch.delenv("LOCAL_ENV_PACK_PATH", raising=False)
    monkeypatch.setattr(local, "_is_git_lfs_pointer", lambda path: False)

    config = local.parse_config(["--agent-file", str(agent)])

    assert config.pack_path == ROOT / "data" / "local-test" / "env-pack.tar.gz"


def test_bundled_pack_matches_released_runtime_contracts() -> None:
    pack_path = ROOT / "data" / "local-test" / "env-pack.tar.gz"

    assert hashlib.sha256(pack_path.read_bytes()).hexdigest() == EXPECTED_PACK_SHA256
    assert version("oro-env-runtime") == "0.2.11"

    with tarfile.open(pack_path, "r:gz") as archive:
        manifest_file = archive.extractfile("epoch/manifest.json")
        assert manifest_file is not None
        manifest = json.load(manifest_file)

    assert manifest["contracts"] == {
        "environment": ENV_CONTRACT_VERSION,
        "event": EVENT_CONTRACT_VERSION,
        "replay": REPLAY_CONTRACT_VERSION,
        "result": RESULT_SCHEMA_VERSION,
        "runtime": RUNTIME_VERSION,
        "tools": TOOL_CONTRACT_VERSION,
        "verifier": VERIFIER_VERSION,
    }
    family_counts = manifest["epoch"]["family_counts"]
    assert set(family_counts) == local.GENERATED_FAMILIES
    assert all(
        count >= local.QUALIFYING_TASKS_PER_FAMILY for count in family_counts.values()
    )
    assert manifest["epoch"]["tasks"] == sum(family_counts.values())


@pytest.mark.parametrize("model", ['bad"model', "bad$model", "bad;model"])
def test_invalid_model_cannot_reach_proxy_configuration(
    monkeypatch, tmp_path, model
) -> None:
    agent = tmp_path / "agent.py"
    agent.touch()
    monkeypatch.setenv("CHUTES_API_KEY", "ch-test")
    monkeypatch.setenv("SANDBOX_MODEL", model)
    with pytest.raises(ValueError, match="SANDBOX_MODEL"):
        local.parse_config(["--agent-file", str(agent)])


@pytest.mark.parametrize(
    ("name", "value"),
    [("LOCAL_MAX_WORKERS", "0"), ("LOCAL_TIMEOUT", "-1"), ("LOCAL_TIMEOUT", "nan")],
)
def test_invalid_limits_fail_before_evaluation(
    monkeypatch, tmp_path, name, value
) -> None:
    agent = tmp_path / "agent.py"
    agent.touch()
    monkeypatch.setenv("CHUTES_API_KEY", "ch-test")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        local.parse_config(["--agent-file", str(agent)])


def test_missing_credentials_fail_without_running_agent(tmp_path) -> None:
    agent = tmp_path / "agent.py"
    agent.touch()
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY.*CHUTES_API_KEY"):
        local.parse_config(["--agent-file", str(agent)])


def test_missing_agent_is_a_configuration_error(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("CHUTES_API_KEY", "ch-test")
    assert local.main(["--agent-file", str(tmp_path / "missing.py")]) == 2
    assert "agent file does not exist" in capsys.readouterr().err


def test_lfs_pointer_pack_is_a_configuration_error(
    monkeypatch, tmp_path, capsys
) -> None:
    agent = tmp_path / "agent.py"
    agent.write_text("def agent_main(problem_data):\n    return {}\n")
    pointer = tmp_path / "env-pack.tar.gz"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "9" * 64 + "\n"
        "size 6304798\n"
    )
    monkeypatch.setenv("CHUTES_API_KEY", "ch-test")
    monkeypatch.setenv("LOCAL_ENV_PACK_PATH", str(pointer))
    # Point the run directory somewhere observable: the default is CWD-relative,
    # so asserting on tmp_path without this would pass whether or not a run started.
    monkeypatch.setenv("LOCAL_OUTPUT_ROOT", str(tmp_path / "logs"))

    assert local.main(["--agent-file", str(agent)]) == 2
    err = capsys.readouterr().err
    assert "Git LFS pointer" in err
    assert "git lfs pull" in err
    assert not (tmp_path / "logs").exists()


def test_cli_prints_generated_results(monkeypatch, tmp_path, capsys) -> None:
    agent = tmp_path / "agent.py"
    agent.touch()
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setattr(
        local,
        "run_local_generated_validator",
        lambda config: SimpleNamespace(
            results=[
                {
                    "family": "recovery",
                    "outcome": "completed",
                    "verdict": {
                        "correct": True,
                        "paid_reward": 0.5,
                    },
                }
            ],
            aggregate_score=0.5,
            artifact_dir=tmp_path / "results",
        ),
    )
    assert local.main(["--agent-file", str(agent)]) == 0
    output = capsys.readouterr().out
    assert "recovery: completed, reward=0.500000" in output
    assert "Aggregate score: 0.500000" in output
    assert f"Artifacts: {tmp_path / 'results'}" in output
