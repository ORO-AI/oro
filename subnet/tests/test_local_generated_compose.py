from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("model", "image_tag", "search_override", "expected_search_image"),
    [
        (
            "vendor/custom-model",
            "",
            "",
            "ghcr.io/oro-ai/oro/search-server:stable",
        ),
        (
            "",
            "latest",
            "",
            "ghcr.io/oro-ai/oro/search-server:latest",
        ),
        (
            "",
            "latest",
            "ghcr.io/oro-ai/oro/search-server@sha256:exact-pack-image",
            "ghcr.io/oro-ai/oro/search-server@sha256:exact-pack-image",
        ),
    ],
)
def test_compose_supports_existing_miner_command_and_env_file(
    tmp_path,
    model,
    image_tag,
    search_override,
    expected_search_image,
) -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker Compose is not installed")
    env_file = tmp_path / "miner.env"
    env_file.write_text(
        "OPENROUTER_API_KEY=or-test\nCHUTES_API_KEY=ch-test\n"
        f"INFERENCE_PROVIDER=chutes\nSANDBOX_MODEL={model}\n"
        f"IMAGE_TAG={image_tag}\nLOCAL_SEARCH_SERVER_IMAGE={search_override}\n"
    )
    environment = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "OPENROUTER_API_KEY",
            "CHUTES_API_KEY",
            "INFERENCE_PROVIDER",
            "SANDBOX_MODEL",
            "IMAGE_TAG",
            "LOCAL_TEST_IMAGE",
            "LOCAL_SEARCH_SERVER_IMAGE",
        }
    }
    completed = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(env_file),
            "-p",
            "oro-pr42-test",
            "--profile",
            "test",
            "config",
            "--format",
            "json",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    config = json.loads(completed.stdout)
    service = config["services"]["test"]
    proxy = config["services"]["test-proxy"]
    search = config["services"]["test-search-server"]
    assert service["entrypoint"] == [
        "/app/.venv/bin/python",
        "-m",
        "subnet.local_generated_validator",
    ]
    assert service["working_dir"] == "/workspace"
    assert service["image"] == "oro-local-test:local"
    assert service["pull_policy"] == "build"
    assert service["build"]["args"]["RUNTIME_PROFILE"] == "local-test"
    assert service["environment"]["LOCAL_OUTPUT_ROOT"] == "/app/logs/environment-runs"
    assert service["environment"]["OPENROUTER_API_KEY"] == "or-test"
    assert service["environment"]["CHUTES_API_KEY"] == "ch-test"
    assert service["environment"]["INFERENCE_PROVIDER"] == "chutes"
    assert service["environment"]["SANDBOX_MODEL"] == (
        model or "deepseek-ai/DeepSeek-V3.2-TEE"
    )
    assert service["network_mode"] == "service:test-proxy"
    assert proxy["environment"]["SESSION_RUNTIME_HOST"] == "127.0.0.1"
    assert proxy["environment"]["BACKEND_URL"] == "https://api.oroagents.com"
    assert proxy["environment"]["BACKEND_HOST"] == "api.oroagents.com"
    assert "ORO_LOCAL_INFERENCE_MODE" not in proxy["environment"]
    assert "ORO_LOCAL_ALLOWED_MODEL" not in proxy["environment"]
    assert service["environment"]["SEARCH_SERVER_URL"] == (
        "http://test-search-server:5632"
    )
    assert search["image"] == expected_search_image
    assert "platform" not in search
    network = config["networks"]["test-sandbox"]
    assert network["internal"] is True
    assert network["name"] == service["environment"]["SANDBOX_NETWORK"]
    assert "proxy" in proxy["networks"]["test-sandbox"]["aliases"]
    assert not proxy.get("ports")
    assert "ORO_LOCAL_INFERENCE_MODE" not in config["services"]["proxy"]["environment"]
