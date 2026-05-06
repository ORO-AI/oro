"""Tests for subnet.sandbox shared utilities."""

import json
import os
from unittest import mock


from subnet.sandbox import (
    host_path,
    load_problems,
    build_sandbox_command,
    attach_title_embeddings,
)


# ---------------------------------------------------------------------------
# host_path()
# ---------------------------------------------------------------------------


class TestHostPath:
    """Tests for host_path() volume-mount translation."""

    def test_no_host_project_dir_returns_unchanged(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            with mock.patch("subnet.sandbox.HOST_PROJECT_DIR", None):
                assert host_path("/app/data/test.jsonl") == "/app/data/test.jsonl"

    def test_app_prefix_stripped(self):
        with mock.patch("subnet.sandbox.HOST_PROJECT_DIR", "/home/user/project"):
            assert (
                host_path("/app/data/test.jsonl")
                == "/home/user/project/data/test.jsonl"
            )

    def test_workspace_prefix_stripped(self):
        with mock.patch("subnet.sandbox.HOST_PROJECT_DIR", "/home/user/project"):
            assert (
                host_path("/workspace/data/test.jsonl")
                == "/home/user/project/data/test.jsonl"
            )

    def test_no_matching_prefix_returns_unchanged(self):
        with mock.patch("subnet.sandbox.HOST_PROJECT_DIR", "/home/user/project"):
            assert host_path("/other/path/file.py") == "/other/path/file.py"

    def test_workspace_dir_param_strips_workspace(self):
        with mock.patch("subnet.sandbox.HOST_PROJECT_DIR", "/home/user/project"):
            result = host_path(
                "/opt/validator/logs/output.jsonl",
                workspace_dir="/opt/validator",
            )
            assert result == "/home/user/project/logs/output.jsonl"

    def test_workspace_dir_param_no_match_returns_unchanged(self):
        with mock.patch("subnet.sandbox.HOST_PROJECT_DIR", "/home/user/project"):
            result = host_path(
                "/other/path/file.py",
                workspace_dir="/opt/validator",
            )
            assert result == "/other/path/file.py"


# ---------------------------------------------------------------------------
# load_problems()
# ---------------------------------------------------------------------------


class TestLoadProblems:
    """Tests for load_problems() JSON/JSONL loader."""

    def test_json_array(self, tmp_path):
        problems = [{"query": "a", "reward": 1}, {"query": "b", "reward": 2}]
        p = tmp_path / "problems.json"
        p.write_text(json.dumps(problems))
        assert load_problems(p) == problems

    def test_jsonl(self, tmp_path):
        lines = [{"query": "a"}, {"query": "b"}]
        p = tmp_path / "problems.jsonl"
        p.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
        assert load_problems(p) == lines

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text("")
        assert load_problems(p) == []

    def test_jsonl_with_blank_lines(self, tmp_path):
        p = tmp_path / "problems.jsonl"
        p.write_text('{"query": "a"}\n\n{"query": "b"}\n\n')
        result = load_problems(p)
        assert len(result) == 2
        assert result[0]["query"] == "a"
        assert result[1]["query"] == "b"


# ---------------------------------------------------------------------------
# build_sandbox_command()
# ---------------------------------------------------------------------------


class TestBuildSandboxCommand:
    """Tests for build_sandbox_command() Docker command builder."""

    def test_base_structure(self):
        cmd = build_sandbox_command(
            agent_host_path="/host/agent.py",
            logs_host_path="/host/logs",
            problem_file_arg="/tmp/problems.jsonl",
            output_path="/app/logs/output.jsonl",
        )
        assert cmd[:3] == ["docker", "run", "--rm"]
        assert "--network" in cmd
        assert "/app/user_agent.py" in cmd
        assert "--problem-file" in cmd
        assert "/tmp/problems.jsonl" in cmd
        assert "--output" in cmd
        assert "/app/logs/output.jsonl" in cmd

    def test_no_enable_scoring_flag(self):
        cmd = build_sandbox_command(
            agent_host_path="/host/agent.py",
            logs_host_path="/host/logs",
            problem_file_arg="/tmp/problems.jsonl",
            output_path="/app/logs/output.jsonl",
        )
        assert "--enable-scoring" not in cmd

    def test_extra_volumes_mounted_readonly(self):
        cmd = build_sandbox_command(
            agent_host_path="/host/agent.py",
            logs_host_path="/host/logs",
            problem_file_arg="/tmp/problems.jsonl",
            output_path="/app/logs/output.jsonl",
            extra_volumes=[("/host/data", "/app/data")],
        )
        assert "-v" in cmd
        assert "/host/data:/app/data:ro" in cmd

    def test_max_workers_included(self):
        cmd = build_sandbox_command(
            agent_host_path="/host/agent.py",
            logs_host_path="/host/logs",
            problem_file_arg="/tmp/problems.jsonl",
            output_path="/app/logs/output.jsonl",
            max_workers=5,
        )
        idx = cmd.index("--max-workers")
        assert cmd[idx + 1] == "5"

    def test_max_workers_omitted_when_none(self):
        cmd = build_sandbox_command(
            agent_host_path="/host/agent.py",
            logs_host_path="/host/logs",
            problem_file_arg="/tmp/problems.jsonl",
            output_path="/app/logs/output.jsonl",
        )
        assert "--max-workers" not in cmd

    def test_custom_image_and_network(self):
        cmd = build_sandbox_command(
            agent_host_path="/host/agent.py",
            logs_host_path="/host/logs",
            problem_file_arg="/tmp/problems.jsonl",
            output_path="/app/logs/output.jsonl",
            image="custom:latest",
            network="custom-net",
        )
        net_idx = cmd.index("--network")
        assert cmd[net_idx + 1] == "custom-net"
        assert "custom:latest" in cmd

    def test_resource_limits_present(self):
        """Verify Docker resource limits are included in the command."""
        cmd = build_sandbox_command(
            agent_host_path="/host/agent.py",
            logs_host_path="/host/logs",
            problem_file_arg="/tmp/problems.jsonl",
            output_path="/app/logs/output.jsonl",
        )
        # Memory
        mem_idx = cmd.index("--memory")
        assert cmd[mem_idx + 1] == "4g"
        swap_idx = cmd.index("--memory-swap")
        assert cmd[swap_idx + 1] == "4g"
        # PIDs
        pids_idx = cmd.index("--pids-limit")
        assert cmd[pids_idx + 1] == "256"
        # File descriptors
        ulimit_idx = cmd.index("--ulimit")
        assert cmd[ulimit_idx + 1] == "nofile=1024:1024"
        # Non-root user
        user_idx = cmd.index("--user")
        assert cmd[user_idx + 1] == "1000:1000"
        # CPU shares — sandbox at half the default so validator preempts under contention
        shares_idx = cmd.index("--cpu-shares")
        assert cmd[shares_idx + 1] == "512"

    def test_chutes_access_token_injected(self):
        cmd = build_sandbox_command(
            agent_host_path="/host/agent.py",
            logs_host_path="/host/logs",
            problem_file_arg="/tmp/problems.jsonl",
            output_path="/app/logs/output.jsonl",
            chutes_access_token="miner-token-abc",
        )
        # Token should be injected as both CHUTES_ACCESS_TOKEN and INFERENCE_ACCESS_TOKEN env vars
        assert "CHUTES_ACCESS_TOKEN=miner-token-abc" in cmd
        assert "INFERENCE_ACCESS_TOKEN=miner-token-abc" in cmd

    def test_chutes_access_token_omitted_when_none(self):
        cmd = build_sandbox_command(
            agent_host_path="/host/agent.py",
            logs_host_path="/host/logs",
            problem_file_arg="/tmp/problems.jsonl",
            output_path="/app/logs/output.jsonl",
        )
        assert not any("CHUTES_ACCESS_TOKEN" in arg for arg in cmd)

    def test_chutes_access_token_omitted_when_empty(self):
        cmd = build_sandbox_command(
            agent_host_path="/host/agent.py",
            logs_host_path="/host/logs",
            problem_file_arg="/tmp/problems.jsonl",
            output_path="/app/logs/output.jsonl",
            chutes_access_token="",
        )
        assert not any("CHUTES_ACCESS_TOKEN" in arg for arg in cmd)

    def test_inference_provider_injected(self):
        cmd = build_sandbox_command(
            agent_host_path="/host/agent.py",
            logs_host_path="/host/logs",
            problem_file_arg="/tmp/problems.jsonl",
            output_path="/app/logs/output.jsonl",
            inference_provider="chutes",
        )
        assert "INFERENCE_PROVIDER=chutes" in cmd

    def test_inference_provider_omitted_when_none(self):
        cmd = build_sandbox_command(
            agent_host_path="/host/agent.py",
            logs_host_path="/host/logs",
            problem_file_arg="/tmp/problems.jsonl",
            output_path="/app/logs/output.jsonl",
        )
        assert not any("INFERENCE_PROVIDER" in arg for arg in cmd)

    def test_inference_base_url_injected(self):
        cmd = build_sandbox_command(
            agent_host_path="/host/agent.py",
            logs_host_path="/host/logs",
            problem_file_arg="/tmp/problems.jsonl",
            output_path="/app/logs/output.jsonl",
            inference_base_url="https://api.example.com",
        )
        assert "INFERENCE_BASE_URL=https://api.example.com" in cmd

    def test_inference_base_url_omitted_when_none(self):
        cmd = build_sandbox_command(
            agent_host_path="/host/agent.py",
            logs_host_path="/host/logs",
            problem_file_arg="/tmp/problems.jsonl",
            output_path="/app/logs/output.jsonl",
        )
        assert not any("INFERENCE_BASE_URL" in arg for arg in cmd)

    def test_all_inference_params_together(self):
        cmd = build_sandbox_command(
            agent_host_path="/host/agent.py",
            logs_host_path="/host/logs",
            problem_file_arg="/tmp/problems.jsonl",
            output_path="/app/logs/output.jsonl",
            chutes_access_token="test-token-123",
            inference_provider="custom",
            inference_base_url="https://custom.example.com",
        )
        assert "CHUTES_ACCESS_TOKEN=test-token-123" in cmd
        assert "INFERENCE_ACCESS_TOKEN=test-token-123" in cmd
        assert "INFERENCE_PROVIDER=custom" in cmd
        assert "INFERENCE_BASE_URL=https://custom.example.com" in cmd

    def test_security_hardening_flags_present(self):
        """Verify Docker security hardening flags are included."""
        cmd = build_sandbox_command(
            agent_host_path="/host/agent.py",
            logs_host_path="/host/logs",
            problem_file_arg="/tmp/problems.jsonl",
            output_path="/app/logs/output.jsonl",
        )
        # Drop all Linux capabilities
        assert "--cap-drop=ALL" in cmd
        # Prevent privilege escalation via setuid
        sec_idx = cmd.index("--security-opt")
        assert cmd[sec_idx + 1] == "no-new-privileges=true"
        # Read-only root filesystem
        assert "--read-only" in cmd
        # Writable /tmp with noexec (Python tempfile needs /tmp)
        tmpfs_idx = cmd.index("--tmpfs")
        assert "/tmp:rw,noexec,nosuid,size=256m" in cmd[tmpfs_idx + 1]


# ---------------------------------------------------------------------------
# attach_title_embeddings()
# ---------------------------------------------------------------------------


class TestAttachTitleEmbeddings:
    """Tests for attach_title_embeddings() shared utility."""

    def test_dict_reward_gets_embeddings(self):
        reward = {"product_id": "123", "title": "Test"}
        embeddings = [0.1, 0.2, 0.3]
        attach_title_embeddings(reward, embeddings)
        assert reward["_title_embeddings"] == [0.1, 0.2, 0.3]

    def test_list_reward_all_dicts_get_embeddings(self):
        reward = [{"product_id": "1"}, {"product_id": "2"}]
        embeddings = [0.4, 0.5]
        attach_title_embeddings(reward, embeddings)
        assert reward[0]["_title_embeddings"] == [0.4, 0.5]
        assert reward[1]["_title_embeddings"] == [0.4, 0.5]

    def test_list_reward_skips_non_dict_items(self):
        reward = [{"product_id": "1"}, "not-a-dict", None]
        embeddings = [0.1]
        attach_title_embeddings(reward, embeddings)
        assert reward[0]["_title_embeddings"] == [0.1]
        assert reward[1] == "not-a-dict"

    def test_none_embeddings_is_noop(self):
        reward = {"product_id": "123"}
        attach_title_embeddings(reward, None)
        assert "_title_embeddings" not in reward

    def test_empty_list_embeddings_is_noop(self):
        reward = {"product_id": "123"}
        attach_title_embeddings(reward, [])
        assert "_title_embeddings" not in reward

    def test_non_dict_non_list_reward_is_noop(self):
        reward = "just-a-string"
        attach_title_embeddings(reward, [0.1, 0.2])
        # No crash, no mutation possible on a string
