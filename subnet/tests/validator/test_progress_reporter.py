"""Tests for ProgressReporter module.

Uses mock_backend_client and temp_output_file fixtures from conftest.py.
"""

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add test-subnet to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from oro_sdk.models import ProblemProgressUpdate, ProblemStatus

from validator.backend_client import BackendError
from validator.progress_reporter import ProgressReporter, extract_inference_stats
from validator.retry_queue import LocalRetryQueue


@pytest.fixture
def sample_problems():
    """Sample problem list for testing."""
    return [
        {
            "problem_id": "11111111-1111-1111-1111-111111111111",
            "query": "Find a laptop under $1000",
            "category": "product",
            "reward": {"field_1": "laptop"},
        },
        {
            "problem_id": "22222222-2222-2222-2222-222222222222",
            "query": "Find a smartphone with good camera",
            "category": "product",
            "reward": {"field_1": "smartphone"},
        },
    ]


@pytest.fixture
def mock_workspace_dir(tmp_path):
    """Create a mock workspace directory."""
    # Create minimal structure for ProblemScorer
    (tmp_path / "src" / "agent").mkdir(parents=True)
    (tmp_path / "indexes").mkdir()
    return tmp_path


class TestProgressReporter:
    def test_start_creates_thread(
        self, mock_backend_client, temp_output_file, sample_problems, mock_workspace_dir
    ):
        with patch.object(ProgressReporter, "_initialize_scorers"):
            reporter = ProgressReporter(stop_retry_interval=0.01, 
                backend_client=mock_backend_client,
                eval_run_id="run-123",
                output_file=temp_output_file,
                problems=sample_problems,
                workspace_dir=mock_workspace_dir,
            )

            reporter.start_monitoring()
            assert reporter._thread is not None
            assert reporter._thread.is_alive()

            reporter.stop_monitoring()

    def test_stop_terminates_thread(
        self, mock_backend_client, temp_output_file, sample_problems, mock_workspace_dir
    ):
        with patch.object(ProgressReporter, "_initialize_scorers"):
            reporter = ProgressReporter(stop_retry_interval=0.01, 
                backend_client=mock_backend_client,
                eval_run_id="run-123",
                output_file=temp_output_file,
                problems=sample_problems,
                workspace_dir=mock_workspace_dir,
            )

            reporter.start_monitoring()
            reporter.stop_monitoring()

            assert not reporter._thread.is_alive()

    def test_processes_remaining_lines_on_stop(
        self, mock_backend_client, temp_output_file, sample_problems, mock_workspace_dir
    ):
        """Test that stop_monitoring processes lines written after thread stops."""
        with patch.object(ProgressReporter, "_initialize_scorers"):
            reporter = ProgressReporter(stop_retry_interval=0.01, 
                backend_client=mock_backend_client,
                eval_run_id="12345678-1234-1234-1234-123456789012",
                output_file=temp_output_file,
                problems=sample_problems,
                workspace_dir=mock_workspace_dir,
                poll_interval=0.05,
            )
            # Mock the scorer
            mock_scorer = MagicMock()
            mock_scorer.score_problem.return_value = {
                "gt": 1.0,
                "rule": 1.0,
                "format": 0.8,
                "product": 1.0,
            }
            reporter._scorers = {"product": mock_scorer}

            reporter.start_monitoring()
            time.sleep(0.1)

            # Write dialogue output AFTER the thread has been running
            # (simulates sandbox writing at end of execution)
            dialogue = [
                {
                    "completion": {"content": "test"},
                    "extra_info": {
                        "step": 1,
                        "query": "Find a laptop under $1000",
                        "problem_id": "11111111-1111-1111-1111-111111111111",
                    },
                }
            ]
            with open(temp_output_file, "w") as f:
                f.write(json.dumps(dialogue) + "\n")

            # Stop should process remaining lines
            reporter.stop_monitoring()

            # Verify scoring and reporting happened
            mock_scorer.score_problem.assert_called()
            mock_backend_client.report_progress.assert_called()

    def test_get_aggregate_score_computes_if_not_done(
        self, mock_backend_client, temp_output_file, sample_problems, mock_workspace_dir
    ):
        """Test that get_aggregate_score computes aggregate if scores exist but aggregate wasn't computed."""
        with patch.object(ProgressReporter, "_initialize_scorers"):
            reporter = ProgressReporter(stop_retry_interval=0.01, 
                backend_client=mock_backend_client,
                eval_run_id="12345678-1234-1234-1234-123456789012",
                output_file=temp_output_file,
                problems=sample_problems,
                workspace_dir=mock_workspace_dir,
            )

            # Manually add some scores without computing aggregate
            reporter._problem_scores = {
                "11111111-1111-1111-1111-111111111111": 1.0,
                "22222222-2222-2222-2222-222222222222": 0.0,
            }
            reporter._problem_score_dicts = {
                "11111111-1111-1111-1111-111111111111": {
                    "gt": 1.0,
                    "rule": 1.0,
                    "format": 0.8,
                    "product": 1.0,
                },
                "22222222-2222-2222-2222-222222222222": {
                    "gt": 0.0,
                    "rule": 0.0,
                    "format": 0.6,
                    "product": 0.0,
                },
            }
            reporter._problem_categories = {
                "11111111-1111-1111-1111-111111111111": "product",
                "22222222-2222-2222-2222-222222222222": "product",
            }

            # get_aggregate_score should compute the aggregate
            result = reporter.get_aggregate_score()

            assert result is not None
            assert result["success_rate"] == 0.5
            assert result["ground_truth_rate"] == 0.5
            assert result["format_score"] == 0.7
            assert result["field_matching"] == 0.5
            assert result["total_problems"] == 2
            assert result["successful_problems"] == 1

    def test_skips_dialogue_without_problem_id(
        self, mock_backend_client, temp_output_file, sample_problems, mock_workspace_dir
    ):
        """Test that dialogues without problem_id in extra_info are skipped."""
        with patch.object(ProgressReporter, "_initialize_scorers"):
            reporter = ProgressReporter(stop_retry_interval=0.01, 
                backend_client=mock_backend_client,
                eval_run_id="12345678-1234-1234-1234-123456789012",
                output_file=temp_output_file,
                problems=sample_problems,
                workspace_dir=mock_workspace_dir,
                poll_interval=0.05,
            )
            # Mock the scorer
            mock_scorer = MagicMock()
            reporter._scorers = {"product": mock_scorer}

            # Write dialogue without problem_id
            dialogue = [
                {
                    "completion": {"content": "test"},
                    "extra_info": {"step": 1, "query": "test query"},
                    # Note: no problem_id
                }
            ]
            with open(temp_output_file, "w") as f:
                f.write(json.dumps(dialogue) + "\n")

            reporter.start_monitoring()
            time.sleep(0.15)
            reporter.stop_monitoring()

            # Should NOT have called the scorer (no problem_id)
            mock_scorer.score_problem.assert_not_called()

    def test_handles_empty_output_file(
        self, mock_backend_client, temp_output_file, sample_problems, mock_workspace_dir
    ):
        """Test that empty output file is handled gracefully."""
        with patch.object(ProgressReporter, "_initialize_scorers"):
            reporter = ProgressReporter(stop_retry_interval=0.01, 
                backend_client=mock_backend_client,
                eval_run_id="12345678-1234-1234-1234-123456789012",
                output_file=temp_output_file,
                problems=sample_problems,
                workspace_dir=mock_workspace_dir,
                poll_interval=0.05,
            )

            reporter.start_monitoring()
            time.sleep(0.1)
            reporter.stop_monitoring()

            # Should not crash, aggregate has 0 success_rate with no scored problems
            agg = reporter.get_aggregate_score()
            assert agg is not None
            assert agg["success_rate"] == 0.0
            assert agg["scored_problems"] == 0
            mock_backend_client.report_progress.assert_not_called()

    def test_processes_multiple_problems(
        self, mock_backend_client, temp_output_file, sample_problems, mock_workspace_dir
    ):
        """Test scoring and reporting multiple problems."""
        with patch.object(ProgressReporter, "_initialize_scorers"):
            reporter = ProgressReporter(stop_retry_interval=0.01, 
                backend_client=mock_backend_client,
                eval_run_id="12345678-1234-1234-1234-123456789012",
                output_file=temp_output_file,
                problems=sample_problems,
                workspace_dir=mock_workspace_dir,
                poll_interval=0.05,
            )
            # Mock the scorer to return different scores
            mock_scorer = MagicMock()
            mock_scorer.score_problem.side_effect = [
                {"gt": 1.0, "rule": 1.0, "format": 0.8, "product": 1.0},
                {"gt": 0.0, "rule": 0.0, "format": 0.6, "product": 0.0},
            ]
            reporter._scorers = {"product": mock_scorer}

            # Write two dialogues
            with open(temp_output_file, "w") as f:
                for i, problem in enumerate(sample_problems):
                    dialogue = [
                        {
                            "completion": {"content": f"test {i}"},
                            "extra_info": {
                                "step": 1,
                                "query": problem["query"],
                                "problem_id": problem["problem_id"],
                            },
                        }
                    ]
                    f.write(json.dumps(dialogue) + "\n")

            reporter.start_monitoring()
            time.sleep(0.15)
            reporter.stop_monitoring()

            # Should have scored and reported 2 problems
            assert mock_scorer.score_problem.call_count == 2
            assert mock_backend_client.report_progress.call_count == 2

            # Aggregate should show 50% success rate
            aggregate = reporter.get_aggregate_score()
            assert aggregate["success_rate"] == 0.5


class TestReportProgressRetry:
    """Tests for _report_progress failure handling and retry queue (ORO-340)."""

    @pytest.fixture
    def reporter(self, mock_backend_client, tmp_path):
        output_file = tmp_path / "output.jsonl"
        output_file.touch()
        storage_path = tmp_path / "retry_queue.json"
        retry_queue = LocalRetryQueue(mock_backend_client, storage_path)
        problems = [
            {
                "problem_id": "11111111-1111-1111-1111-111111111111",
                "query": "test",
                "category": "product",
                "reward": {"field_1": "test"},
            }
        ]
        with patch.object(ProgressReporter, "_initialize_scorers"):
            return ProgressReporter(stop_retry_interval=0.01, 
                backend_client=mock_backend_client,
                eval_run_id="12345678-1234-1234-1234-123456789012",
                output_file=output_file,
                problems=problems,
                workspace_dir=tmp_path,
                retry_queue=retry_queue,
            )

    @pytest.fixture
    def progress(self):
        from uuid import UUID

        return ProblemProgressUpdate(
            problem_id=UUID("11111111-1111-1111-1111-111111111111"),
            status=ProblemStatus.SUCCESS,
            score=0.85,
        )

    def test_success_no_failures(self, reporter, progress, mock_backend_client):
        reporter._report_progress(progress)
        assert mock_backend_client.report_progress.call_count == 1
        assert reporter._failed_reports == []

    def test_failure_queued_in_memory(self, reporter, progress, mock_backend_client):
        mock_backend_client.report_progress.side_effect = BackendError("fail", status_code=500)
        reporter._report_progress(progress)
        assert len(reporter._failed_reports) == 1

    def test_stop_retries_then_queues_to_retry_queue(
        self, reporter, progress, mock_backend_client, tmp_path
    ):
        """stop_monitoring retries failed reports; adds to retry queue if still failing."""
        mock_backend_client.report_progress.side_effect = BackendError("fail", status_code=500)
        reporter._report_progress(progress)
        reporter.stop_monitoring()

        storage_path = tmp_path / "retry_queue.json"
        with open(storage_path) as f:
            data = json.load(f)
        assert len(data["pending"]) == 1
        assert data["pending"][0]["type"] == "progress"

    def test_stop_retry_succeeds_no_queue(self, reporter, progress, mock_backend_client, tmp_path):
        """If the retry in stop_monitoring succeeds, nothing goes to retry queue."""
        mock_backend_client.report_progress.side_effect = [BackendError("fail", status_code=500), None]
        reporter._report_progress(progress)
        reporter.stop_monitoring()

        storage_path = tmp_path / "retry_queue.json"
        with open(storage_path) as f:
            data = json.load(f)
        assert len(data["pending"]) == 0

    def test_start_monitoring_resets_failed_list(self, reporter, progress):
        reporter._failed_reports = [progress]
        reporter.start_monitoring()
        assert reporter._failed_reports == []
        reporter.stop_monitoring()


class TestExtractInferenceStats:
    def test_from_last_step(self):
        dialogue = [
            {"extra_info": {"step": 1}},
            {"extra_info": {"step": 2, "inference_failed": 3, "inference_total": 10}},
        ]
        failed, total = extract_inference_stats(dialogue)
        assert failed == 3
        assert total == 10

    def test_missing_stats_returns_zeros(self):
        dialogue = [{"extra_info": {"step": 1}}]
        failed, total = extract_inference_stats(dialogue)
        assert failed == 0
        assert total == 0

    def test_empty_dialogue_returns_zeros(self):
        failed, total = extract_inference_stats([])
        assert failed == 0
        assert total == 0

    def test_no_extra_info_returns_zeros(self):
        dialogue = [{"completion": {}}]
        failed, total = extract_inference_stats(dialogue)
        assert failed == 0
        assert total == 0
