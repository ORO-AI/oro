"""Tests for ProblemScorer - per-problem evaluation scoring."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import sys

# Mock external dependencies before importing problem_scorer
sys.modules["rewards"] = MagicMock()
sys.modules["rewards.orm"] = MagicMock()
sys.modules["rewards.prm"] = MagicMock()
sys.modules["util"] = MagicMock()
sys.modules["util.message"] = MagicMock()

from problem_scorer import ProblemScorer, _product_cache  # noqa: E402


@pytest.fixture(autouse=True)
def clear_product_cache():
    """Clear the product cache before each test."""
    _product_cache.clear()
    yield
    _product_cache.clear()


class TestProblemScorerInit:
    """Test ProblemScorer initialization."""

    def test_init_product_task(self):
        """Test initializing scorer for product task."""
        rewards = {"query1": {"product_id": "p1"}}
        vouchers = {}

        scorer = ProblemScorer(task="product", rewards=rewards, vouchers=vouchers)

        assert scorer.task == "product"
        assert scorer.rewards == rewards
        assert scorer.vouchers == vouchers

    def test_init_shop_task(self):
        """Test initializing scorer for shop task."""
        rewards = {"query1": [{"product_id": "p1"}, {"product_id": "p2"}]}
        vouchers = {}

        scorer = ProblemScorer(task="shop", rewards=rewards, vouchers=vouchers)

        assert scorer.task == "shop"

    def test_init_voucher_task(self):
        """Test initializing scorer for voucher task."""
        rewards = {"query1": [{"product_id": "p1"}]}
        vouchers = {"query1": {"budget": 100}}

        scorer = ProblemScorer(task="voucher", rewards=rewards, vouchers=vouchers)

        assert scorer.task == "voucher"
        assert scorer.vouchers == vouchers


class TestProblemScorerScoreProblem:
    """Test scoring individual problems."""

    @patch("problem_scorer.length_reward")
    @patch("problem_scorer.format_reward")
    @patch("problem_scorer.ground_truth_reward")
    @patch("problem_scorer.rule_score_reward")
    @patch("problem_scorer.Message")
    def test_score_product_task_success(
        self, mock_message, mock_rule_score, mock_gt_reward, mock_format, mock_length
    ):
        """Test scoring a successful product recommendation."""
        # Setup mocks
        mock_length.return_value = 0.8
        mock_format.return_value = 0.95
        mock_gt_reward.return_value = 1.0
        mock_rule_score.return_value = (
            1.0,
            {"title": 1, "price": 1},
            {"title": 1, "price": 1},
        )
        mock_message.from_dict.return_value = Mock(to_string=Mock(return_value="test"))

        rewards = {"query1": {"product_id": "test_product"}}
        vouchers = {}
        scorer = ProblemScorer(task="product", rewards=rewards, vouchers=vouchers)

        output = [
            {
                "completion": {
                    "message": {
                        "tool_call": [
                            {
                                "name": "recommend_product",
                                "parameters": {"product_ids": "test_product"},
                            }
                        ]
                    }
                }
            }
        ]

        with patch("problem_scorer.get_product") as mock_get_product:
            mock_get_product.return_value = {
                "product_id": "test_product",
                "title": "Test Product",
                "price": 100,
                "service": "test",
                "sku": "test-sku",
            }

            score = scorer.score_problem(query="query1", output=output)

        assert isinstance(score, dict)
        assert "gt" in score
        assert "rule" in score
        assert "format" in score
        assert "length" in score
        assert score["product"] == 1
        assert score["length"] == 0.8
        assert score["format"] == 0.95

    @patch("problem_scorer.length_reward")
    @patch("problem_scorer.get_product")
    def test_score_problem_with_no_output(self, mock_get_product, mock_length):
        """Test scoring a problem that produced no output."""
        mock_length.return_value = 0
        mock_get_product.return_value = None  # No product found

        rewards = {"query1": {"product_id": "test_product"}}
        vouchers = {}
        scorer = ProblemScorer(task="product", rewards=rewards, vouchers=vouchers)

        score = scorer.score_problem(query="query1", output=[])

        assert isinstance(score, dict)
        assert score["length"] == 0
        assert score["format"] == 0

    def test_score_problem_missing_reward(self):
        """Test scoring a problem with no reward data."""
        rewards = {}
        vouchers = {}
        scorer = ProblemScorer(task="product", rewards=rewards, vouchers=vouchers)

        output = [{"completion": {"message": {}}}]

        # Should return None or raise exception when reward not found
        result = scorer.score_problem(query="missing_query", output=output)
        assert result is None

    @patch("problem_scorer.length_reward")
    @patch("problem_scorer.format_reward")
    @patch("problem_scorer.ground_truth_reward")
    @patch("problem_scorer.rule_score_reward")
    @patch("problem_scorer.Message")
    def test_score_shop_task_multiple_products(
        self, mock_message, mock_rule_score, mock_gt_reward, mock_format, mock_length
    ):
        """Test scoring shop task with multiple products."""
        # Setup mocks
        mock_length.return_value = 0.8
        mock_format.return_value = 0.95
        mock_gt_reward.return_value = 1.0
        mock_rule_score.return_value = (
            1.0,
            {"title": 1, "price": 1},
            {"title": 1, "price": 1},
        )
        mock_message.from_dict.return_value = Mock(to_string=Mock(return_value="test"))

        rewards = {
            "query1": [
                {"product_id": "p1", "shop_id": "shop1"},
                {"product_id": "p2", "shop_id": "shop1"},
            ]
        }
        vouchers = {}
        scorer = ProblemScorer(task="shop", rewards=rewards, vouchers=vouchers)

        output = [
            {
                "completion": {
                    "message": {
                        "tool_call": [
                            {
                                "name": "recommend_product",
                                "parameters": {"product_ids": "p1,p2"},
                            }
                        ]
                    }
                }
            }
        ]

        with patch("problem_scorer.get_product") as mock_get_product:

            def mock_get_product_fn(product_id):
                return {
                    "product_id": product_id,
                    "shop_id": "shop1",
                    "title": f"Product {product_id}",
                    "price": 100,
                }

            mock_get_product.side_effect = mock_get_product_fn

            score = scorer.score_problem(query="query1", output=output)

        assert score["shop"] == 1  # Both from same shop

    @patch("problem_scorer.length_reward")
    @patch("problem_scorer.format_reward")
    @patch("problem_scorer.ground_truth_reward")
    @patch("problem_scorer.rule_score_reward")
    @patch("problem_scorer.Message")
    def test_score_voucher_task_with_budget(
        self, mock_message, mock_rule_score, mock_gt_reward, mock_format, mock_length
    ):
        """Test scoring voucher task with budget constraint."""
        # Setup mocks
        mock_length.return_value = 0.8
        mock_format.return_value = 0.95
        mock_gt_reward.return_value = 1.0
        mock_rule_score.return_value = (
            1.0,
            {"title": 1, "price": 1},
            {"title": 1, "price": 1},
        )
        mock_message.from_dict.return_value = Mock(to_string=Mock(return_value="test"))

        rewards = {"query1": [{"product_id": "p1"}]}
        vouchers = {
            "query1": {
                "budget": 100,
                "voucher_type": "platform",
                "threshold": 50,
                "discount_type": "fixed",
                "face_value": 10,
            }
        }
        scorer = ProblemScorer(task="voucher", rewards=rewards, vouchers=vouchers)

        output = [
            {
                "completion": {
                    "message": {
                        "tool_call": [
                            {
                                "name": "recommend_product",
                                "parameters": {"product_ids": "p1"},
                            }
                        ]
                    }
                }
            }
        ]

        with patch("problem_scorer.get_product") as mock_get_product:
            mock_get_product.return_value = {
                "product_id": "p1",
                "shop_id": "shop1",
                "price": 80,
                "title": "Product 1",
            }

            score = scorer.score_problem(query="query1", output=output)

        assert "budget" in score


class TestProblemScorerWriteProgress:
    """Test writing progress to file."""

    def test_write_progress_creates_jsonl(self):
        """Test that write_progress writes valid JSONL."""
        rewards = {"query1": {"product_id": "p1"}}
        vouchers = {}
        scorer = ProblemScorer(task="product", rewards=rewards, vouchers=vouchers)

        score = {
            "gt": 1.0,
            "rule": 1.0,
            "format": 0.95,
            "length": 0.8,
            "product": 1,
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            output_file = Path(f.name)

        try:
            scorer.write_progress(
                problem_id="test-problem-123", score=score, output_file=output_file
            )

            # Read back and verify
            with open(output_file) as f:
                line = f.read().strip()
                data = json.loads(line)

            assert data["problem_id"] == "test-problem-123"
            assert data["status"] == "SUCCESS"
            assert data["score"] == score
        finally:
            output_file.unlink()

    def test_write_progress_appends_to_existing_file(self):
        """Test that write_progress appends to existing file."""
        rewards = {}
        vouchers = {}
        scorer = ProblemScorer(task="product", rewards=rewards, vouchers=vouchers)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            output_file = Path(f.name)
            # Write existing content
            f.write('{"existing": "data"}\n')

        try:
            scorer.write_progress(
                problem_id="prob-1", score={"gt": 1.0}, output_file=output_file
            )

            # Read all lines
            with open(output_file) as f:
                lines = f.readlines()

            assert len(lines) == 2
            assert "existing" in lines[0]
            assert "prob-1" in lines[1]
        finally:
            output_file.unlink()

    def test_write_progress_determines_status_from_score(self):
        """Test that status is determined correctly from score."""
        rewards = {}
        vouchers = {}
        scorer = ProblemScorer(task="product", rewards=rewards, vouchers=vouchers)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            output_file = Path(f.name)

        try:
            # Test failed score
            failed_score = {"rule": 0.0, "gt": 0.0}
            scorer.write_progress("prob-1", failed_score, output_file)

            with open(output_file) as f:
                data = json.loads(f.read().strip())

            assert data["status"] == "FAILED"
        finally:
            output_file.unlink()

    def test_write_progress_with_no_score_data(self):
        """Test writing progress when scoring returned None."""
        rewards = {}
        vouchers = {}
        scorer = ProblemScorer(task="product", rewards=rewards, vouchers=vouchers)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            output_file = Path(f.name)

        try:
            scorer.write_progress(
                problem_id="prob-1", score=None, output_file=output_file
            )

            with open(output_file) as f:
                data = json.loads(f.read().strip())

            assert data["problem_id"] == "prob-1"
            assert data["status"] == "FAILED"
            assert data.get("score") is None
        finally:
            output_file.unlink()


class TestProblemScorerEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_task_type_raises_exception(self):
        """Test that invalid task type is caught."""
        with pytest.raises(ValueError, match="Invalid task"):
            ProblemScorer(task="invalid", rewards={}, vouchers={})

    @patch("problem_scorer.length_reward")
    @patch("problem_scorer.format_reward")
    @patch("problem_scorer.Message")
    @patch("problem_scorer.get_product")
    def test_score_problem_with_malformed_output(
        self, mock_get_product, mock_message, mock_format, mock_length
    ):
        """Test scoring with malformed output data."""
        # Setup mocks
        mock_length.return_value = 0.5
        mock_format.return_value = 0.0
        mock_message.from_dict.return_value = Mock(to_string=Mock(return_value="test"))
        mock_get_product.return_value = None  # No product found

        rewards = {"query1": {"product_id": "p1"}}
        vouchers = {}
        scorer = ProblemScorer(task="product", rewards=rewards, vouchers=vouchers)

        # Malformed output missing expected structure
        output = [{"bad": "data"}]

        # Should handle gracefully without crashing
        score = scorer.score_problem(query="query1", output=output)
        assert isinstance(score, dict)
