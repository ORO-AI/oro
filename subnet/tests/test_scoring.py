"""Tests for shared scoring module."""

# pytest.ini adds src/agent to pythonpath, so bare imports work
import pytest
from scoring import is_problem_successful, compute_aggregate


class TestIsProblemSuccessful:
    """Category-aware success: product=rule>=1, shop=rule+shop, voucher=rule+budget."""

    @pytest.mark.parametrize("category,score_dict,expected", [
        # Product: rule >= 1 is enough
        ("product", {"gt": 1, "rule": 1.0}, True),
        ("product", {"gt": 0, "rule": 1.0}, True),
        ("product", {"gt": 1, "rule": 0.5}, False),
        ("product", {"gt": 0, "rule": 0}, False),
        # Shop: rule >= 1 AND shop >= 1
        ("shop", {"gt": 0.5, "rule": 1.0, "shop": 1}, True),
        ("shop", {"gt": 1, "rule": 1.0, "shop": 0}, False),
        ("shop", {"gt": 0, "rule": 0.5, "shop": 1}, False),
        # Voucher: rule >= 1 AND budget >= 1
        ("voucher", {"gt": 0.5, "rule": 1.0, "budget": 1}, True),
        ("voucher", {"gt": 1, "rule": 1.0, "budget": 0}, False),
        ("voucher", {"gt": 0, "rule": 0.5, "budget": 1}, False),
        # Edge: None score_dict
        ("product", None, False),
        # Edge: unknown category defaults to product logic
        ("unknown", {"rule": 1.0}, True),
    ], ids=[
        "product-gt1-rule1", "product-gt0-rule1", "product-gt1-rule05", "product-all0",
        "shop-pass", "shop-no-shop", "shop-low-rule",
        "voucher-pass", "voucher-no-budget", "voucher-low-rule",
        "none-score", "unknown-category",
    ])
    def test_is_problem_successful(self, category, score_dict, expected):
        assert is_problem_successful(score_dict, category) == expected


class TestComputeAggregate:
    def test_basic_aggregate(self):
        results = [
            {"category": "product", "score_dict": {"gt": 1, "rule": 1.0, "format": 1.0}},
            {"category": "product", "score_dict": {"gt": 0, "rule": 0.5, "format": 0.5}},
        ]
        agg = compute_aggregate(results, total_problems=4)
        assert agg["ground_truth_rate"] == 0.25  # 1/4
        assert agg["success_rate"] == 0.25  # 1/4 (only first has rule>=1)
        assert agg["total_problems"] == 4
        assert agg["scored_problems"] == 2

    def test_shop_category_aware(self):
        results = [
            {"category": "shop", "score_dict": {"gt": 1, "rule": 1.0, "shop": 1, "format": 1.0}},
            {"category": "shop", "score_dict": {"gt": 1, "rule": 1.0, "shop": 0, "format": 1.0}},
        ]
        agg = compute_aggregate(results, total_problems=2)
        assert agg["success_rate"] == 0.5  # only first passes (shop=1)

    def test_unscored_count_as_zero(self):
        results = [
            {"category": "product", "score_dict": {"gt": 1, "rule": 1.0, "format": 1.0}},
        ]
        agg = compute_aggregate(results, total_problems=10)
        assert agg["ground_truth_rate"] == 0.1  # 1/10
        assert agg["success_rate"] == 0.1

    def test_empty_results(self):
        agg = compute_aggregate([], total_problems=30)
        assert agg["ground_truth_rate"] == 0.0
        assert agg["success_rate"] == 0.0
        assert agg["total_problems"] == 30
        assert agg["scored_problems"] == 0

    def test_none_score_dict_filtered(self):
        results = [
            {"category": "product", "score_dict": {"gt": 1, "rule": 1.0, "format": 1.0}},
            {"category": "product", "score_dict": None},
        ]
        agg = compute_aggregate(results, total_problems=2)
        assert agg["scored_problems"] == 1
        assert agg["success_rate"] == 0.5
