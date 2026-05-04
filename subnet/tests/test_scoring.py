"""Tests for shared scoring module."""

import pytest
from scoring import (
    is_problem_successful,
    compute_aggregate,
    blend_final_score,
    reasoning_coefficient,
    COEFF_FLOOR,
)


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


class TestReasoningCoefficient:
    def test_zero_reasoning(self):
        assert reasoning_coefficient(0.0) == COEFF_FLOOR

    def test_perfect_reasoning(self):
        assert reasoning_coefficient(1.0) == 1.0

    def test_at_ceiling_threshold(self):
        """Reasoning quality >= 0.80 should give coefficient 1.0."""
        assert reasoning_coefficient(0.80) == 1.0

    def test_above_ceiling_threshold(self):
        assert reasoning_coefficient(0.85) == 1.0
        assert reasoning_coefficient(0.95) == 1.0

    def test_just_below_ceiling(self):
        """0.79 should be close to 1.0 but not quite."""
        coeff = reasoning_coefficient(0.79)
        assert coeff < 1.0
        assert coeff > 0.95

    def test_half_reasoning(self):
        """0.5 maps linearly between floor and ceiling."""
        coeff = reasoning_coefficient(0.5)
        assert COEFF_FLOOR < coeff < 1.0
        assert coeff == pytest.approx(0.7375, abs=0.01)

    def test_clamps_below_zero(self):
        assert reasoning_coefficient(-0.5) == COEFF_FLOOR

    def test_clamps_above_one(self):
        assert reasoning_coefficient(1.5) == 1.0


class TestBlendFinalScore:
    def test_perfect_reasoning_full_credit(self):
        """Perfect reasoning = coefficient 1.0, full outcome credit."""
        result = blend_final_score(success_rate=0.6, reasoning_quality=1.0)
        assert result == pytest.approx(0.6, abs=0.001)

    def test_zero_reasoning_floor(self):
        """Zero reasoning = coefficient 0.3, outcome heavily penalized."""
        result = blend_final_score(success_rate=0.6, reasoning_quality=0.0)
        assert result == pytest.approx(0.6 * COEFF_FLOOR, abs=0.001)

    def test_regex_agent_crushed(self):
        """Perfect outcome but zero reasoning = only 30% credit."""
        result = blend_final_score(success_rate=1.0, reasoning_quality=0.0)
        assert result == pytest.approx(COEFF_FLOOR, abs=0.001)

    def test_reasoning_agent_beats_regex(self):
        """Mediocre outcome + good reasoning beats perfect outcome + no reasoning."""
        reasoning_score = blend_final_score(success_rate=0.5, reasoning_quality=0.8)
        regex_score = blend_final_score(success_rate=1.0, reasoning_quality=0.0)
        assert reasoning_score > regex_score

    def test_both_zero(self):
        result = blend_final_score(success_rate=0.0, reasoning_quality=0.0)
        assert result == 0.0

    def test_both_perfect(self):
        result = blend_final_score(success_rate=1.0, reasoning_quality=1.0)
        assert result == pytest.approx(1.0, abs=0.001)
