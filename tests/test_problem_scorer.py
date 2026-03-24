"""Tests for ProblemScorer -- covers scoring hardening fixes."""

import logging
from unittest.mock import patch

import pytest

from src.agent.problem_scorer import ProblemScorer


# -- Helpers --


def make_output(product_ids: str) -> list[dict]:
    """Build a minimal rollout output that recommends given product_ids."""
    return [
        {
            "completion": {
                "message": {
                    "tool_call": [
                        {
                            "name": "recommend_product",
                            "parameters": {"product_ids": product_ids},
                        },
                        {
                            "name": "terminate",
                            "parameters": {},
                        },
                    ]
                }
            }
        }
    ]


def make_product(product_id: str, shop_id: int = 1, price: float = 10.0, title: str = "Test Product") -> dict:
    """Build a minimal product dict."""
    return {
        "product_id": product_id,
        "shop_id": shop_id,
        "price": price,
        "title": title,
        "service": "",
        "sku_options": {},
        "attributes": {},
    }


def make_reward(product_id: str) -> dict:
    """Build a minimal reward (ground truth) dict."""
    return {"product_id": product_id}


QUERY = "find me a phone case"


def mock_get_product(catalog: dict):
    """Return a patched get_product that does a strict lookup from catalog dict.

    Deliberately does NOT strip whitespace -- the production code must strip
    before calling get_product, otherwise the lookup fails (just like the real
    search-server would).
    """
    def _get(product_id: str):
        return catalog.get(product_id)
    return _get


# ---------------------------------------------------------------------------
# Task 2: Whitespace stripping on product ID split
# ---------------------------------------------------------------------------


class TestWhitespaceStripping:
    """Verify that whitespace around product IDs in comma-separated lists
    is stripped before product lookup in all three task types."""

    # ---- shop task ----

    @patch("src.agent.problem_scorer.get_product")
    def test_product_ids_with_spaces_in_shop_task(self, mock_gp):
        """'p1, p2' (space after comma) should still match both products."""
        catalog = {
            "p1": make_product("p1", shop_id=42),
            "p2": make_product("p2", shop_id=42),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="shop",
            rewards={QUERY: [make_reward("p1"), make_reward("p2")]},
            vouchers={},
        )
        # Note the space after the comma
        output = make_output("p1, p2")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        # Both products should be found (product score > 0)
        assert result["product"] > 0, "Products should be found despite space after comma"
        # Ground truth should match for both
        assert result["gt"] == 1.0, "Ground truth should be 1.0 for exact ID matches"
        # Shop should be 1 since both products are from same shop
        assert result["shop"] == 1, "Shop score should be 1 when all from same shop"

    @patch("src.agent.problem_scorer.get_product")
    def test_product_ids_with_multiple_spaces_in_shop_task(self, mock_gp):
        """'p1,   p2,  p3' (multiple spaces) should still match."""
        catalog = {
            "p1": make_product("p1", shop_id=1),
            "p2": make_product("p2", shop_id=1),
            "p3": make_product("p3", shop_id=1),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="shop",
            rewards={QUERY: [make_reward("p1"), make_reward("p2"), make_reward("p3")]},
            vouchers={},
        )
        output = make_output("p1,   p2,  p3")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["product"] > 0
        assert result["gt"] == 1.0
        assert result["shop"] == 1

    # ---- product task ----

    @patch("src.agent.problem_scorer.get_product")
    def test_product_ids_with_leading_trailing_spaces(self, mock_gp):
        """'  p1  ' (leading/trailing whitespace) in product task should match."""
        catalog = {
            "p1": make_product("p1"),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="product",
            rewards={QUERY: make_reward("p1")},
            vouchers={},
        )
        output = make_output("  p1  ")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["product"] == 1, "Product should be found despite leading/trailing spaces"
        assert result["gt"] == 1.0

    @patch("src.agent.problem_scorer.get_product")
    def test_product_ids_with_leading_trailing_spaces_multiple(self, mock_gp):
        """'  p1  ,  p2  ' — product task only uses first ID, but it must be stripped."""
        catalog = {
            "p1": make_product("p1"),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="product",
            rewards={QUERY: make_reward("p1")},
            vouchers={},
        )
        output = make_output("  p1  ,  p2  ")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["product"] == 1
        assert result["gt"] == 1.0

    # ---- tabs and newlines ----

    @patch("src.agent.problem_scorer.get_product")
    def test_product_ids_with_tabs_and_newlines(self, mock_gp):
        """'p1,\\t p2 \\n' edge case — tabs and newlines should be stripped."""
        catalog = {
            "p1": make_product("p1", shop_id=7),
            "p2": make_product("p2", shop_id=7),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="shop",
            rewards={QUERY: [make_reward("p1"), make_reward("p2")]},
            vouchers={},
        )
        output = make_output("p1,\t p2 \n")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["product"] > 0, "Products should be found despite tabs/newlines"
        assert result["gt"] == 1.0

    @patch("src.agent.problem_scorer.get_product")
    def test_product_ids_with_tabs_and_newlines_product_task(self, mock_gp):
        """'\\tp1\\n' in product task — first ID with tabs/newlines should be stripped."""
        catalog = {
            "p1": make_product("p1"),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="product",
            rewards={QUERY: make_reward("p1")},
            vouchers={},
        )
        output = make_output("\tp1\n")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["product"] == 1
        assert result["gt"] == 1.0

    # ---- regression: no spaces ----

    @patch("src.agent.problem_scorer.get_product")
    def test_no_spaces_still_works(self, mock_gp):
        """'p1,p2' (no spaces) should still work — regression check."""
        catalog = {
            "p1": make_product("p1", shop_id=5),
            "p2": make_product("p2", shop_id=5),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="shop",
            rewards={QUERY: [make_reward("p1"), make_reward("p2")]},
            vouchers={},
        )
        output = make_output("p1,p2")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["product"] > 0
        assert result["gt"] == 1.0
        assert result["shop"] == 1

    @patch("src.agent.problem_scorer.get_product")
    def test_no_spaces_product_task(self, mock_gp):
        """Single product ID with no spaces — regression check for product task."""
        catalog = {
            "p1": make_product("p1"),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="product",
            rewards={QUERY: make_reward("p1")},
            vouchers={},
        )
        output = make_output("p1")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["product"] == 1
        assert result["gt"] == 1.0

    # ---- voucher task ----

    @patch("src.agent.problem_scorer.get_product")
    def test_voucher_task_with_spaces(self, mock_gp):
        """Verify whitespace stripping works in voucher context too."""
        catalog = {
            "p1": make_product("p1", shop_id=3, price=20.0),
            "p2": make_product("p2", shop_id=3, price=15.0),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        voucher = {
            "budget": 50.0,
            "voucher_type": "platform",
            "threshold": 30.0,
            "discount_type": "fixed",
            "face_value": 5.0,
        }
        scorer = ProblemScorer(
            task="voucher",
            rewards={QUERY: [make_reward("p1"), make_reward("p2")]},
            vouchers={QUERY: voucher},
        )
        # Space after comma
        output = make_output("p1, p2")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["product"] > 0, "Products should be found in voucher task despite spaces"
        assert result["gt"] == 1.0

    @patch("src.agent.problem_scorer.get_product")
    def test_voucher_task_with_tabs(self, mock_gp):
        """Verify whitespace stripping handles tabs in voucher context."""
        catalog = {
            "p1": make_product("p1", shop_id=3, price=10.0),
            "p2": make_product("p2", shop_id=3, price=10.0),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        voucher = {
            "budget": 30.0,
            "voucher_type": "shop",
            "threshold": 15.0,
            "discount_type": "percentage",
            "discount": 0.1,
            "cap": 5.0,
        }
        scorer = ProblemScorer(
            task="voucher",
            rewards={QUERY: [make_reward("p1"), make_reward("p2")]},
            vouchers={QUERY: voucher},
        )
        output = make_output("p1,\tp2")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["product"] > 0
        # Both products must be found for full ground truth match
        assert result["gt"] == 1.0, "Both products should match despite tab separator"

    @patch("src.agent.problem_scorer.get_product")
    def test_voucher_task_no_spaces_regression(self, mock_gp):
        """Voucher task with no spaces should still work — regression check."""
        catalog = {
            "p1": make_product("p1", shop_id=3, price=10.0),
            "p2": make_product("p2", shop_id=3, price=10.0),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        voucher = {
            "budget": 30.0,
            "voucher_type": "platform",
            "threshold": 15.0,
            "discount_type": "fixed",
            "face_value": 5.0,
        }
        scorer = ProblemScorer(
            task="voucher",
            rewards={QUERY: [make_reward("p1"), make_reward("p2")]},
            vouchers={QUERY: voucher},
        )
        output = make_output("p1,p2")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["product"] > 0
        assert result["gt"] == 1.0

    # ---- edge cases ----

    @patch("src.agent.problem_scorer.get_product")
    def test_only_whitespace_product_id_returns_none(self, mock_gp):
        """A product ID that is only whitespace should not match anything."""
        mock_gp.return_value = None

        scorer = ProblemScorer(
            task="product",
            rewards={QUERY: make_reward("p1")},
            vouchers={},
        )
        output = make_output("   ")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        # Product should not be found
        assert result.get("product", 0) == 0

    @patch("src.agent.problem_scorer.get_product")
    def test_mixed_valid_and_whitespace_only_ids_shop(self, mock_gp):
        """'p1,   , p2' — middle entry is whitespace-only, should still find p1 and skip blank."""
        catalog = {
            "p1": make_product("p1", shop_id=1),
            "p2": make_product("p2", shop_id=1),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="shop",
            rewards={QUERY: [make_reward("p1"), make_reward("???"), make_reward("p2")]},
            vouchers={},
        )
        output = make_output("p1,   , p2")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        # At least p1 should be found
        assert result["product"] > 0

    @patch("src.agent.problem_scorer.get_product")
    def test_single_product_with_newline_suffix(self, mock_gp):
        """'p1\\n' — single product ID with trailing newline in product task."""
        catalog = {
            "p1": make_product("p1"),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="product",
            rewards={QUERY: make_reward("p1")},
            vouchers={},
        )
        output = make_output("p1\n")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["product"] == 1
        assert result["gt"] == 1.0

    @patch("src.agent.problem_scorer.get_product")
    def test_unicode_spaces_around_ids(self, mock_gp):
        """Standard strip() handles \\r, \\n, \\t, spaces but not unicode nbsp.
        We only guarantee standard whitespace stripping."""
        catalog = {
            "p1": make_product("p1", shop_id=1),
            "p2": make_product("p2", shop_id=1),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="shop",
            rewards={QUERY: [make_reward("p1"), make_reward("p2")]},
            vouchers={},
        )
        # \r\n around IDs
        output = make_output("p1\r\n,\r\np2")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["product"] > 0
        assert result["gt"] == 1.0


# ---------------------------------------------------------------------------
# Task 3: Deduplication of product IDs
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Verify that duplicate product IDs are removed before scoring,
    preventing miners from inflating scores by repeating IDs."""

    # 1. Duplicate IDs should not be double-counted in shop tasks
    @patch("src.agent.problem_scorer.get_product")
    def test_duplicate_ids_not_double_counted_shop(self, mock_gp):
        """'p1,p1,p1' for a 3-product shop task: only p1 at position 0 gets
        scored, positions 1 and 2 are empty (duplicates removed), so product
        score should be 1/3."""
        catalog = {
            "p1": make_product("p1", shop_id=1),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="shop",
            rewards={QUERY: [make_reward("p1"), make_reward("p2"), make_reward("p3")]},
            vouchers={},
        )
        output = make_output("p1,p1,p1")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        # After dedup, only ["p1"] remains. Only position 0 is filled.
        # product score = 1 hit / 3 reward items = 1/3
        assert result["product"] == pytest.approx(1.0 / 3.0), (
            "Duplicates should be removed; only 1 of 3 positions filled"
        )
        # shop should be 0 because num_hits (1) != len(reward) (3)
        assert result["shop"] == 0

    # 2. Duplicate IDs should not game budget in voucher tasks
    @patch("src.agent.problem_scorer.get_product")
    def test_duplicate_ids_not_double_counted_voucher(self, mock_gp):
        """Miner repeats a cheap product to game budget: duplicates removed
        means num_hits < len(reward), so budget=0."""
        catalog = {
            "cheap": make_product("cheap", shop_id=1, price=5.0),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        voucher = {
            "budget": 20.0,
            "voucher_type": "platform",
            "threshold": 10.0,
            "discount_type": "fixed",
            "face_value": 2.0,
        }
        scorer = ProblemScorer(
            task="voucher",
            rewards={QUERY: [make_reward("cheap"), make_reward("p2"), make_reward("p3")]},
            vouchers={QUERY: voucher},
        )
        # Miner submits same cheap product 3 times
        output = make_output("cheap,cheap,cheap")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        # After dedup only ["cheap"], num_hits=1 != len(reward)=3 => budget=0
        assert result["budget"] == 0, (
            "Duplicated cheap product should not pass budget check"
        )

    # 3. Unique IDs should still work in shop tasks (regression)
    @patch("src.agent.problem_scorer.get_product")
    def test_unique_ids_still_work_shop(self, mock_gp):
        """'p1,p2,p3' (no duplicates) should work normally — regression check."""
        catalog = {
            "p1": make_product("p1", shop_id=1),
            "p2": make_product("p2", shop_id=1),
            "p3": make_product("p3", shop_id=1),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="shop",
            rewards={QUERY: [make_reward("p1"), make_reward("p2"), make_reward("p3")]},
            vouchers={},
        )
        output = make_output("p1,p2,p3")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["product"] == pytest.approx(1.0)
        assert result["gt"] == 1.0
        assert result["shop"] == 1

    # 4. Unique IDs should still work in voucher tasks (regression)
    @patch("src.agent.problem_scorer.get_product")
    def test_unique_ids_still_work_voucher(self, mock_gp):
        """'p1,p2' (no duplicates) in voucher task — regression check."""
        catalog = {
            "p1": make_product("p1", shop_id=1, price=10.0),
            "p2": make_product("p2", shop_id=1, price=10.0),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        voucher = {
            "budget": 30.0,
            "voucher_type": "platform",
            "threshold": 15.0,
            "discount_type": "fixed",
            "face_value": 5.0,
        }
        scorer = ProblemScorer(
            task="voucher",
            rewards={QUERY: [make_reward("p1"), make_reward("p2")]},
            vouchers={QUERY: voucher},
        )
        output = make_output("p1,p2")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["product"] == pytest.approx(1.0)
        assert result["gt"] == 1.0
        assert result["budget"] == 1

    # 5. Unique IDs should still work in product tasks (regression)
    @patch("src.agent.problem_scorer.get_product")
    def test_unique_ids_still_work_product(self, mock_gp):
        """Single product task with a unique ID — regression check."""
        catalog = {
            "p1": make_product("p1"),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="product",
            rewards={QUERY: make_reward("p1")},
            vouchers={},
        )
        output = make_output("p1")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["product"] == 1
        assert result["gt"] == 1.0

    # 6. Partial duplicates should be deduplicated
    @patch("src.agent.problem_scorer.get_product")
    def test_partial_duplicates(self, mock_gp):
        """'p1,p2,p1' for 3-product task: only ['p1','p2'] remain,
        position 2 is missing."""
        catalog = {
            "p1": make_product("p1", shop_id=1),
            "p2": make_product("p2", shop_id=1),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="shop",
            rewards={QUERY: [make_reward("p1"), make_reward("p2"), make_reward("p3")]},
            vouchers={},
        )
        output = make_output("p1,p2,p1")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        # After dedup: ["p1", "p2"], only 2 of 3 positions filled
        assert result["product"] == pytest.approx(2.0 / 3.0), (
            "Partial duplicates: only 2 unique IDs for 3 reward slots"
        )
        # shop=0 because num_hits (2) != len(reward) (3)
        assert result["shop"] == 0

    # 7. All same ID in product task should still find the first
    @patch("src.agent.problem_scorer.get_product")
    def test_all_same_id_product_task(self, mock_gp):
        """'p1,p1,p1' in product task: should still find the first p1."""
        catalog = {
            "p1": make_product("p1"),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="product",
            rewards={QUERY: make_reward("p1")},
            vouchers={},
        )
        output = make_output("p1,p1,p1")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["product"] == 1, (
            "Product task should find first (deduplicated) p1"
        )
        assert result["gt"] == 1.0

    # 8. All-empty IDs after dedup should return no products
    @patch("src.agent.problem_scorer.get_product")
    def test_empty_ids_after_dedup(self, mock_gp):
        """',,' (all empty after strip) should return no products."""
        mock_gp.return_value = None

        scorer = ProblemScorer(
            task="shop",
            rewards={QUERY: [make_reward("p1"), make_reward("p2")]},
            vouchers={},
        )
        output = make_output(",,")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result.get("product", 0) == 0, (
            "All-empty IDs should yield no product matches"
        )

    # 9. Dedup should preserve insertion order
    def test_dedup_preserves_order(self):
        """'p3,p1,p2' should stay in that order after extraction."""
        scorer = ProblemScorer(
            task="shop",
            rewards={QUERY: []},
            vouchers={},
        )
        output = make_output("p3,p1,p2")
        result = scorer._extract_recommended_product(output)

        assert result == ["p3", "p1", "p2"], (
            "Order should be preserved: p3 first, then p1, then p2"
        )

    # 10. Whitespace variants of same ID should collapse to one
    def test_dedup_with_whitespace_variants(self):
        """'p1, p1,  p1' should all collapse to one 'p1'."""
        scorer = ProblemScorer(
            task="shop",
            rewards={QUERY: []},
            vouchers={},
        )
        output = make_output("p1, p1,  p1")
        result = scorer._extract_recommended_product(output)

        assert result == ["p1"], (
            "Whitespace variants of same ID should collapse to single entry"
        )


# ---------------------------------------------------------------------------
# Task 4: Shop/voucher score requires relevance (non-zero rule_score)
# ---------------------------------------------------------------------------


class TestShopScoreRequiresRelevance:
    """Verify that num_hits only increments when rule_score > 0,
    preventing miners from submitting random valid product IDs to game
    shop/budget scores."""

    # 1. Wrong products from same shop should NOT get shop=1
    @patch("src.agent.problem_scorer.get_product")
    def test_wrong_products_same_shop_no_shop_score(self, mock_gp):
        """Two completely wrong products from same shop: shop=0.
        rule_score will be 0 because product_ids don't match and there
        are no scoreable constraints in the reward."""
        catalog = {
            "wrong1": make_product("wrong1", shop_id=42),
            "wrong2": make_product("wrong2", shop_id=42),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="shop",
            rewards={QUERY: [make_reward("correct1"), make_reward("correct2")]},
            vouchers={},
        )
        output = make_output("wrong1,wrong2")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        # Products were found in database, so product count is incremented
        assert result["product"] > 0, "Products exist in DB so product count > 0"
        # Ground truth should be 0 (wrong product IDs)
        assert result["gt"] == 0, "Wrong products should have gt=0"
        # Shop score MUST be 0: wrong products shouldn't count as hits
        assert result["shop"] == 0, (
            "Wrong products (rule_score=0) should not count toward shop score"
        )

    # 2. Correct products from same shop should get shop=1
    @patch("src.agent.problem_scorer.get_product")
    def test_correct_products_same_shop_gets_shop_score(self, mock_gp):
        """Two correct products (ground truth match) from same shop: shop=1."""
        catalog = {
            "p1": make_product("p1", shop_id=42),
            "p2": make_product("p2", shop_id=42),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="shop",
            rewards={QUERY: [make_reward("p1"), make_reward("p2")]},
            vouchers={},
        )
        output = make_output("p1,p2")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["gt"] == 1.0, "Correct products should have gt=1.0"
        assert result["shop"] == 1, (
            "Correct products from same shop should get shop=1"
        )

    # 3. Partially correct: one correct + one wrong from same shop
    @patch("src.agent.problem_scorer.get_product")
    def test_partially_correct_products_shop_score(self, mock_gp):
        """One correct + one wrong from same shop: shop=0.
        num_hits=1 (only correct product counts) != len(reward)=2."""
        catalog = {
            "p1": make_product("p1", shop_id=42),
            "wrong1": make_product("wrong1", shop_id=42),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="shop",
            rewards={QUERY: [make_reward("p1"), make_reward("p2")]},
            vouchers={},
        )
        output = make_output("p1,wrong1")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        # shop=0 because num_hits (1, only p1 has rule_score>0) != len(reward) (2)
        assert result["shop"] == 0, (
            "Partially correct products: num_hits=1 != len(reward)=2, shop should be 0"
        )

    # 4. Wrong products should NOT pass budget even if under budget
    @patch("src.agent.problem_scorer.get_product")
    def test_wrong_products_voucher_no_budget_score(self, mock_gp):
        """Wrong products shouldn't pass budget even if their prices are under budget.
        num_hits=0 (rule_score=0 for both) != len(reward)=2, so budget=0."""
        catalog = {
            "wrong1": make_product("wrong1", shop_id=1, price=5.0),
            "wrong2": make_product("wrong2", shop_id=1, price=5.0),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        voucher = {
            "budget": 100.0,
            "voucher_type": "platform",
            "threshold": 5.0,
            "discount_type": "fixed",
            "face_value": 2.0,
        }
        scorer = ProblemScorer(
            task="voucher",
            rewards={QUERY: [make_reward("correct1"), make_reward("correct2")]},
            vouchers={QUERY: voucher},
        )
        output = make_output("wrong1,wrong2")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["budget"] == 0, (
            "Wrong products (rule_score=0) should not pass budget check"
        )

    # 5. Correct products under budget should get budget=1
    @patch("src.agent.problem_scorer.get_product")
    def test_correct_products_voucher_gets_budget_score(self, mock_gp):
        """Correct products under budget get budget=1."""
        catalog = {
            "p1": make_product("p1", shop_id=1, price=10.0),
            "p2": make_product("p2", shop_id=1, price=10.0),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        voucher = {
            "budget": 30.0,
            "voucher_type": "platform",
            "threshold": 15.0,
            "discount_type": "fixed",
            "face_value": 5.0,
        }
        scorer = ProblemScorer(
            task="voucher",
            rewards={QUERY: [make_reward("p1"), make_reward("p2")]},
            vouchers={QUERY: voucher},
        )
        output = make_output("p1,p2")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        assert result["budget"] == 1, (
            "Correct products under budget should get budget=1"
        )

    # 6. Wrong products' prices should NOT be summed into total_price
    @patch("src.agent.problem_scorer.get_product")
    def test_wrong_products_dont_affect_total_price(self, mock_gp):
        """Wrong products' prices should not be summed into total_price.
        Even if wrong products are expensive, they shouldn't affect budget calc.
        Here we submit two correct + one wrong (expensive) product.
        Only correct products' prices should count."""
        catalog = {
            "p1": make_product("p1", shop_id=1, price=10.0),
            "p2": make_product("p2", shop_id=1, price=10.0),
            "wrong1": make_product("wrong1", shop_id=1, price=9999.0),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        voucher = {
            "budget": 25.0,
            "voucher_type": "platform",
            "threshold": 15.0,
            "discount_type": "fixed",
            "face_value": 5.0,
        }
        # Reward expects 3 products: p1, p2, and p3
        scorer = ProblemScorer(
            task="voucher",
            rewards={QUERY: [make_reward("p1"), make_reward("p2"), make_reward("p3")]},
            vouchers={QUERY: voucher},
        )
        # Miner submits p1, p2 (correct) and wrong1 (wrong, expensive)
        output = make_output("p1,p2,wrong1")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        # num_hits=2 (only p1 and p2 have rule_score>0) != len(reward)=3
        # so budget=0 regardless, but importantly wrong1's price ($9999)
        # should NOT have been added to total_price
        assert result["budget"] == 0, (
            "num_hits=2 != len(reward)=3, so budget should be 0"
        )

    # 7. Mixed relevance: correct cheap + wrong expensive in voucher
    @patch("src.agent.problem_scorer.get_product")
    def test_mixed_relevance_voucher(self, mock_gp):
        """One correct product (cheap) + one wrong (expensive).
        Only correct product's price should count for budget calculation.
        With the fix, the expensive wrong product's price is excluded."""
        catalog = {
            "cheap": make_product("cheap", shop_id=1, price=5.0),
            "expensive_wrong": make_product("expensive_wrong", shop_id=1, price=500.0),
        }
        mock_gp.side_effect = mock_get_product(catalog)

        voucher = {
            "budget": 20.0,
            "voucher_type": "platform",
            "threshold": 3.0,
            "discount_type": "fixed",
            "face_value": 1.0,
        }
        # Reward expects 2 products: cheap and another_correct
        scorer = ProblemScorer(
            task="voucher",
            rewards={QUERY: [make_reward("cheap"), make_reward("another_correct")]},
            vouchers={QUERY: voucher},
        )
        # Miner submits cheap (correct) + expensive_wrong (wrong)
        output = make_output("cheap,expensive_wrong")
        result = scorer.score_problem(QUERY, output)

        assert result is not None
        # num_hits=1 (only cheap has rule_score>0) != len(reward)=2
        # so budget=0 — but crucially, if the old code were running,
        # it would have added $500 to total_price, which is wrong
        assert result["budget"] == 0, (
            "num_hits=1 != len(reward)=2, budget should be 0"
        )


# ---------------------------------------------------------------------------
# Task 5: Logging on silent try/except blocks
# ---------------------------------------------------------------------------


class TestLoggingOnErrors:
    """Verify that try/except blocks log warnings instead of failing silently."""

    # 1. Malformed output during format scoring should log a warning
    @patch("src.agent.problem_scorer.get_product")
    def test_malformed_output_logs_warning_format_scoring(self, mock_gp, caplog):
        """Output with a broken step (missing 'completion') should log a
        warning during format scoring. A valid last step is included so
        length_reward doesn't crash before we reach the format loop."""
        mock_gp.return_value = None

        scorer = ProblemScorer(
            task="product",
            rewards={QUERY: make_reward("p1")},
            vouchers={},
        )
        # First step is broken (missing 'completion'), second step is valid
        # so length_reward can access output[-1]["completion"]["message"]
        valid_step = {
            "completion": {
                "message": {
                    "tool_call": [
                        {"name": "terminate", "parameters": {}},
                    ]
                }
            }
        }
        output = [{"not_completion": "broken"}, valid_step]

        with caplog.at_level(logging.WARNING):
            scorer.score_problem(QUERY, output)

        # Should log a warning specifically about format scoring
        format_warnings = [
            r.message for r in caplog.records
            if r.levelno == logging.WARNING and "format scoring" in r.message.lower()
        ]
        assert len(format_warnings) >= 1, (
            f"Expected a warning about malformed output during format scoring, "
            f"got: {[r.message for r in caplog.records if r.levelno == logging.WARNING]}"
        )

    # 2. Malformed output during product extraction should log a warning
    @patch("src.agent.problem_scorer.get_product")
    def test_malformed_output_logs_warning_product_extraction(self, mock_gp, caplog):
        """Output with a tool_call entry missing the 'parameters' key should
        log a warning during product extraction (KeyError on 'parameters')."""
        mock_gp.return_value = None

        scorer = ProblemScorer(
            task="product",
            rewards={QUERY: make_reward("p1")},
            vouchers={},
        )
        # The tool_call has "recommend_product" but is missing "parameters",
        # which causes KeyError in _extract_recommended_product.
        # length_reward only checks for "terminate" and won't crash on this.
        output = [
            {
                "completion": {
                    "message": {
                        "tool_call": [
                            {"name": "recommend_product"},  # missing "parameters"
                            {"name": "terminate", "parameters": {}},
                        ]
                    }
                }
            }
        ]

        with caplog.at_level(logging.WARNING):
            scorer.score_problem(QUERY, output)

        # Should log a warning about malformed output during product extraction
        extraction_warnings = [
            r.message for r in caplog.records
            if r.levelno == logging.WARNING and "product extraction" in r.message.lower()
        ]
        assert len(extraction_warnings) >= 1, (
            f"Expected a warning about malformed output during product extraction, "
            f"got: {[r.message for r in caplog.records if r.levelno == logging.WARNING]}"
        )

    # 3. Well-formed output should produce NO warnings (regression check)
    @patch("src.agent.problem_scorer.get_product")
    def test_valid_output_no_warnings(self, mock_gp, caplog):
        """Well-formed output should not produce any warnings."""
        catalog = {"p1": make_product("p1")}
        mock_gp.side_effect = mock_get_product(catalog)

        scorer = ProblemScorer(
            task="product",
            rewards={QUERY: make_reward("p1")},
            vouchers={},
        )
        output = make_output("p1")

        with caplog.at_level(logging.WARNING):
            result = scorer.score_problem(QUERY, output)

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_messages) == 0, (
            f"Valid output should produce no warnings, got: {warning_messages}"
        )
        assert result is not None
        assert result["product"] == 1

    # 4. Multiple broken steps should log multiple warnings
    @patch("src.agent.problem_scorer.get_product")
    def test_multiple_malformed_steps_log_multiple_warnings(self, mock_gp, caplog):
        """Multiple broken output steps should each log a warning.
        A valid last step is included so length_reward doesn't crash."""
        mock_gp.return_value = None

        scorer = ProblemScorer(
            task="product",
            rewards={QUERY: make_reward("p1")},
            vouchers={},
        )
        valid_step = {
            "completion": {
                "message": {
                    "tool_call": [
                        {"name": "terminate", "parameters": {}},
                    ]
                }
            }
        }
        # Three broken steps + one valid at the end
        output = [
            {"broken": "step1"},
            {"broken": "step2"},
            {"broken": "step3"},
            valid_step,
        ]

        with caplog.at_level(logging.WARNING):
            scorer.score_problem(QUERY, output)

        # Filter for format-scoring warnings specifically
        format_warnings = [
            r.message for r in caplog.records
            if r.levelno == logging.WARNING and "format scoring" in r.message.lower()
        ]
        assert len(format_warnings) == 3, (
            f"Expected 3 format scoring warnings (one per broken step), "
            f"got {len(format_warnings)}: {format_warnings}"
        )
