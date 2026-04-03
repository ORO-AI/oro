"""Tests for scoring performance optimizations in rule_score_reward."""

from collections import Counter
from unittest.mock import MagicMock, call, patch

from src.agent.rewards.orm import rule_score_reward
from src.agent.problem_scorer import ProblemScorer


# -- Helpers --

def _make_product(product_id="prod-1", title="Cool Widget", price=25.0, shop_id="shop-1"):
    return {
        "product_id": product_id,
        "title": title,
        "price": price,
        "shop_id": shop_id,
        "service": ["fast-shipping"],
        "sku_options": {},
        "attributes": {},
    }


def _make_reward(product_id="prod-1", titles=None, prices=None, services=None):
    reward = {"product_id": product_id}
    if titles is not None:
        reward["title"] = titles
    if prices is not None:
        reward["price"] = prices
    if services is not None:
        reward["service"] = services
    return reward


# -- Tests --


class TestGTMatchSkipsEmbedding:
    """When product_id matches (GT match), model.encode() must NOT be called."""

    @patch("src.agent.rewards.orm._get_sentence_model")
    def test_gt_match_skips_embedding(self, mock_get_model):
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        product = _make_product(product_id="ABC")
        reward = _make_reward(product_id="ABC", titles=["Cool Widget", "Nice Widget"])

        score, total_counter, hit_counter = rule_score_reward(product, reward)

        # model.encode should never be called on GT match
        mock_model.encode.assert_not_called()
        mock_model.similarity.assert_not_called()

        # Score must be 1.0 (GT match)
        assert score == 1

        # Title counters should still reflect all titles as hits
        assert total_counter["title"] == 2
        assert hit_counter["title"] == 2


class TestNonGTStillComputesEmbeddings:
    """When product_id does NOT match, model.encode() must be called."""

    @patch("src.agent.rewards.orm._get_sentence_model")
    def test_non_gt_still_computes_embeddings(self, mock_get_model):
        mock_model = MagicMock()
        # Simulate encode returning embeddings and similarity returning a high score
        mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_model.similarity.return_value = [[0.95]]
        mock_get_model.return_value = mock_model

        product = _make_product(product_id="ABC")
        reward = _make_reward(product_id="XYZ", titles=["Cool Widget"])

        rule_score_reward(product, reward)

        # model.encode MUST be called for non-GT
        assert mock_model.encode.call_count >= 1


class TestNonGTScoringFieldsCorrect:
    """Verify price/service counters work correctly for non-GT products."""

    @patch("src.agent.rewards.orm._get_sentence_model")
    def test_non_gt_scoring_fields_correct(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_model.similarity.return_value = [[0.95]]
        mock_get_model.return_value = mock_model

        product = _make_product(product_id="ABC", price=15.0)
        reward = _make_reward(
            product_id="XYZ",
            titles=["Cool Widget"],
            prices=[{"less than": (0, 20.0)}],
            services=["fast-shipping"],
        )

        score, total_counter, hit_counter = rule_score_reward(product, reward)

        # Non-GT: score is computed from hit_count / total_count
        # All three fields should be counted
        assert total_counter["title"] == 1
        assert total_counter["price"] == 1
        assert total_counter["service"] == 1
        # Title similarity 0.95 >= 0.7 -> hit; price 15 <= 20 -> hit; service present -> hit
        assert hit_counter["title"] == 1
        assert hit_counter["price"] == 1
        assert hit_counter["service"] == 1
        # 3 hits / 3 total (ignoring sku & attrs which add 0/0)
        assert score == 1.0


def _make_output(product_ids: list[str]):
    """Build minimal rollout output recommending the given product IDs."""
    return [
        {
            "completion": {
                "message": {
                    "tool_call": [
                        {
                            "name": "recommend_product",
                            "parameters": {"product_ids": ",".join(product_ids)},
                        }
                    ]
                }
            },
            "extra_info": {"timestamp": 1.0},
        }
    ]


class TestShopBatchEncodesTitles:
    """For shop tasks, non-GT product titles are batch-encoded in one call."""

    @patch("src.agent.problem_scorer.get_product")
    @patch("rewards.orm._get_sentence_model")
    def test_shop_batch_encodes_titles(self, mock_get_model, mock_get_product):
        mock_model = MagicMock()

        def encode_side_effect(inputs):
            # Return one embedding per input
            return [[0.1, 0.2]] * len(inputs)

        mock_model.encode.side_effect = encode_side_effect
        mock_model.similarity.return_value = [[0.95]]
        mock_get_model.return_value = mock_model

        # 3 products: first is GT match, other two are non-GT
        products = {
            "pid-gt": _make_product(product_id="pid-gt", title="GT Product"),
            "pid-a": _make_product(product_id="pid-a", title="Product A"),
            "pid-b": _make_product(product_id="pid-b", title="Product B"),
        }
        mock_get_product.side_effect = lambda pid: products.get(pid)

        rewards_list = [
            _make_reward(product_id="pid-gt", titles=["GT Product"]),
            _make_reward(product_id="pid-x", titles=["Reward Title A"]),
            _make_reward(product_id="pid-y", titles=["Reward Title B"]),
        ]

        query = "test-query"
        scorer = ProblemScorer(
            task="shop",
            rewards={query: rewards_list},
            vouchers={},
        )

        output = _make_output(["pid-gt", "pid-a", "pid-b"])
        scorer.score_problem(query, output, model="human")

        # The first model.encode call should be the batch call with both non-GT titles
        first_encode_call = mock_model.encode.call_args_list[0]
        batch_titles = first_encode_call[0][0]
        assert batch_titles == ["Product A", "Product B"], (
            f"Expected batch encode of 2 non-GT titles, got: {batch_titles}"
        )

        # The GT product (pid-gt) should NOT appear in any encode call
        for c in mock_model.encode.call_args_list:
            titles_arg = c[0][0]
            if isinstance(titles_arg, list):
                assert "GT Product" not in titles_arg, (
                    "GT product title should not be encoded"
                )
