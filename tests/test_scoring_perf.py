"""Tests for scoring performance optimizations in rule_score_reward."""

from collections import Counter
from unittest.mock import MagicMock, patch

from src.agent.rewards.orm import rule_score_reward


# -- Helpers --

def _make_product(product_id="prod-1", title="Cool Widget", price=25.0):
    return {
        "product_id": product_id,
        "title": title,
        "price": price,
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
