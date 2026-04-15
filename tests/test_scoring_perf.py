"""Tests for scoring performance optimizations."""

from unittest.mock import MagicMock, patch

from src.agent.rewards.orm import rule_score_reward


def _product(pid="p1", title="Widget", price=25.0):
    return {"product_id": pid, "title": title, "price": price,
            "shop_id": "s1", "service": ["fast"], "sku_options": {}, "attributes": {}}


def _reward(pid="p1", titles=None, prices=None, services=None):
    r = {"product_id": pid}
    if titles:
        r["title"] = titles
    if prices:
        r["price"] = prices
    if services:
        r["service"] = services
    return r


@patch("src.agent.rewards.orm._get_sentence_model")
def test_gt_match_skips_embedding(mock_get):
    """GT match: no encode calls, title counters still accurate."""
    mock_get.return_value = MagicMock()

    score, total, hits = rule_score_reward(
        _product(pid="A"), _reward(pid="A", titles=["T1", "T2"])
    )

    mock_get.return_value.encode.assert_not_called()
    assert score == 1
    assert total["title"] == 2
    assert hits["title"] == 2


@patch("src.agent.rewards.orm._get_sentence_model")
def test_non_gt_calls_encode(mock_get):
    """Non-GT: encode is called, field counters correct."""
    m = MagicMock()
    m.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
    m.similarity.return_value = [[0.95]]
    mock_get.return_value = m

    score, total, hits = rule_score_reward(
        _product(pid="A", price=15.0),
        _reward(pid="Z", titles=["X"], prices=[{"less than": (0, 20.0)}], services=["fast"]),
    )

    assert m.encode.call_count >= 1
    assert hits["title"] == 1   # similarity 0.95 >= 0.7
    assert hits["price"] == 1   # 15 <= 20
    assert hits["service"] == 1
    assert score == 1.0         # 3/3


@patch("src.agent.problem_scorer.get_product")
@patch("src.agent.rewards.orm._get_sentence_model")
def test_shop_batch_encodes_non_gt_titles(mock_get, mock_prod):
    """Shop task: non-GT titles are batch-encoded, GT title is skipped."""
    from src.agent.problem_scorer import ProblemScorer

    m = MagicMock()
    m.encode.side_effect = lambda inputs: [[0.1]] * len(inputs)
    m.similarity.return_value = [[0.95]]
    mock_get.return_value = m

    mock_prod.side_effect = lambda pid: _product(pid=pid, title=f"Title-{pid}")

    rewards = [
        _reward(pid="gt", titles=["Title-gt"]),   # GT match
        _reward(pid="x", titles=["Reward A"]),     # non-GT
        _reward(pid="y", titles=["Reward B"]),     # non-GT
    ]
    output = [{"completion": {"message": {"tool_call": [
        {"name": "recommend_product", "parameters": {"product_ids": "gt,a,b"}}
    ]}}, "extra_info": {"timestamp": 1.0}}]

    scorer = ProblemScorer(task="shop", rewards={"q": rewards}, vouchers={})
    scorer.score_problem("q", output, model="human")

    # First encode call should be the batch of non-GT titles
    batch = m.encode.call_args_list[0][0][0]
    assert "Title-gt" not in batch
    assert len(batch) == 2
