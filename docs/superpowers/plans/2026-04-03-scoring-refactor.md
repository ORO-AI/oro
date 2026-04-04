# Scoring Performance Refactor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce per-problem scoring time on slow validators by separating field comparison logic from embedding computation and skipping embeddings when they won't affect the result.

**Architecture:** The current `rule_score_reward` function does two things: computes the rule score (1.0 on GT match, field match ratio otherwise) AND populates per-field counters used by aggregate metrics. The embedding computation (the expensive part) only affects the title field counter, but runs unconditionally. We'll separate the concerns so embeddings can be skipped on GT match without losing counter accuracy, and batch encode product titles across multiple products in shop/voucher tasks.

**Tech Stack:** Python, sentence-transformers (Qwen/Qwen3-Embedding-0.6B)

---

## Current Architecture

```
score_problem(query, output)
  → length_reward(output)                    # cheap
  → format_reward(output)                    # cheap
  → _eval_product / _eval_shop / _eval_voucher
      → get_product(id)                      # HTTP to search-server (fast, local network)
      → _set_eval_score(product, score, reward)
          → ground_truth_reward(product, reward)     # cheap (ID comparison)
          → rule_score_reward(product, reward)        # EXPENSIVE
              → model.encode([product_title])         # ~1-20s depending on hardware
              → for each GT title:
                  → model.encode([gt_title]) or use precomputed
                  → model.similarity(...)
              → price/service/sku comparisons          # cheap
              → returns (rule_score, total_counter, hit_counter)
```

### Problems

1. **GT match still runs embeddings**: When product_id matches reward product_id, the function returns 1.0 at line 125 regardless, but embeddings still compute at lines 43-60 for the title counter accuracy
2. **Product title re-encoded per GT title**: In the precomputed path, `model.encode([product["title"]])` is called once per reward title in the loop — redundant since the product title doesn't change
3. **Shop/voucher score sequentially**: For shop tasks with N products, scoring is N sequential calls to `rule_score_reward`, each potentially running `model.encode`. All product titles could be batch-encoded in one call
4. **No separation of concerns**: Rule score computation, field counting, and embedding logic are all interleaved in one 100-line function

### Key Insight: When Embeddings Matter

Embeddings are ONLY used for **title field matching** — comparing the agent's recommended product title against ground truth product titles via cosine similarity ≥ 0.7.

On a GT match (agent found the exact right product), the rule score is 1.0 and the title field will obviously match (it's the same product). So embeddings are provably unnecessary on GT match — we can count title as a hit without computing similarity.

On a non-GT match, embeddings are needed to check if the agent's wrong product has a similar title to the correct one.

### Data: How Often GT Match Occurs

From Yuma's slow production run: 10/14 scored problems (71%) were GT matches that ran embeddings unnecessarily. At ~20s per encode on slow hardware, that's ~200s wasted per run.

---

## Refactored Architecture

```
rule_score_reward(product, reward)
  → ground_truth_reward()                    # cheap
  → IF GT match:
      → title: count as hit (skip embedding)  # FREE
  → ELSE:
      → title: encode + compare              # expensive but necessary
  → price/service/sku comparisons            # cheap (unchanged)
  → return (rule_score, total_counter, hit_counter)
```

For shop/voucher with N products:
```
_eval_shop(score, output, rewards)
  → get_product() for each product           # fast (local HTTP)
  → separate GT-match products from non-GT
  → batch encode all non-GT product titles in ONE model.encode() call
  → for each product:
      → rule_score_reward(product, reward, product_title_emb=precomputed_emb)
```

---

### Task 1: Skip title embeddings on GT match in rule_score_reward

**Files:**
- Modify: `src/agent/rewards/orm.py:32-131`
- Test: `tests/test_scoring_perf.py` (create)

The key change: on GT match, the title field should count as a full hit without running embeddings. This is safe because:
- GT match means same product_id → same product → same title → similarity would be 1.0
- The return value is 1.0 regardless (line 125)
- The counters need to reflect the hit, but don't need the actual similarity value

- [ ] **Step 1: Write failing test for GT match skipping**

```python
# tests/test_scoring_perf.py
from unittest.mock import patch, MagicMock
from src.agent.rewards.orm import rule_score_reward

def test_gt_match_skips_embedding():
    """GT match should not call model.encode()."""
    product = {"product_id": "123", "title": "Test Product", "price": 10.0, "service": [], "sku_options": {}, "attributes": {}}
    reward = {"product_id": "123", "title": ["Test Product"], "price": [{"between": [5, 15]}]}

    with patch("src.agent.rewards.orm._get_sentence_model") as mock_model:
        mock_model.return_value = MagicMock()
        rule_score, total_counter, hit_counter = rule_score_reward(product, reward)

        # Should return 1.0 (GT match)
        assert rule_score == 1

        # Should NOT have called encode
        mock_model.return_value.encode.assert_not_called()

        # Title counter should still reflect a hit
        assert hit_counter["title"] == len(reward["title"])
        assert total_counter["title"] == len(reward["title"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.11 -m pytest tests/test_scoring_perf.py::test_gt_match_skips_embedding -v`
Expected: FAIL — encode IS called on GT match currently

- [ ] **Step 3: Write test for non-GT still uses embeddings**

```python
def test_non_gt_still_computes_embeddings():
    """Non-GT match should call model.encode() for title comparison."""
    product = {"product_id": "999", "title": "Wrong Product", "price": 10.0, "service": [], "sku_options": {}, "attributes": {}}
    reward = {"product_id": "123", "title": ["Correct Product"]}

    mock_encode = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    mock_similarity = MagicMock(return_value=[[0.5]])
    mock_model = MagicMock()
    mock_model.encode = mock_encode
    mock_model.similarity = mock_similarity

    with patch("src.agent.rewards.orm._get_sentence_model", return_value=mock_model):
        rule_score, total_counter, hit_counter = rule_score_reward(product, reward)

    # Should have called encode for the product title
    assert mock_encode.called
```

- [ ] **Step 4: Implement GT match skip in rule_score_reward**

In `src/agent/rewards/orm.py`, change the title comparison block:

```python
def rule_score_reward(product: dict, reward: dict) -> tuple[float, Counter, Counter]:
    total_count = 0
    hit_count = 0
    total_counter = Counter()
    hit_counter = Counter()

    is_ground_truth = ground_truth_reward(product, reward) == 1

    # title
    if "title" in reward:
        if is_ground_truth:
            # GT match — same product, title similarity is guaranteed.
            # Count as hit without expensive embedding computation.
            for title in reward["title"]:
                total_count += 1
                total_counter["title"] += 1
                hit_count += 1
                hit_counter["title"] += 1
        else:
            # Non-GT — compute title similarity via embeddings
            model = _get_sentence_model()
            precomputed = reward.get("_title_embeddings", {})
            if model is not None or precomputed:
                product_emb = model.encode([product["title"]]) if model is not None else None
                if product_emb is not None:
                    for title in reward["title"]:
                        if title in precomputed:
                            gt_emb = [precomputed[title]]
                        elif model is not None:
                            gt_emb = model.encode([title])
                        else:
                            continue
                        sim = model.similarity(product_emb, gt_emb)[0][0]
                        total_count += 1
                        total_counter["title"] += 1
                        if sim >= 0.7:
                            hit_count += 1
                            hit_counter["title"] += 1

    # price, service, sku comparisons unchanged...
```

- [ ] **Step 5: Run tests to verify both pass**

Run: `python3.11 -m pytest tests/test_scoring_perf.py -v`
Expected: Both tests PASS

- [ ] **Step 6: Write test verifying scoring output is unchanged for non-GT**

```python
def test_non_gt_scoring_unchanged():
    """Verify non-GT scoring produces same results as before refactor."""
    product = {"product_id": "999", "title": "Blue Phone Case", "price": 25.0,
               "service": ["free shipping"], "sku_options": {}, "attributes": {}}
    reward = {"product_id": "123", "title": ["Red Phone Case"],
              "price": [{"between": [20, 30]}], "service": ["free shipping"],
              "_title_embeddings": {}}

    # Use a real-ish mock that returns consistent embeddings
    import numpy as np
    mock_model = MagicMock()
    mock_model.encode = MagicMock(side_effect=lambda x: np.random.rand(len(x), 384))
    mock_model.similarity = MagicMock(return_value=np.array([[0.65]]))  # Below threshold

    with patch("src.agent.rewards.orm._get_sentence_model", return_value=mock_model):
        rule_score, total_counter, hit_counter = rule_score_reward(product, reward)

    # Price should match (25 is between 20-30)
    assert hit_counter["price"] == 1
    # Service should match
    assert hit_counter["service"] == 1
    # Title should NOT match (similarity 0.65 < 0.7)
    assert hit_counter["title"] == 0
```

- [ ] **Step 7: Run all tests**

Run: `python3.11 -m pytest tests/test_scoring_perf.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/agent/rewards/orm.py tests/test_scoring_perf.py
git commit -m "fix: skip title embedding on GT match in rule_score_reward (ORO-669)

GT match means same product — title similarity is guaranteed 1.0.
Skip model.encode() and count title as hit directly.
Saves ~20s per GT-match problem on slow validators."
```

---

### Task 2: Batch encode product titles for shop/voucher tasks

**Files:**
- Modify: `src/agent/rewards/orm.py` (add `product_title_emb` parameter)
- Modify: `src/agent/problem_scorer.py:228-360` (`_set_eval_score`, `_eval_shop`, `_eval_voucher`)
- Test: `tests/test_scoring_perf.py` (extend)

For shop/voucher tasks with N products, encode all N product titles in one `model.encode()` call instead of N separate calls.

- [ ] **Step 1: Add product_title_emb parameter to rule_score_reward**

In `src/agent/rewards/orm.py`, the non-GT title block should accept an optional pre-encoded embedding:

```python
def rule_score_reward(
    product: dict, reward: dict, product_title_emb=None,
) -> tuple[float, Counter, Counter]:
    # ... existing code ...

    # In the non-GT title block:
    else:
        model = _get_sentence_model()
        precomputed = reward.get("_title_embeddings", {})
        if model is not None or precomputed:
            # Use pre-encoded embedding if provided, otherwise encode now
            if product_title_emb is None and model is not None:
                product_title_emb = model.encode([product["title"]])[0]
            if product_title_emb is not None:
                for title in reward["title"]:
                    # ... similarity comparison using product_title_emb ...
```

- [ ] **Step 2: Add product_title_emb parameter to _set_eval_score**

```python
def _set_eval_score(self, product, score, reward, product_title_emb=None):
    score["product"] += 1
    score["gt"] += ground_truth_reward(product, reward)
    rule_score, total_counter, hit_counter = rule_score_reward(
        product, reward, product_title_emb=product_title_emb
    )
    # ... rest unchanged ...
```

- [ ] **Step 3: Batch encode in _eval_shop**

```python
def _eval_shop(self, score, output, reward):
    product_id_list = self._extract_recommended_product(output)

    # Fetch all products first
    products = [get_product(product_id_list[i]) if i < len(product_id_list) else None
                for i in range(len(reward))]

    # Collect non-GT product titles for batch encoding
    non_gt_titles = []
    non_gt_indices = []
    for i, (product, sub_reward) in enumerate(zip(products, reward)):
        if product and ground_truth_reward(product, sub_reward) != 1:
            non_gt_titles.append(product["title"])
            non_gt_indices.append(i)

    # Batch encode all non-GT product titles in one call
    title_embs = {}
    if non_gt_titles:
        model = _get_sentence_model()
        if model is not None:
            encoded = model.encode(non_gt_titles)
            for idx, emb in zip(non_gt_indices, encoded):
                title_embs[idx] = emb

    # Score each product with pre-encoded embeddings
    num_hits = 0
    shop_ids = set()
    for i, sub_reward in enumerate(reward):
        product = products[i] if i < len(products) else None
        if not product:
            continue
        rule_score = self._set_eval_score(
            product, score, sub_reward,
            product_title_emb=title_embs.get(i),
        )
        if rule_score > 0:
            num_hits += 1
            shop_ids.add(product["shop_id"])

    # ... normalization unchanged ...
```

- [ ] **Step 4: Same pattern for _eval_voucher**

Apply the same batch encoding pattern to `_eval_voucher`.

- [ ] **Step 5: Write test for batch encoding**

```python
def test_shop_batch_encodes_titles():
    """Shop task should batch encode non-GT product titles."""
    # Mock 3 products, 1 GT match + 2 non-GT
    # Verify model.encode is called once with 2 titles, not twice with 1 each
```

- [ ] **Step 6: Run all tests**

Run: `python3.11 -m pytest tests/test_scoring_perf.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/agent/rewards/orm.py src/agent/problem_scorer.py tests/test_scoring_perf.py
git commit -m "perf: batch encode product titles for shop/voucher tasks (ORO-669)

For shop/voucher tasks with N products, encode all non-GT product
titles in a single model.encode() call instead of N separate calls.
Combined with GT skip from previous commit, this significantly
reduces encoding overhead on slow validators."
```

---

### Task 3: Remove SLOW_SCORING_DELAY test code

**Files:**
- Modify: `src/agent/problem_scorer.py` (remove test delay)

- [ ] **Step 1: Remove SLOW_SCORING_DELAY from problem_scorer.py**

Remove the artificial delay code added for testing ORO-669 reproduction. It should not ship to production.

- [ ] **Step 2: Commit**

```bash
git add src/agent/problem_scorer.py
git commit -m "chore: remove SLOW_SCORING_DELAY test instrumentation"
```

---

## Verification

After all tasks, run end-to-end:

1. `python3.11 -m pytest tests/test_scoring_perf.py -v` — all scoring tests pass
2. `ruff check src/agent/rewards/orm.py src/agent/problem_scorer.py` — lint clean
3. Local test run: `docker compose run --rm test -- --agent-file <agent> --problems <problems>` — verify scores match expected values
4. Compare scores before/after on a known agent to confirm no behavioral change
