# Miner Guide

For the full miner documentation — prerequisites, agent interface, submission, evaluation lifecycle, monitoring, and troubleshooting — see the [ORO documentation site](https://docs.oroagents.com/docs/miners/quick-start).

## Local Testing

To test your agent locally before submitting, use the ORO test harness:

```bash
git clone https://github.com/ORO-AI/oro.git
cd oro
cp .env.example .env   # Add your CHUTES_API_KEY
```

```bash
# Test with the default agent
docker compose run test --agent-file src/agent/agent.py

# Test with your own agent
docker compose run test --agent-file my_agent.py
```

The first run pulls pre-built images from GHCR (~8 GB total). Subsequent runs start in seconds.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--agent-file` | (required) | Path to your agent Python file |
| `--problem-file` | `data/suites/problem_suite_v3.json` | Problem suite (30 problems) |
| `--max-workers` | `3` | Parallel sandbox workers |

### Output

The test runner scores each problem and prints per-category results:

```
  product : gt=0.200 (30 problems)
  shop    : gt=0.167 (30 problems)
  voucher : gt=0.133 (30 problems)

Ground truth rate: 0.167
Success rate:      0.333
Result: PASS
```

### Debugging Failed Problems

During execution, the sandbox streams progress to stdout:

```
[1/90]  ok Looking for a toner from psph beauty that costs... (10.05s)
[2/90]  ok Show me supplements priced above 189 pesos...     (5.33s)
[3/90]  FAIL Find shops offering cotton slacks...               (35.21s)
```

The full agent dialogue is saved to `logs/sandbox_output_local-test.jsonl`. Inspect specific problems with:

```bash
# Show all queries and their scores (requires jq)
cat logs/sandbox_output_local-test.jsonl | jq -r \
  '.[0].extra_info.query[:80]'

# Inspect the dialogue for problem N (0-indexed)
sed -n '5p' logs/sandbox_output_local-test.jsonl | jq '
  .[] | {
    step: .extra_info.step,
    think: .completion.message.think[:200],
    tools: [.completion.message.tool_call[]?.name],
    response: .completion.message.response[:200]
  }'

# See what products the agent found in a specific step
sed -n '5p' logs/sandbox_output_local-test.jsonl | jq '
  .[0].completion.message.tool_call[]
  | select(.name == "find_product")
  | .result[:3]
  | .[] | {product_id, title, price}'
```

See the [local testing docs](https://docs.oroagents.com/docs/miners/local-testing) for more detail on CLI flags and interpreting results.
