# Miner Guide

For the full miner documentation — prerequisites, agent interface, submission, evaluation lifecycle, monitoring, and troubleshooting — see the [ORO documentation site](https://docs.oroagents.com/docs/miners/quick-start).

## Local Testing

The local workflow validates the sealed 35-task qualifying EnvPack, then runs
all five tasks from each TF1 through TF7 family. It uses the generated
validator's `oro-env-runtime` sessions, family verifiers, rewards, proxy,
search server, and sandbox. It does not compile tasks or fetch evaluation work
from the Backend. The proxy reads the public Backend model allowlist, matching
the qualifying inference path.

The exact release EnvPack is included at `data/local-test/env-pack.tar.gz`
using Git LFS. The local runner verifies the pack, its runtime contracts, and
the matching search-index identity before it starts the agent sandbox.
Your agent file must define a synchronous callable
`agent_main()`. From the repository root, use the existing command:

```bash
docker compose run test --agent-file my_agent.py
```

For a reference agent, pass `--agent-file src/agent/environment_agent.py`.
It reads `problem_data["environment"]["policy_view"]`, uses the supplied dynamic
tools via `/environment/call`, and continues until `done=true`. Legacy
ShoppingBench agents need to adopt this generated-environment contract.

### Setup and configuration

Keep your existing `.env`, or copy `.env.example` for a new checkout. Set
`OPENROUTER_API_KEY` or `CHUTES_API_KEY`. Both keys may remain configured;
`INFERENCE_PROVIDER=chutes` or `INFERENCE_PROVIDER=openrouter` selects one.
Without an explicit choice, OpenRouter takes precedence when both keys exist.
`SANDBOX_MODEL` is an optional override used by the included reference agent.
Its default is
`deepseek-ai/DeepSeek-V3.2-TEE`, preserving the existing local-testing default.
The proxy maps it to the paired OpenRouter identifier when using OpenRouter.
Custom agents may choose one or more models in their own code. The proxy checks
every request against the same live Backend allowlist used for qualifying. The
shopper simulator keeps its model sealed in the EnvPack and uses that same
proxy path. For example, set
`SANDBOX_MODEL=Qwen/Qwen3.5-397B-A17B-TEE` to run the frontier Qwen model used
for the reference trajectory run.

Install the Git LFS pack and current validator, sandbox, and proxy images.
For a source checkout, build the services after pulling the pack:

```bash
git lfs pull
docker compose build test test-proxy sandbox
```

The `test-search-server` service uses the same promoted `stable` image as
production validators. Docker downloads it on the first run. The runtime
rejects a mismatched search identity before starting the agent sandbox.

Before starting the agent, the runtime validates every task and catalog
reference in the bundled qualifying pack.

`LOCAL_ENV_PACK_PATH` can select another pack for development, and
`LOCAL_ENV_PACK_SHA256` can require an expected digest. Pack paths must be
available inside `/workspace`.
`LOCAL_MAX_WORKERS` defaults to 7 and `LOCAL_TIMEOUT` to 1800 seconds.
Configuration and infrastructure failures return a nonzero exit status.

Local tests use a dedicated search server and proxy. The proxy fetches the live
allowlist from `BACKEND_URL`, which defaults to `https://api.oroagents.com`.
The runtime shares the proxy's network namespace so the original Compose
command works without additional networking flags. Run one local test at a
time per Compose project. Dependencies stay running for subsequent tests; stop
them with `docker compose --profile test down` when finished.

**Note on `find_product`:** the `q` parameter matches against product title and the values within `attributes` / `sku_options`. Field names (keys) themselves are not searchable.

### Output

The command prints 35 finalized task results followed by the aggregate and
artifact location:

```text
intent_decomposition: completed, reward=...
intent_decomposition: completed, reward=...
... 33 more task rows ...
Aggregate score: ...
Artifacts: /app/logs/environment-runs/local-...
```

The family names correspond to TF1 through TF7 in the order shown. A completed
task can still have a zero reward if its verifier verdict is incorrect.

Each run directory contains:

- `summary.json`, with the pack digest, task roster, per-task and per-family
  rewards, runtime error classification, and aggregate score;
- `sandbox/sandbox_output.jsonl`, with the untrusted sandbox trajectory output;
- `episode_results.jsonl`, with finalized runtime receipts including verifier
  verdicts, call traces, ledgers, and provenance;
- `environment_sessions.json` and `problems.jsonl`, which are runtime inputs.

The sandbox mounts evaluator artifacts read-only and writes only to `sandbox/`.
Scores and task diagnostics come from runtime receipts. Self-reported timings,
inference failures, and request logs remain in `sandbox/` for inspection and do
not determine the summary. The output reader rejects symlinks, hard-linked or
non-regular files, non-object rows, and files larger than 128 MiB. Interrupted
runs retain finalized runtime receipts, including partial task outcomes.

The printed `/app/logs/` path maps to `./logs/` on your host. Inspect the summary:

```bash
run_dir=./logs/environment-runs/local-...
python3 -m json.tool "$run_dir/summary.json"
wc -l "$run_dir/sandbox/sandbox_output.jsonl"
```

The artifacts contain sealed task data and agent trajectories. Keep them local
and do not publish them.
