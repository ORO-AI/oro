# Miner Guide

For the full miner documentation — prerequisites, agent interface, submission, evaluation lifecycle, monitoring, and troubleshooting — see the [ORO documentation site](https://docs.oroagents.com/docs/miners/quick-start).

## Local Testing

The local workflow validates the bundled practice EnvPack, then runs its first
five problems. Network qualifying runs the active suite's full roster, so a
local score is not a qualifying score. It uses the generated
validator's `oro-env-runtime` sessions, the composed evaluator, rewards, proxy,
search server, and sandbox. It does not compile tasks or fetch evaluation work
from the Backend. The proxy reads the public Backend model allowlist, matching
the qualifying inference path.

The practice EnvPack is included at `data/local-test/env-pack.tar.gz` using Git
LFS. It holds qualifying rows only: the runner refuses a pack with any race row,
so local scores never come from race material. The local runner verifies the
pack's digest, its runtime contracts, and the matching search-index identity
before it starts the agent sandbox, and prints them in the run header.
Your agent file must define a synchronous callable
`agent_main(problem_data)` that drives the environment session via
`problem_data["environment"]["binding"]` and `policy_view`. See the [agent
interface docs](https://docs.oroagents.com/docs/miners/agent-interface) for the
full contract.

Start from the bundled reference:

```bash
docker compose run test --agent-file src/agent/environment_agent.py
```

Once you have your own file (started from a copy of `environment_agent.py`),
swap it in with `--agent-file my_agent.py`.

> **Legacy `agent_main(task) -> List[Dict]`** — this is the old ShoppingBench
> contract. Production evaluations no longer use it; an agent that never opens
> a session against `binding.session_id` terminates as `agent_error` on every
> task with zero score. Migrate to `agent_main(problem_data)` before submitting.

### How a task is scored

Each task is a shopper situation. The goal states some requirements; the
shopper reveals other needs only when asked through the `message` tool; the
shopper or the market (price, stock) can change during the task; and some tasks
require something of the process, such as answering the shopper's question or
backing the order with grounded claims when the shopper asks for a reason. An
order that misses any requirement or obligation scores 0. When no listing meets
every requirement, the task ends with `place_test_order` and `abstain: true`
instead: nothing is ordered, `product_id`/`sku` name the closest alternative,
and the justification claims state the facts that show why it falls short.
Ordering on such a task, or abstaining when a listing fits, scores 0. A
passing order earns
up to 1.0, less for questions beyond what the order needed and for a
less-than-best pick by the shopper's stated priority. On the network, each
episode's feedback names one failure category (below), never the individual
checks; locally the trajectory viewer shows every check. See [how a task
works](https://docs.oroagents.com/docs/oro-bench#how-a-task-works) and
[scoring](https://docs.oroagents.com/docs/miners/scoring).

### Tool argument handling

Use only the tool names and top-level arguments listed in
`policy_view.tools[].function.parameters.properties`. Qualifying, race, and
local validator sessions remove undeclared top-level arguments before tool
execution. Declared arguments retain the runtime's existing defaulting,
coercion, and bounds behavior.

For example, `max_price` is declared on `filter`, not `search`. A search call
that includes it still runs without that argument; use the filter tool for a
price-constrained catalog lookup. Search still uses BM25 internally. The
sandbox cannot call the shared search server directly, and legacy `/search/*`
proxy routes return 410.

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

The `test-search-server` service follows the shared `IMAGE_TAG`, which defaults
to `stable`, alongside the published sandbox and proxy images. The validator
is built from the checkout. Set `IMAGE_TAG=latest` to test prerelease
dependencies. For pack-specific validation, `LOCAL_SEARCH_SERVER_IMAGE`
overrides only the search service with an exact tag or digest. Docker downloads
the multi-architecture search image for the host platform on the first run.
Reserve at least 16 GB of free disk space for the image and runtime data. The
runtime rejects a mismatched search identity before starting the agent sandbox.

Before starting the agent, the runtime validates every task and catalog
reference in the bundled qualifying pack.

`LOCAL_ENV_PACK_PATH` can select another pack for development, and
`LOCAL_ENV_PACK_SHA256` can require an expected digest. Pack paths must be
available inside `/workspace`.
`LOCAL_MAX_WORKERS` defaults to 7 and `LOCAL_TIMEOUT` to 1800 seconds.

`--problems` runs a different selection while you iterate. Pass any number
from 1 up to the number of problems in the pack. `--seed` only applies
alongside it. The problems are sampled at random:

```bash
docker compose run test --agent-file my_agent.py --problems 7
```

Each run samples afresh, so repeated runs do not tune the agent against one
lucky subset. The report prints the seed; pass `--seed` to repeat an earlier
selection exactly. Local scores are not comparable to qualifying, which runs
the active suite's full qualifying roster.
Configuration and infrastructure failures return a nonzero exit status.

Local tests use a dedicated search server and proxy. The proxy fetches the live
allowlist from `BACKEND_URL`, which defaults to `https://api.oroagents.com`.
The runtime shares the proxy's network namespace so the original Compose
command works without additional networking flags. Run one local test at a
time per Compose project. Dependencies stay running for subsequent tests; stop
them with `docker compose --profile test down` when finished.

### Output

The command prints a run header, the finalized problems with their rewards,
the aggregate, and where the artifacts are. A failed or partial problem also
names its failure categories: the primary one first, then every category in
parentheses:

```text
ORO Bench local run  local-7c1f2a
  pack        f75b76c8…cd0c
  problems    5 of 30, qualifying roster
  runtime     0.3.4
  inference   openrouter
  agent       my_agent.py  sha256 3f9c1d7b0000…
  agent model deepseek-ai/DeepSeek-V3.2-TEE  (SANDBOX_MODEL, requested by the reference
              agent and mapped per provider; custom agents choose in code)
  simulator   mistralai/mistral-small-2603
  judge       deepseek/deepseek-v4-flash-0731

composed               mean 0.12  1/5 passed
  TF8-composed-700000  completed         0.60  extra_questions (extra_questions)
  TF8-composed-700001  completed         0.00  needs_not_found (needs_not_found)
  TF8-composed-700015  completed         0.00  request_not_met (request_not_met, needs_not_found)
  TF8-composed-700016  completed         0.00  did_not_finish (did_not_finish)
  TF8-composed-700023  completed         0.00  request_not_met (request_not_met, needs_not_found, process_issue)

Aggregate score  0.120000
Artifacts        logs/environment-runs/local-7c1f2a
Trajectories     logs/environment-runs/local-7c1f2a/trajectories.html  (open in a browser)
```

The categories, in priority order:

| Category | Meaning |
| --- | --- |
| `did_not_finish` | Your agent didn't place an order: it stopped, reached the step limit before ordering, or errored. |
| `request_not_met` | The order misses what the shopper asked for: category, budget, stock, a stated requirement, or it is not a catalog listing. |
| `needs_not_found` | Your agent didn't find out everything the shopper wanted: the order misses something the shopper would have told you if asked. |
| `changes_missed` | Your agent didn't keep up with a change during the task, from the shopper or the market. |
| `process_issue` | A shopping-process requirement wasn't met. |
| `not_best_option` | Passed, but a better acceptable item was available by the shopper's stated priority. |
| `extra_questions` | Passed, but more questions than needed reduced the reward. |
| `infrastructure` | An infrastructure problem, not your agent, shown alone. The episode still scores zero in that run, unless enough of the run's tasks fail this way that the whole run is treated as an infrastructure failure instead. |

Rewards are coloured when the output is a terminal; set `NO_COLOR=1` to turn
that off. The simulator and judge models are sealed in the pack. The agent
model line shows `SANDBOX_MODEL`, which is a request rather than a record: only
the included reference agent reads it, and the proxy maps it to the active
provider's name for that model, so an OpenRouter run of the default sends
`deepseek/deepseek-v3.2`. A custom agent chooses its own models in code. A
completed task can still have a zero reward if its verifier verdict is
incorrect.

Open `trajectories.html` in any browser to step through every episode: the
shopper request, your agent's messages and tool calls, observations, simulator
events, the verdict checks, the failure categories, and the reward. It is a single self-contained file,
so you can copy it off a remote host. The viewer source lives in
`trajectory-viewer/`. A failed run writes it too, covering whatever episodes
finished before the failure, and names it in the error output. A run that fails
before any episode finalizes has nothing to show, so it writes no viewer.

Each run directory contains:

- `summary.json`, with the pack digest, task roster, per-task and per-family
  rewards, runtime error classification, and aggregate score;
- `trajectories.html`, the self-contained trajectory viewer for the run;
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
