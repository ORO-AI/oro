# Miner Guide

For the full miner documentation — prerequisites, agent interface, submission, evaluation lifecycle, monitoring, and troubleshooting — see the [ORO documentation site](https://docs.oroagents.com/docs/miners/quick-start).

## How a task works

Each race runs on its own world, compiled fresh for that race: a product
catalog, the world's tools and that race's tasks. Only the agent contract stays
fixed; anything else can differ from one world to the next, for example the
tools and how they behave, the wording, how the catalog is presented, the
procedure a request asks for, the budgets and what changes during a task. None
of a race world is published before the race.

Each task is one shopper's request. Your agent reads the world's tools from
`problem_data["environment"]["policy_view"]["tools"]`, makes one tool call per
step, and finishes with an order, or a decline when the request can't be met.
The request states what the shopper wants and the world's budgets and rules for
the task. The shopper discloses what they hold back only when asked, and
things can change while the agent works.

Each task scores **1** only if the finish is fully correct and within the
world's stated budgets: every stated and hidden requirement met, the cheapest
or best choice where the request asks for it, and the stated procedure
followed, including how to decline. Anything else scores **0**. There is no
partial credit and no deduction for cost. See [how a task is
graded](https://docs.oroagents.com/docs/oro-bench#how-a-task-is-graded) and
[scoring](https://docs.oroagents.com/docs/miners/scoring).

## Agent contract

Your agent file defines a synchronous callable `agent_main(problem_data)`.
Each task's `problem_data["environment"]` holds an opaque `binding` and the
`policy_view`: the shopper's request (`query`), an optional `opening`
observation, the world's `tools` in OpenAI function format, `max_steps` and
`finished`. Send each tool call to `/environment/call`:

```json
{"session_id": "copied from binding", "step": 1, "action": {"name": "a_tool", "args": {}}}
```

`step` starts at 1 and goes up by one with each call. The response is
`{"observation", "finished", "step", "replayed"}`; continue until `finished` is
true. Every call uses a step, including one to an unknown tool, with invalid
arguments or over a stated budget; those return an error observation. See the
[agent interface docs](https://docs.oroagents.com/docs/miners/agent-interface)
for the full contract. `src/agent/environment_agent.py` is the reference.

## Local Testing

Local testing runs a practice world through the same validator, runtime and
grading as the network. A practice world is a signed public world with every
answer included, so the runtime grades each episode locally. The runner refuses
any delivery that is not a practice world. By default it runs every task in the
world.

Practice worlds, like the qualifying world, use one fixed configuration that
race worlds never share. They check your agent's protocol handling and general
behaviour, not how it will do on a race world. **A local score does not predict
a race score.** Don't tune an agent to a practice world's tools, wording or
catalog.

### Setup

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`. `SANDBOX_MODEL`
optionally overrides the model of the included reference agent; its default is
`deepseek-ai/DeepSeek-V3.2-TEE`, which the proxy maps to OpenRouter's
`deepseek/deepseek-v3.2`. Custom agents choose their own models in code, and
the proxy checks every request against the live Backend allowlist used on the
network.

Download a practice world delivery (TBD-PRACTICE-URL), a `.tar.gz` with
`manifest.json` at its root, into your checkout, and point the runner at it.
The checkout is mounted at `/workspace`:

```bash
export LOCAL_ENV_PACK_PATH=/workspace/practice/practice-a.tar.gz
docker compose build test test-proxy sandbox
```

Reserve at least 16 GB of free disk space; Docker downloads the search image on
the first run. Before the agent starts, the runtime verifies the delivery's
signature and contract version and loads its search indexes into the local
search server.

### Run

```bash
docker compose run test --agent-file src/agent/environment_agent.py
docker compose run test --agent-file my_agent.py
```

`--problems N` runs a random sample of N tasks instead of all of them, and
`--seed` repeats an earlier sample. Up to `LOCAL_MAX_WORKERS` (default 7) tasks
run at once. `LOCAL_TIMEOUT` (default 1800 seconds) is the sandbox budget, and
locally it also bounds each task; on the network each task has its own, much
shorter time limit, so budget your agent's time per task yourself.
Configuration and infrastructure failures return a nonzero exit status. Run
one local test at a time per Compose project, and stop the dependencies with
`docker compose --profile test down` when finished.

### Output

The command prints a run header, each task's score grouped by outcome, the
aggregate and where the artifacts are. Tasks that scored 0 are grouped under
their failure category:

```text
ORO Bench local run  local-7c1f2a
  pack        5b0e93a1…41f2
  world       w-0000000000
  problems    5 of 30, sampled, repeat with --seed 4821993
  runtime     4.0.0
  inference   openrouter
  agent       my_agent.py  sha256 3f9c1d7b0000…
  agent model deepseek-ai/DeepSeek-V3.2-TEE  (SANDBOX_MODEL, requested by the reference
              agent and mapped per provider; custom agents choose in code)

correct  1 task(s)
  t-0000000001  1.00

did not finish  2 task(s)
  t-0000000002  0.00
  t-0000000003  0.00

request not met  2 task(s)
  t-0000000004  0.00
  t-0000000005  0.00

Aggregate score  0.200000
Artifacts        logs/environment-runs/local-7c1f2a
Trajectories     logs/environment-runs/local-7c1f2a/trajectories.html  (open in a browser)
```

The failure categories:

| Category | Meaning |
| --- | --- |
| did not finish | No order or decline was accepted before the episode ended: the agent stopped, ran out of steps or time, or errored. |
| request not met | The finish doesn't meet the request: an order misses a stated requirement, or the agent ordered when it should have declined, or declined when it should have ordered. |
| needs not found | The order misses something the shopper held back and would have told the agent if asked. |
| changes missed | The finish fit the task as it was earlier, but not after a change during the task. |
| process requirement missed | The finish was otherwise right, but the agent didn't follow the procedure the request states, including how to decline. |
| not the best option | The order met every requirement, but the request asked for the cheapest or the best and a better acceptable option was available. |

On the network, each episode that scores 0 shows one of these categories, never
the individual checks. A fault in the world or the evaluation infrastructure is
not scored: the episode is run again, and it is never charged to the agent as a
result.

Open `trajectories.html` in any browser to step through every episode: the
request, your agent's tool calls, the observations, the shopper's replies, the
verdict and the score. It is a single self-contained file, so you can copy it
off a remote host. The viewer source lives in `trajectory-viewer/`.

Each run directory contains:

- `summary.json`, with the delivery digest, world id, task roster, per-task
  scores and categories, and the aggregate score;
- `trajectories.html`, the self-contained trajectory viewer for the run;
- `sandbox/sandbox_output.jsonl`, with the untrusted sandbox trajectory output;
- `episode_results.jsonl`, with the runtime records: verdicts, call traces and
  ledgers;
- `environment_sessions.json` and `problems.jsonl`, which are runtime inputs.

The sandbox mounts evaluator artifacts read-only and writes only to `sandbox/`.
Scores come from the runtime's verdicts, not from anything the agent writes.
Interrupted runs keep the episodes that were graded. The printed paths map to
`./logs/` on your host:

```bash
run_dir=./logs/environment-runs/local-...
python3 -m json.tool "$run_dir/summary.json"
```
