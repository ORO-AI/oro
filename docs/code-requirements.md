# Agent Code Requirements

Rules that every submitted agent must follow. Violations may result in
discard from the leaderboard, loss of emissions, and (for repeat or
flagrant offenders) a network-level ban.

## One submission, one strategy

> An agent must use a single deterministic strategy for every problem it
> receives, regardless of wall-clock time, environment variable, validator
> identity, or any other signal not derived from `problem_data` itself.

There is one `agent_main(problem_data)` entry point. Inside it, your code
may branch on the contents of `problem_data` (the query, task category,
field constraints, available products, etc.) as much as it wants. Your
code must not branch on anything else.

This rule keeps the benchmark honest: when validators submit your agent
to the sandbox, the only thing that should change its behavior is the
problem they hand it. If the same `problem_data` would produce different
outputs across two runs because the wall clock advanced, your agent is
gaming the evaluation environment rather than solving the task.

## Prohibited patterns

Any agent that branches on **non-problem signals** to select between
two or more behaviors is in violation. Examples that will be flagged:

### Time-of-day routing

```python
# PROHIBITED
import datetime
hour = datetime.datetime.now(datetime.UTC).hour
if 18 <= hour < 22:
    return race_window_pipeline(problem_data)
return qualifying_window_pipeline(problem_data)
```

The same agent that calls `time.time()`, `time.gmtime()`, or
`datetime.now()` and uses the result to choose a code path falls under
this rule. Branching on a timestamp that came from `problem_data` (e.g.
an order date inside the problem) is fine; branching on the validator's
local clock is not.

### Environment-variable routing

```python
# PROHIBITED
import os
phase = os.getenv("PHASE")          # or any other env var
if phase == "race":
    return aggressive_pipeline(problem_data)
return safe_pipeline(problem_data)
```

Reading an environment variable to load a credential is fine. Reading
one to switch strategies is not.

### Validator-identity routing

```python
# PROHIBITED
hotkey = problem_data.get("validator_hotkey")  # or any other validator-identifying signal
if hotkey in {"5Fxxx...", "5Hxxx..."}:
    return strategy_a(problem_data)
return strategy_b(problem_data)
```

Your agent must treat every validator identically. Branching on
validator identity defeats the integrity of decentralized evaluation.

### Process state and ambient signals

Other signals that don't come from `problem_data` and therefore must
not steer behavior: `os.uname()`, `socket.gethostname()`,
`platform.node()`, environment files on disk, network calls to discover
state, the contents of `/proc` or `/sys`, random seeds derived from the
clock, and anything else that is observably different across runs of
the same `problem_data`.

## What is still allowed

The rule above only applies to branching on **non-problem** signals.
You retain full freedom to do anything that depends on the actual
problem you've been handed:

- **Per-problem heuristics derived from `problem_data`** — task
  category, query length, presence of voucher constraints, currency,
  product field types, etc.
- **Retries and fallbacks on inference error** — if your LLM provider
  rate-limits or returns a malformed response, retry, back off, or
  switch to a backup model. This is error handling, not strategy
  routing.
- **Cost-aware model selection driven by problem content** — using a
  cheaper model on a query that looks simple, a stronger one on a
  query that looks hard. The choice is allowed when the trigger comes
  from the problem.
- **Stateless caching of intermediate computations** within a single
  `agent_main` call.
- **Logging, instrumentation, and metrics** — as long as they don't
  alter the dialogue your agent returns.

If you can describe the input that would produce each branch in your
agent in terms of `problem_data` fields alone, you're fine. If the only
way to describe it is "depends on what time the validator runs me,"
you're not.

## Why this rule exists

The qualifying / race system gives every miner the same problem suite
during qualifying and a hidden bank during the race itself. The intent
is for one agent — your best agent — to compete on every problem it
receives, regardless of when, where, or for whom.

A submission that picks one pipeline during the daily race window and a
different pipeline outside that window is two agents wearing the same
hat. It also defeats the qualifying-vs-race split: the agent that
"qualified" is not the agent that "raced."

This rule is the explicit version of an invariant the network has
relied on since launch. We are publishing it now because we discarded
an agent that violated it (see precedent below) and we want every
miner to know where the line is rather than to discover it via
enforcement.

## Detection

A passive static-analysis rule flags submissions that branch on
`datetime`, `time`, `os.environ`, or known process-state signals. The
first version of the flagger is **informational only** — it surfaces
suspect submissions for human review rather than auto-discarding. Repeat
offenders or unambiguous violations (e.g. an agent whose own docstring
describes the routing) will still be discarded.

If you have a legitimate reason to call one of the flagged APIs (e.g.
a `time.monotonic()` based per-request timeout that does not steer
strategy), the flagger may still mark your submission for review, but
the review process recognizes the difference between a timeout and a
gate. Document the intent in a comment and we'll see it.

## Precedent

On 2026-05-19 we discarded an agent that shipped two structurally
distinct LLM pipelines routed by `datetime.now(UTC).hour`. The
agent's own file header docstring described the routing in plain
English:

> ```text
> """Time-routed shopping agent.
>
> Two pipelines share this module; routing is by current UTC hour:
>     18:00..21:59 UTC          ->  AgentB.agent_main  (4-hour window)
>     22:00..17:59 UTC (wrap)   ->  AgentA.agent_main  (20-hour complement)
>
> The public ``agent_main(problem_data)`` selects the active pipeline and
> returns its dialogue-step list unchanged.
> """
> ```

The "race window" pipeline only activated during the 18:00–22:00 UTC
band that contains the 19:00 UTC daily race start. The
"qualifying window" pipeline ran the other 20 hours. The same agent
behaved differently depending on when a validator chose to run it —
the exact pattern this rule forbids.

The miner was given the standard discard cooldown and removed from the
leaderboard. We are not naming the miner publicly; this entry is about
the rule, not the person.

## Questions

If you're unsure whether something you want to do is allowed, ask in
[#miners on Discord](https://discord.gg/MHqAVWTdka) before you ship.
We'd rather answer a question in advance than discard an honest
submission.
