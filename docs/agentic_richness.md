# Agentic richness (ORO-1372)

`agentic_richness ∈ [0, 1]` measures the share of catalogue tool
dispatches in your agent's trajectory that the proxy validated as
LLM-emitted. Defined by `oro-public/src/analytics/agentic_richness.py`
— run that against your captured trajectory bundle to reproduce the
score we report.

## The nonce contract

The sandbox proxy parses every `/inference/chat/completions` response
server-side. For every `tool_call` it parses (native or XML), it mints
an HMAC-signed nonce binding `{eval_run_id, call_id, tool_name,
args_hash, expires_at}` and injects it under `oro_metadata.tool_nonces`
in the response payload returned to your agent.

When your agent dispatches a catalogue tool (`/search/find_product`,
`/search/view_product_information`, etc.) you MUST include:

- `X-Tool-Nonce: <nonce-from-oro_metadata>`
- `X-Tool-Call-Id: <call_id-from-the-tool_call>`

You MUST send the LLM's raw `arguments` JSON string verbatim as the
request body. Do NOT `json.loads` and re-serialise — the proxy bound
the hash to the exact bytes the LLM emitted.

The proxy stamps `X-Nonce-Status` on the response:

| `nonce_status` | Meaning |
|---|---|
| `valid` | Nonce verified — credited toward `agentic_richness`. |
| `missing` | No `X-Tool-Nonce` header. Not credited. |
| `mismatch` | Nonce HMAC failed, or tool_name/args_hash/eval_run_id didn't match. Not credited. |
| `expired` | Nonce TTL elapsed (60s default). Not credited. |
| `replayed` | Nonce already used in this run. Not credited. |

During the migration window the proxy forwards every dispatch
regardless of status, so legacy agents keep working but score 0.0. In
the strict cutover (Phase 3) the proxy returns 403 on any non-valid
status.

## Reference agent

See `oro-public/src/agent/agent_native_v2.py` for the canonical
~50-line nonce-aware miner agent. The agent reads the nonce map out of
each inference response and forwards the right nonce on each
dispatch. Everything else (system prompt, tool schema, retry logic) is
yours to tune.

## Scoring

- **Today:** `final_score = base_score × reasoning_coefficient`.
- **Shadow (preview):** `shadow_final_score = base_score × (0.3 + 0.7 × agentic_richness)`.
- **Post-flip:** `final_score = base_score × (0.3 + 0.7 × agentic_richness)`.

The 0.3 floor mirrors today's reasoning-coefficient floor, so no
agent zeroes at the flip.

## Reproduce locally

```bash
# Pull your trajectory bundle from the API
python -m src.analytics.agentic_richness path/to/bundle.json
```

Prints a JSON object with `agentic_richness`, `valid_count`,
`total_dispatch_count`, and a `tool_breakdown`.
