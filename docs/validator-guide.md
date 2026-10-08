# Validator Deployment Guide

For the full validator documentation — hardware requirements, installation, configuration, running, auto-updates, systemd service, monitoring, and troubleshooting — see the [ORO documentation site](https://docs.oroagents.com/docs/validators/overview).

## Quick Start

```bash
git clone https://github.com/ORO-AI/oro.git
cd oro
cp .env.example .env   # Configure wallet name

# Register on the subnet
btcli subnet register --netuid 15 --wallet.name my-validator --wallet.hotkey default

# Start the validator
WALLET_NAME=my-validator docker compose --profile validator up -d
```

See the [full guide](https://docs.oroagents.com/docs/validators/overview) for detailed setup instructions.

## Simulator inference

Simulator inference uses asynchronous HTTP requests so concurrent calls can overlap and canceled calls release their connections. Response and reword phases share the registry's simulator deadline, which cancels pending async requests when exhausted. Completed responses retain usage accounting; canceled requests are logged separately because their upstream usage may be unknown. The completion adapter accepts an optional `reasoning` boolean or object and a structured `response_format`; both are omitted when no preference is supplied. Authenticated validator requests using a JSON schema require OpenRouter providers to support its parameters.

For simulator requests to Chutes Qwen3.5-397B-A17B, the adapter translates `reasoning: {"enabled": false}` to its documented non-thinking setting, `chat_template_kwargs: {"enable_thinking": false}`. Other simulator reasoning preferences on Chutes fail explicitly. Ordinary agent requests retain their own provider parameters. See the [official Qwen model card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) and [Chutes model guide](https://chutes.ai/docs/models/chutes-qwen-qwen3-5-397b-a17b-tee).

All shopper simulation, including configured disclosure Readers and typed decisions, uses the active evaluation’s miner-funded inference credential. The same run-scoped proxy grant and provider restrictions apply as for existing simulator calls. No separate validator credential is required. Pack-selected models must be available through the miner’s provider and its existing model allowlist. Exhausting the miner’s inference budget remains an agent error; it stops further work for that run.

With runtime 3.5, configured Reader tasks require OpenRouter’s Jev decisions endpoint and cannot run through Chutes. Ordinary legacy qualifying tasks without a configured Reader are unaffected.

Any environment or verifier failure on a selected configured Reader task fails the run through the infrastructure failure path. Failures on ordinary tasks retain the existing infrastructure threshold, including in mixed rosters. Completed low-reward episodes and ordinary agent errors continue to count normally.
