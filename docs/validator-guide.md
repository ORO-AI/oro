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

Tasks containing a configured disclosure Reader require a distinct validator-owned OpenRouter credential supplied through `ORO_SIMULATOR_ACCESS_TOKEN`. It is kept outside the logged configuration and is never passed to the sandbox. All shopper simulation for those tasks uses this credential. The proxy accepts it only for authenticated validator requests during an active, unexpired evaluation grant. Missing, non-OpenRouter or miner-matching credentials reject session setup; an exhausted owner credential is an infrastructure failure. Tasks without a configured Reader retain their existing miner-funded simulation.

Any environment or verifier failure on a selected configured Reader task fails the run through the infrastructure failure path. Failures on ordinary tasks retain the existing infrastructure threshold, including in mixed rosters. Completed low-reward episodes and ordinary agent errors continue to count normally.
