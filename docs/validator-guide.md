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

Simulator inference uses asynchronous HTTP requests so concurrent calls can overlap and canceled calls release their connections. Completed responses retain usage accounting; canceled requests are logged separately because their upstream usage may be unknown. The completion adapter accepts an optional `reasoning` boolean and omits that field when no preference is supplied.
