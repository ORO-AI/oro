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
