# Local Chain Subnet Setup

To set up and use the local chain, you can follow the documentation linked here:

https://docs.learnbittensor.org/local-build/deploy

This documentation provides detailed instructions for running a local instance of the Subtensor blockchain, including both Docker-based and source-based setup methods.

---

## Running a Local Miner/Validator Setup

### Prerequisites

1. **Start the Docker services** from the project root:

```bash
cp .env.example .env   # Add your CHUTES_API_KEY
docker compose up -d search-server proxy
```

Wait for all services to be healthy: `docker compose ps`

### Wallet Setup

Before running the miner or validator, you need to create and register wallets on your local chain.

#### Create Wallets

```bash
# Create miner wallet
btcli wallet new_coldkey --wallet.name test-miner-seth
btcli wallet new_hotkey --wallet.name test-miner-seth --wallet.hotkey default

# Create validator wallet
btcli wallet new_coldkey --wallet.name test-validator-seth
btcli wallet new_hotkey --wallet.name test-validator-seth --wallet.hotkey default
```

#### Fund Wallets (Local Chain)

On a local chain, you'll need to fund your wallets with test TAO. Refer to the local chain documentation for instructions on using the faucet or transferring funds from the sudo account.

#### Register on Subnet

```bash
# Register miner on subnet 2
btcli subnet register --wallet.name test-miner-seth --wallet.hotkey default --netuid 2 --subtensor.network local

# Register validator on subnet 2
btcli subnet register --wallet.name test-validator-seth --wallet.hotkey default --netuid 2 --subtensor.network local
```

### Submitting an Agent (Miner)

Submit an agent using the `oro` CLI (from the `oro-sdk` package):

```bash
pip install "oro-sdk[bittensor]"

oro submit \
  --wallet-name test-miner-seth \
  --agent-name "my-agent" \
  --agent-file path/to/agent.py \
  --base-url http://localhost:8000
```

### Running the Validator

Start the validator with the following command:

```bash
cd subnet && python -m validator.main \
  --netuid 2 \
  --subtensor.network ws://127.0.0.1:9944 \
  --wallet.name test-validator-seth \
  --wallet.hotkey default \
  --backend-url http://localhost:8000
```

> **Note:** The validator now uses the ORO Backend API for work claiming, progress reporting, and score submission. Ensure the Backend service is running.

### Notes

- Ensure the local Subtensor chain is running before starting the validator.
- The Backend service must be running for agent submission and validator operations.
- The validator connects to the local Subtensor via WebSocket at `ws://127.0.0.1:9944`.
- Adjust wallet names as needed for your setup.
