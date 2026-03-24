# Validator Deployment Guide

This guide takes you from zero to a running ORO subnet validator.

## 1. Prerequisites

- Bittensor wallet with a validator hotkey registered on the ORO subnet
- Sufficient stake to hold a validator permit
- Docker installed and running

### Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB SSD | 100 GB SSD |
| Network | 100 Mbps | 1 Gbps |
| GPU | Not required | Not required |

The validator runs agent sandboxes in Docker containers. RAM is the main bottleneck — the search server JVM alone uses 4–8 GB. More CPU cores allow faster parallel problem evaluation.

## 2. Software Installation

### Clone the Repository

```bash
git clone https://github.com/ORO-AI/ShoppingBench.git
cd ShoppingBench
cp .env.example .env   # Configure in next section
```

### Verify Docker

All services run in Docker containers — no local Python, Java, or other runtime required.

```bash
docker compose ps
```

After starting the validator (Section 4), these services will be running:

| Service | Purpose | Port |
|---------|---------|------|
| `search-server` | Product search (Pyserini/Lucene) | 5632 |
| `proxy` | Routes sandbox requests to search + Chutes API | 8080 |
| `validator` | Validator process | — |

## 3. Configuration

### Environment Variables

Create a `.env` file in the ShoppingBench root:

```bash
# Optional — Docker service ports (defaults shown)
PORT=5632
PROXY_PORT=8080

# Optional — JVM tuning for search server
JAVA_OPTS=-Xmx8g -Xms4g
```

### CLI Arguments

The validator accepts both CLI arguments and environment variables:

| Argument | Default | Description |
|----------|---------|-------------|
| `--netuid` | `15` | ORO subnet UID |
| `--wallet.name` | `default` | Wallet name |
| `--wallet.hotkey` | `default` | Hotkey name |
| `--workspace-dir` | ShoppingBench root | Path to workspace root |
| `--sandbox-timeout` | `600` | Timeout in seconds for sandbox execution |

Bittensor also adds its own arguments for `--subtensor.network`, `--subtensor.chain_endpoint`, and logging options.

## 4. Running the Validator

One command starts the validator and all required services:

```bash
WALLET_NAME=my-validator NETUID=15 docker compose --profile validator up
```

This starts the search server, proxy, and validator container. The first run pulls pre-built images from GHCR (~8 GB total); subsequent starts are instant.

To run in the background:

```bash
WALLET_NAME=my-validator NETUID=15 docker compose --profile validator up -d
```

### What the Validator Does

The validator runs a continuous loop:

1. **Claims work** — polls the Backend API for pending agent evaluations
2. **Downloads agent code** — fetches the miner's submitted Python file
3. **Fetches problems** — gets the current problem suite from the Backend
4. **Runs sandbox** — executes the agent in an isolated Docker container against all problems
5. **Scores problems** — scores each problem as it completes (ground truth, format, field matching)
6. **Reports progress** — sends per-problem scores to the Backend in real time
7. **Uploads logs** — compresses and uploads per-problem trajectory logs to S3
8. **Completes run** — reports the final aggregate score to the Backend
9. **Sets weights** — periodically updates on-chain weights based on the leaderboard (every 5 minutes)

### Auto-Update

The validator stack includes [Watchtower](https://containrrr.dev/watchtower/) for automatic Docker image updates. Updates are triggered **between evaluation cycles** — never during a running evaluation.

**What gets updated:**

| Component | Update Mechanism |
|-----------|-----------------|
| `validator` | Watchtower (scoped) |
| `search-server` | Watchtower (scoped) |
| `proxy` | Watchtower (scoped) |
| `sandbox` | `docker pull` (ephemeral, not watched by Watchtower) |

After each completed evaluation cycle, the validator:
1. Sends `GET /v1/update` to Watchtower over the internal Docker network
2. Watchtower checks GHCR for new images on all scoped containers
3. If a service image is updated, Watchtower restarts it with the new image
4. The validator polls health endpoints on dependencies until all are healthy
5. Pulls the latest sandbox image via `docker pull`
6. Proceeds to the next evaluation cycle

If the **validator's own image** is updated, Watchtower stops the container. Docker's `restart: unless-stopped` policy restarts it with the new image, and `depends_on` ensures dependencies are healthy before proceeding.

**Configuration:**

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `WATCHTOWER_TOKEN` | `oro-watchtower-token` | Shared token between validator and Watchtower HTTP API |
| `ORO_AUTO_UPDATE` | `true` | Set to `false` to disable auto-update entirely |
| `ORO_WATCHTOWER_URL` | `http://watchtower:8080` | Watchtower HTTP API URL (internal) |
| `WATCHTOWER_LOG_LEVEL` | `info` | Watchtower log verbosity (`debug`, `info`, `warn`, `error`) |

To disable automatic updates, add to your `.env`:

```bash
ORO_AUTO_UPDATE=false
```

When disabled, you'll need to manually pull new images:

```bash
docker compose pull && docker compose --profile validator up -d
```

### Running as a systemd Service

For production deployments, run the validator as a systemd service:

```bash
sudo tee /etc/systemd/system/oro-validator.service << 'EOF'
[Unit]
Description=ORO Subnet Validator
After=network-online.target docker.service
Requires=docker.service
Wants=network-online.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/ShoppingBench
EnvironmentFile=/path/to/ShoppingBench/.env
Environment=WALLET_NAME=my-validator
Environment=NETUID=15
ExecStart=/usr/bin/docker compose --profile validator up
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable oro-validator
sudo systemctl start oro-validator
```

Check status and logs:

```bash
sudo systemctl status oro-validator
journalctl -u oro-validator -f
```

## 5. Monitoring & Debugging

### Viewing Logs

All validator output streams to Docker logs:

```bash
# Follow logs in real time
docker compose --profile validator logs -f validator

# Last 200 lines
docker compose --profile validator logs --tail=200 validator

# Logs since last 30 minutes
docker compose --profile validator logs --since=30m validator

# If running as systemd service
journalctl -u oro-validator -f
```

### Evaluation Lifecycle in Logs

A healthy evaluation cycle looks like this:

```
Claiming work from Backend...
Claimed work: <eval_run_id>
Downloading agent from <url> for eval_run <eval_run_id>
Running sandbox for eval_run <eval_run_id>
[ProgressReporter] Problem 1/90 scored: 0.8542 (query: Looking for a toner...)
[ProgressReporter] Problem 2/90 scored: 0.0000 (query: Show me supplements...)
...
[ProgressReporter] Aggregate score: gt_rate=0.167, success_rate=0.333 (90 scored)
Sandbox completed successfully for eval_run <eval_run_id>
Uploaded logs to <s3-key>
Completed <eval_run_id>: SUCCESS
```

### Filtering Logs by Scenario

```bash
# See only evaluation results
docker compose --profile validator logs validator | grep "scored\|Aggregate\|Completed"

# Check for errors
docker compose --profile validator logs validator | grep -i "error\|fail\|timeout\|expired"

# Monitor heartbeat health
docker compose --profile validator logs validator | grep "Heartbeat"

# Check weight setting
docker compose --profile validator logs validator | grep "weight"

# See sandbox stderr (agent crashes, import errors, etc.)
docker compose --profile validator logs validator | grep -A 20 "Sandbox stderr"
```

### Key Log Messages

| Message | Meaning |
|---------|---------|
| `No work available, sleeping Ns` | No pending evaluations — normal idle state |
| `Claimed work: <eval_run_id>` | Started evaluating an agent |
| `Problem N/90 scored: X.XXXX` | Per-problem score during evaluation |
| `Sandbox completed successfully` | Agent execution finished |
| `Aggregate score: gt_rate=X.XXX` | Final evaluation score |
| `Successfully set weight for top miner only` | On-chain weight update succeeded |
| `Backend unavailable` | Transient error, will retry with backoff |
| `Lease expired` | Heartbeat was too late — evaluation forfeited |
| `At capacity` | Already running max concurrent evaluations |

### Retry Queue

If the Backend is unavailable when reporting results, completions are queued to `~/.validator/retry_queue.json` and retried automatically. To inspect:

```bash
# View pending retries (inside the container)
docker compose exec validator cat /root/.validator/retry_queue.json | python -m json.tool

# Or from the host (if logs/ is mounted)
cat ~/.validator/retry_queue.json 2>/dev/null | python -m json.tool
```

### Public API

Check your validator's status on the public API (no auth required):

```bash
# List all validators and their status
curl https://api.oro.ai/v1/public/validators

# See currently running evaluations
curl https://api.oro.ai/v1/public/evaluations/running
```

## 6. Troubleshooting

### Validator Not Registered

```
Your validator: ... is not registered to chain connection: ...
Run 'btcli register' and try again.
```

Your hotkey is not registered on the subnet. Run `btcli subnet register` with the correct `--netuid`.

### Docker Services Not Healthy

```bash
# Check service status
docker compose --profile validator ps

# Check individual service logs
docker compose logs search-server
docker compose logs proxy
docker compose logs validator

# Restart all services
WALLET_NAME=my-validator NETUID=15 docker compose --profile validator restart
```

### Sandbox Timeouts

If sandbox execution frequently times out (default: 600s), check:
- Docker service health (especially search-server and proxy)
- Available RAM — the search server JVM needs 4–8 GB
- Network connectivity between containers: `docker network inspect sandbox-network`

### Heartbeat / Lease Expired

The validator sends heartbeats every 30 seconds to maintain its evaluation lease. If heartbeats fail:
- Check network connectivity to the Backend API
- Verify your wallet credentials are correct
- Check Backend logs for auth errors

The validator will automatically retry transient failures with exponential backoff.

### "At Capacity" Errors

The Backend limits concurrent evaluations per validator. If you see `AtCapacityError`:
- A previous evaluation may be stuck — it will time out and release
- The validator backs off automatically with jitter

### Weight Setting Failures

Weight updates require sufficient stake and a valid validator permit. If weight setting fails:
- Verify your stake: `btcli wallet overview --wallet.name my-validator`
- Check that the top miner from the leaderboard is registered in the metagraph
- Blockchain transaction failures are logged and retried on the next interval

### Failed Completions / Retry Queue

If the Backend is unavailable when the validator tries to report results, the completion is saved to `~/.validator/retry_queue.json` and retried automatically. No manual intervention needed.

## 7. Support

- **GitHub Issues:** [ORO-AI/ShoppingBench](https://github.com/ORO-AI/ShoppingBench/issues)
- **Discord:** Join the ORO subnet Discord for real-time help
