# Integration Tests

Integration tests verify that ShoppingBench services work correctly when running in Docker containers.

## Prerequisites

- Python virtual environment (`.venv`) - create with `uv venv --python 3.10 && source .venv/bin/activate && uv pip install -r requirements.txt`
- Docker and Docker Compose installed
## Quick Start

1. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Build indexes (if not already built):**
   ```bash
   ./build_index.sh
   ```
   
   **Note:** `build_index.sh` automatically skips building if indexes already exist.

3. **Start required services:**

   The integration tests rely on `docker-compose.test.yml` for the
   `session-runtime` service required by the sandbox isolation tests:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.test.yml up -d search-server proxy session-runtime sandbox
   ```

   **Note:** Docker automatically reuses cached images if they exist, making this step fast.

4. **Run tests:**
   ```bash
   pytest tests/integration/ -v
   ```

## Troubleshooting

**Tests skip with "Container not running":**
- Start all services: `docker compose -f docker-compose.yml -f docker-compose.test.yml up -d search-server proxy session-runtime sandbox`
- Check status: `docker compose -f docker-compose.yml -f docker-compose.test.yml ps`

**Tests fail with connection errors:**
- Verify services are healthy: `docker compose -f docker-compose.yml -f docker-compose.test.yml ps`
- Check logs: `docker compose logs <service-name>`
- Ensure ports are not in use

**Docker Compose errors (ContainerConfig):**
- Clean up stale containers: `docker compose rm -f sandbox`
- Restart: `docker compose up -d sandbox`

## Sealed environment pack loader

With the local Backend and LocalStack running, the test compiles a deterministic
pack fixture, creates the pack bucket, uploads the archive, registers it through
the signed admin endpoint, and loads it through the signed validator endpoint:

```bash
ORO_SUBNET_NETUID=<local-backend-netuid> \
  pytest tests/integration/test_env_pack_loader.py -v
```

The Backend smoke setup must have provisioned its deterministic `//AdminLocal`
and `//ValidatorLocal` identities once. The pack itself is fully provisioned by
this test and registration is idempotent. Override endpoints or identities with
`BACKEND_URL`, `AWS_ENDPOINT_URL`, `REDIS_URL`, `ORO_SUBNET_NETUID`,
`ORO_ADMIN_KEY_URI`, and `ORO_VALIDATOR_KEY_URI`. The test refreshes the local
validator permit cache itself.

## Caching and Performance Optimization

### Index Caching

The integration tests workflow uses GitHub Actions cache to store built indexes:
- Indexes are cached based on the hash of `resources/documents.jsonl`
- If the documents file hasn't changed, indexes are restored from cache (saves significant time)
- Cache is automatically invalidated when `documents.jsonl` changes

### Docker Image Caching

Docker automatically caches image layers:
- **First build**: Takes ~15 minutes (builds all layers)
- **Subsequent builds**: Much faster (only rebuilds changed layers)
- **Image reuse**: Images built in previous Act runs persist and are automatically reused

## Testing GitHub Actions Workflow Locally

You can test the GitHub Actions workflow locally using [Act](https://github.com/nektos/act):

1. **Install Act:**
   ```bash
   # macOS
   brew install act
   
   # Linux
   curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
   ```

2. **Run workflow:**
   ```bash
   # Create secrets file
   cat > .secrets << EOF
   CHUTES_API_KEY=your-api-key
   EOF
   
   # Run workflow
   # Note: Act mounts the workspace, but indexes/ is gitignored, so it won't be visible
   # The workflow will build indexes on first run, then cache them
   
   # If port is in use, set PORT to a different value (e.g., 5631)
   act push --secret-file .secrets --container-options "-v $(pwd)/indexes:$(pwd)/indexes"
   
   # Or without mounting indexes (will build them):
   act push --secret-file .secrets
   ```
   
   **Note:** Since `indexes/` is in `.gitignore`, Act won't see it in the default workspace mount. The command above explicitly mounts the indexes directory so Act can reuse your local indexes. If indexes don't exist locally, Act will build them (first run) or use cache (subsequent runs).

**Note:** Act has limitations with Docker Compose. For full testing, run the services manually and use pytest directly.

### Act Image and Index Reuse

When running with Act:
- **Indexes**: If `indexes/` directory exists locally, Act will use it directly (no rebuild needed)
- **Docker images**: Images built during Act runs persist in your local Docker daemon
- **First Act run**: Builds indexes and images (takes time)
- **Subsequent Act runs**: Reuses local indexes and cached images (much faster)
- **To force rebuild**: Clear Docker cache with `docker system prune -a` (use with caution)

**Tip**: Build indexes locally once with `./build_index.sh`, then Act will reuse them automatically.
