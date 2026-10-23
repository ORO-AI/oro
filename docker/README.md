# Docker Setup for ShoppingBench

This directory contains Docker configuration for running ShoppingBench services in containers.

## Services

- **Search Server**: Serves each world pack's own search indexes
- **Proxy**: Nginx reverse proxy that routes agent requests to the session runtime and the inference providers

## Search Server

The search server holds no catalog of its own. A world pack ships its own Lucene indexes; the validator
(and the local test) installs each one at runtime and searches it with that index's BM25 settings. The
image is `docker/search-server/Dockerfile`: Python, the JDK and Pyserini, with no baked index.

| Route | Purpose |
| --- | --- |
| `GET /health` | `{"status": "healthy", "service": "search-worlds"}` |
| `PUT /internal/worlds/<sha>` | Install a world index archive (the request body); its BM25 settings travel in `X-ORO-World-Settings`. `201` when new, `200` when already installed, `409` on conflicting settings. |
| `POST /internal/worlds/<sha>/search` | `{"query": ..., "params": {...}}` → `{"hits": [{"id": ..., "score": ...}]}`, ranked; `404` for a world not installed. |

These routes are internal: agents reach search only through their world's tools, which the validator's
session runtime serves.

```bash
docker compose build search-server     # or: docker build -f docker/search-server/Dockerfile .
docker compose up -d search-server
curl http://localhost:5632/health
```

Settings (environment):
- `PORT` (default `5632`).
- `SEARCH_WORKERS`: gunicorn sync workers, one JVM each (default 2); keep it at or below free cores.
- `_JAVA_OPTIONS`: per-worker heap (default `-Xmx1g -Xms256m`).
- `WORLDS_DIR`: where installed worlds live, shared by the workers (default `/tmp/worlds`).
- `WORLDS_OPEN` / `WORLDS_KEEP`: searchers kept open and worlds kept on disk (defaults 8 and 32).

Inside the compose network the service is `http://search-server:5632`; the validator reads
`SEARCH_SERVER_URL`. Logs go to stderr (`docker compose logs -f search-server`).

## CA bundle security updates

The Python requirements and images require `certifi>=2026.10.19`; pip-based
images also bootstrap with `pip==26.2.2`, and the validator image uses uv 0.13.4.
After updating this repository, rebuild the source images, then recreate the
long-running services. The local `test` image uses the validator Dockerfile and
is one-shot; rebuilding it updates the next local test run, so rerun your usual
local test command after the build.

```bash
docker compose build --pull validator sandbox search-server test
docker compose --profile validator up -d --force-recreate validator search-server
```

For an existing host Python environment, upgrade pip and certifi, reinstall the
project requirements, then restart Python processes using that environment:

```bash
python -m pip install --upgrade 'pip>=26.2.2' 'certifi>=2026.10.19'
python -m pip install -r requirements.txt
```

For uv-managed environments, use uv 0.13.4 or later and resync the environment.

## Image Caching and Reuse

Docker automatically caches image layers, which significantly speeds up rebuilds:

### How Docker Layer Caching Works

- **Layer caching**: Docker caches each layer (instruction) in the Dockerfile
- **Cache invalidation**: If a layer changes, all subsequent layers are rebuilt
- **Automatic reuse**: `docker compose` automatically uses cached images if they exist

### Optimizing Build Times

The Dockerfiles are optimized for caching:
1. Dependencies are installed first (changes infrequently)
2. Source code is copied next (changes more frequently)
3. Indexes are copied last (largest layer, changes least frequently)

This ensures that when only code changes, dependency layers remain cached.

### Reusing Existing Images

If you've built images before, Docker will automatically reuse them:

```bash
# Check if images exist
docker images | grep shoppingbench

# Images are automatically reused by docker compose
docker compose up -d  # Uses cached images if available
```

### Clearing Caches

If you need to clear Docker caches (e.g., to free disk space):

```bash
# Remove unused images, containers, and build cache
docker system prune -a

# Warning: This will remove all unused images, including ShoppingBench images
# You'll need to rebuild after running this
```

**Note:** Running `docker system prune` will clear cached images, requiring a full rebuild.

## Proxy Service

The proxy service acts as a gateway for sandboxed agent containers, providing controlled access to services.

### Architecture

- **Proxy Container**: Only container with internet access, runs nginx reverse proxy
- **Sandbox Network**: Internal Docker network (no internet access) for agent containers
- **Path-based Routing**:
  - `/environment/call` → session runtime (ORO Bench tool actions)
  - `/inference/*` → configured inference provider (external, auth forwarded from sandbox client)
  - `/search/*` → 410 Gone (ShoppingBench routes, removed)

### Network Topology

```
Internet
  ↓
`main` network (shoppingbench-main, has internet)
  ├─→ search-server (on the `main` network; loopback publish for operator checks)
  └─→ proxy (on `main` for internet access)
        ↓
        └─→ sandbox network (internal, no internet)
              ├─→ proxy (accessible from agent containers)
              └─→ agent containers (isolated, can only reach proxy)
```

### Building and Running

Inference access logs include `fields=` (the first 32 incoming top-level field
names, captured before proxy rewriting; `+more` marks overflow). This field contains no
request values or Authorization header. Aggregate these access lines to see
which parameters reach the proxy, including calls outside `ProxyClient`. Treat
`+more` records as incomplete and count them separately.
Inference calls require the validator-issued key for the active run; missing,
different, or expired keys receive HTTP 401 before reaching the provider.
Rejected-key requests log `fields=-` because the proxy checks the key before parsing the body.

```bash
# Build proxy image
docker compose build proxy

# Start proxy
docker compose up -d proxy

# View logs
docker compose logs -f proxy
```

### Environment Variables

Configure via `.env` file:

```bash
# Proxy configuration
PROXY_PORT=8080  # Port exposed on host
```

### Agent Container Configuration

Agent containers should be configured with:

```bash
SANDBOX_PROXY_URL=http://proxy:80
```

This allows agent containers to access services through the proxy:
- `http://proxy:80/environment/call` → session runtime
- `http://proxy:80/inference/chat/completions` → inference providers

### Network Isolation

- **Agent containers** are placed on the `sandbox` network (internal, no internet access)
- **Proxy** is on both `main` (for internet) and `sandbox` (for agent containers)
- **Search-server** is on the `main` network (reachable by the validator runtime and, via loopback publish, the host; never by the sandbox)

### Verifying the Proxy

```bash
# Health check
curl http://localhost:8080/health

# Legacy search routes are gone
curl -i "http://localhost:8080/search/find_product?q=shoes"   # HTTP 410
```
