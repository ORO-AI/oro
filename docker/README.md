# Docker Setup for ShoppingBench

This directory contains Docker configuration for running ShoppingBench services in containers.

## Services

- **Search Server**: Provides product search functionality
- **Proxy**: Nginx reverse proxy that routes requests to search server and Chutes API

## Search Server

The search server provides product search functionality via a REST API.

### Prerequisites

- Docker and Docker Compose installed
- **Pre-built indexes**: The `indexes/` directory must exist and be built before building the Docker image

### Building the Indexes (One-time Setup)

Before building the Docker image, you need to build the search indexes locally:

```bash
# Ensure documents.jsonl exists
ls resources/documents.jsonl

# Build the indexes (this may take several minutes)
./build_index.sh
```

This will create the `indexes/` directory (~3.1GB) with all product data embedded via the `--storeRaw` flag.

**Note:** The `build_index.sh` script is idempotent - it will skip building if indexes already exist. To force a rebuild, use:
```bash
./build_index.sh --force
```

### Building the Image

The Dockerfile assumes the indexes are already built and copies them into the image.

To build the image:

```bash
docker-compose build search-server
```

Or build directly:

```bash
docker build -f docker/search-server/Dockerfile -t shoppingbench-search-server .
```

**Note**: 
- The build process copies the pre-built `indexes/` directory into the image
- No `documents.jsonl` is needed at build time (all data is in the index)
- Build is fast since it skips index building (~3.1GB copy vs minutes of indexing)
- Docker automatically caches image layers - subsequent builds will be much faster if only code or indexes change
- Images built locally persist and are automatically reused by `docker-compose` (no need to rebuild if images already exist)

### Running the Service

#### Using Docker Compose (Recommended)

```bash
# Start the service
docker-compose up -d search-server

# View logs
docker-compose logs -f search-server

# Stop the service
docker-compose down
```

#### Using Docker directly

```bash
docker run -d \
  --name shoppingbench-search-server \
  -p 5632:5632 \
  -e HOST=0.0.0.0 \
  -e PORT=5632 \
  shoppingbench-search-server
```

### Environment Variables

Configure via `.env` file or environment variables:

- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `5632`)

Example `.env` file:

```bash
HOST=0.0.0.0
PORT=5632
```

### Verifying the Service

1. **Check health endpoint**:
   ```bash
   curl http://localhost:5632/health
   ```
   Should return: `{"status":"healthy","service":"search-server"}`

2. **Check API usage**:
   ```bash
   curl http://localhost:5632/
   ```
   Should return API endpoint information

3. **Test search endpoint**:
   ```bash
   curl "http://localhost:5632/find_product?q=shoes&page=1"
   ```
   Should return a JSON array of products

4. **Test product information endpoint**:
   ```bash
   curl "http://localhost:5632/view_product_information?product_ids=123456"
   ```
   Should return product details

### Service Discovery

When running with Docker Compose, the service is available at:
- **Service name**: `search-server`
- **Internal URL**: `http://search-server:5632`
- **External URL**: `http://localhost:5632` (or configured port)

Other services in the same Docker network can access it using the service name.

### Troubleshooting

1. **Index not found during build**: 
   - Ensure `indexes/` directory exists and was built using `./build_index.sh`
   - The indexes must be built locally before building the Docker image
   - Check that `indexes/` is not excluded in `.dockerignore`

2. **Port already in use**: Change the `PORT` environment variable

3. **Build fails**: 
   - Verify `indexes/` directory exists: `ls -la indexes/`
   - Ensure indexes were built: `du -sh indexes/` should show ~3.1GB
   - If indexes don't exist, run `./build_index.sh` first (requires `resources/documents.jsonl`)

4. **Health check fails**: Wait for the index to load (may take 30+ seconds on first start)

5. **Index directory empty in container**: 
   - Verify indexes exist locally before build
   - Check `.dockerignore` doesn't exclude `indexes/`

### Logs

View logs:
```bash
# Docker Compose
docker-compose logs -f search-server

# Docker
docker logs -f shoppingbench-search-server
```

The server logs to stderr, which Docker captures automatically.

## Image Caching and Reuse

Docker automatically caches image layers, which significantly speeds up rebuilds:

### How Docker Layer Caching Works

- **Layer caching**: Docker caches each layer (instruction) in the Dockerfile
- **Cache invalidation**: If a layer changes, all subsequent layers are rebuilt
- **Automatic reuse**: `docker-compose` automatically uses cached images if they exist

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

# Images are automatically reused by docker-compose
docker-compose up -d  # Uses cached images if available
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
  - `/search/*` → search-server
  - `/inference/*` → Chutes API (external, auth forwarded from sandbox client)

### Network Topology

```
Internet
  ↓
Bridge Network (default, has internet)
  ├─→ search-server (accessible from host via port mapping)
  └─→ proxy (connected to bridge for internet + search-server access)
        ↓
        └─→ sandbox network (internal, no internet)
              ├─→ proxy (accessible from agent containers)
              └─→ agent containers (isolated, can only reach proxy)
```

### Building and Running

```bash
# Build proxy image
docker-compose build proxy

# Start proxy (requires search-server to be running)
docker-compose up -d proxy

# View logs
docker-compose logs -f proxy
```

### Environment Variables

Configure via `.env` file:

```bash
# Proxy configuration
PROXY_PORT=8080  # Port exposed on host
SEARCH_SERVER_URL=search-server  # Service name for search-server
SEARCH_SERVER_PORT=5632  # Port of search-server
```

### Agent Container Configuration

Agent containers should be configured with:

```bash
SANDBOX_PROXY_URL=http://proxy:80
```

This allows agent containers to access services through the proxy:
- `http://proxy:80/search/find_product` → search-server

### Network Isolation

- **Agent containers** are placed on the `sandbox` network (internal, no internet access)
- **Proxy** is on both `bridge` (for internet + search-server) and `sandbox` (for agent containers)
- **Search-server** is on default `bridge` network (accessible from host and proxy)

### Verifying the Proxy

```bash
# Health check
curl http://localhost:8080/health

# Test search routing through proxy
curl "http://localhost:8080/search/find_product?q=shoes&page=1"
```
