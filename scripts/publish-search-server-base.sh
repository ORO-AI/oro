#!/bin/bash
set -euo pipefail

IMAGE="ghcr.io/oro-ai/oro/search-server-base:latest"

# Ensure we're in the repo root
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: Run this script from the ShoppingBench repo root."
    exit 1
fi

# Ensure indexes exist
if [ ! -d "indexes" ] || [ -z "$(ls -A indexes 2>/dev/null)" ]; then
    echo "No indexes found. Building..."
    ./build_index.sh
fi

echo "Building base image..."
docker build -f docker/search-server/Dockerfile.base -t "$IMAGE" .

echo "Pushing to GHCR..."
docker push "$IMAGE"

echo "Done. Base image published: $IMAGE"
