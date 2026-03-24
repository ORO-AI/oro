#!/bin/bash
set -e

# Check for --force flag
FORCE_REBUILD=false
if [ "$1" = "--force" ]; then
    FORCE_REBUILD=true
    echo "Force rebuild requested, will rebuild indexes even if they exist"
fi

# Check if resources/documents.jsonl exists
if [ ! -f "resources/documents.jsonl" ]; then
    echo "Error: resources/documents.jsonl not found"
    echo "Please place the documents.jsonl file in the resources directory"
    exit 1
fi

# Check if indexes already exist
if [ "$FORCE_REBUILD" = false ] && [ -d "indexes" ] && [ -n "$(ls -A indexes 2>/dev/null)" ]; then
    echo "Indexes directory already exists and is not empty. Skipping build."
    echo "To force a rebuild, run: ./build_index.sh --force"
    exit 0
fi

# Clean up existing indexes if rebuilding
if [ -d "indexes" ]; then
    echo "Removing existing indexes..."
    rm -rf indexes
fi
mkdir -p indexes

# Build indexes using Docker (no local Java required)
echo "Building search indexes using Docker..."

# Use docker compose (newer) or docker-compose (older)
if command -v docker &> /dev/null && docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
else
    echo "Error: docker-compose not found. Please install Docker."
    exit 1
fi

# Build and run the index-builder service
$DOCKER_COMPOSE_CMD build index-builder
$DOCKER_COMPOSE_CMD run --rm index-builder

if [ $? -eq 0 ]; then
    echo "Build indexes success"
    echo "Index files:"
    ls -la indexes/
else
    echo "Error: Failed to build indexes"
    exit 1
fi
