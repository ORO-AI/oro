#!/usr/bin/env bash
# Import smoke for the validator image: the runtime the image actually installs
# (from docker/validator/uv.lock, via `uv sync --frozen`) must be the 4.x runtime
# with contract 8, and the validator entry modules must import against it.
#
# Usage: scripts/validator_image_smoke.sh [IMAGE]
#   With no IMAGE, builds docker/validator/Dockerfile from the repo root first.
#
# `uv sync --frozen` does not check the lock against pyproject.toml, so an image
# built from a stale lock installs the old runtime; this check catches that.
set -euo pipefail

image="${1:-}"
if [ -z "$image" ]; then
  image="oro-validator-smoke:local"
  docker build -f "$(dirname "$0")/../docker/validator/Dockerfile" -t "$image" "$(dirname "$0")/.."
fi

# The baked .venv is the interpreter the ENTRYPOINT runs; `uv run` would
# re-resolve from pyproject.toml and hide lock drift.
docker run --rm --entrypoint .venv/bin/python "$image" -c '
import importlib.metadata as m
import oro_env_runtime
version = m.version("oro-env-runtime")
contract = getattr(oro_env_runtime, "RUNTIME_CONTRACT", None)
assert contract == 8, f"oro-env-runtime {version}: RUNTIME_CONTRACT={contract}, want 8"
import subnet.validator.main, subnet.validator.session_registry, subnet.validator.env_pack_loader
print(f"Validator image OK: oro-env-runtime {version}, contract {contract}")
'
