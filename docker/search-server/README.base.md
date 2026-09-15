# search-server-base image

The `ghcr.io/oro-ai/oro/search-server-base:latest` image bakes the
multi-GB Lucene index, its `lucene_manifest.json`, and the
`pyserini`-patched Python runtime. The code-layer build
(`docker/search-server/Dockerfile`, wired into
`.github/workflows/publish-images.yml`) does `FROM
search-server-base:latest`, so it inherits the index without ever
touching the ~14GB payload. This split keeps every tag-push publish
fast while the base itself is refreshed only when the index changes.

## When to rebuild the base

- A new pack was compiled and the index changed
  (`documents_sha256` / `index_sha256` differ from what
  `ghcr.io/oro-ai/oro/search-server:stable` currently reports on
  `/health`).
- The base drifted (missing manifest, missing index files, arch not
  present in the multi-arch manifest). Code-layer publishes will
  fail the smoke-test identity guardrail with
  `documents_sha256 = None` / `index_sha256 = None` — that is the
  tell.

Code-only fixes (entrypoint changes, `server.py` edits, new gunicorn
flags) DO NOT require a base rebuild. Publish a normal `v*` tag and
let `publish-images.yml` layer the code on top of the current base.

## Inputs

`Dockerfile.base` expects the build context to contain:

- `docker/search-server/requirements.txt`
- `docker/search-server/patch_pyserini.py`
- `docker/search-server/Dockerfile.base` (this Dockerfile)
- `index.tar.gz` — root has an `index/` directory holding the
  Lucene shard files. Optionally bundles `lucene_manifest.json` at
  the tarball root; when it does, `ADD` will overwrite the COPY'd
  version below.
- `lucene_manifest.json` — the sealed identity manifest. `files`
  must map to every regular file inside `index/` with the expected
  `size_bytes` + `sha256`, and `java_version` must match the JDK
  the image ships with.

The `RUN` steps inside `Dockerfile.base` validate every index file's
size + SHA against `files`, and verify `java_version` matches the
running Temurin — a base that ships mismatched data cannot be
built.

## GitHub-hosted runner limitation

`.github/workflows/publish-search-server-base.yml` exists as a
`workflow_dispatch` scaffold, but `ubuntu-latest` runners do not
have enough disk to hold `~14GB index + build cache + push staging`
even after `Free disk space` cleanup. Until a self-hosted runner is
wired up, the base image must be built + pushed from an operator's
machine.

## Local build

```bash
# 1. Extract index + manifest from the currently-promoted :stable image
#    (which was baked through the private-fork pipeline).
mkdir -p /tmp/base-inputs
CID=$(docker create ghcr.io/oro-ai/oro/search-server:stable)
# :stable has /app/indexes as a symlink → /app/index; copy the real dir.
docker cp "$CID:/app/index" /tmp/base-inputs/index
docker cp "$CID:/app/lucene_manifest.json" /tmp/base-inputs/
docker rm "$CID"

# 2. Repackage as index.tar.gz with `index/` at the tarball root — that is
#    what Dockerfile.base's `ADD index.tar.gz /app/` expects.
( cd /tmp/base-inputs && tar czf /tmp/base-inputs/index.tar.gz index )

# 3. Assemble the build context (all paths relative to the checkout root
#    that the Dockerfile references directly).
BUILD_CTX=$(mktemp -d)
cp -r docker "$BUILD_CTX/"
cp /tmp/base-inputs/index.tar.gz "$BUILD_CTX/"
cp /tmp/base-inputs/lucene_manifest.json "$BUILD_CTX/"

# 4. Authenticate to GHCR with a PAT that has `write:packages`.
echo "$GHCR_PAT" | docker login ghcr.io -u <your-github-username> --password-stdin

# 5. Set up a multi-arch builder if not already.
docker buildx create --name multi --driver docker-container --use \
  || docker buildx use multi

# 6. Build + push multi-arch. This IS the publish — multi-arch push cannot
#    go through a local single-arch cache. amd64 leg goes through QEMU on
#    Apple Silicon (and vice versa); expect ~30-60 min per arch on 14GB.
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --push \
  -t ghcr.io/oro-ai/oro/search-server-base:latest \
  -f "$BUILD_CTX/docker/search-server/Dockerfile.base" \
  "$BUILD_CTX"
```

## Verify

```bash
# Manifest should list both amd64 + arm64.
docker buildx imagetools inspect ghcr.io/oro-ai/oro/search-server-base:latest

# Boot briefly + confirm /app/lucene_manifest.json is present and carries
# both identity SHAs.
docker run --rm --entrypoint sh \
  ghcr.io/oro-ai/oro/search-server-base:latest \
  -c 'python3 -c "import json; d=json.load(open(\"/app/lucene_manifest.json\")); print(d[\"documents_sha256\"], d[\"index_sha256\"])"'
```

Then re-cut a `v*` tag on `oro`; the publish workflow's identity
guardrail (`publish-images.yml`) should now pass and search-server
will land in `:latest`.
