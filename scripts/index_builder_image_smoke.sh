#!/usr/bin/env bash
# Build a one-document Lucene index with the actual index-builder image and
# verify the search image can read it with the catalog fields intact.
set -euo pipefail

image="${1:-oro-index-builder-smoke:local}"
if [ "$#" -eq 0 ]; then
  docker build -f docker/index-builder/Dockerfile -t "$image" .
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/resources"
cat > "$tmp/resources/documents.jsonl" <<'JSON'
{"id":"p1","contents":"Acme blue widget","catalog_brand":"Acme","catalog_candidate":"true","catalog_category":"tools","catalog_price":"3.5","catalog_source":"test","catalog_source_variant":"false"}
JSON

docker run --rm --user "$(id -u):$(id -g)" -v "$tmp:/data" "$image"
docker run --rm -v "$tmp:/data" --entrypoint python oro-search-server-smoke:local -c '
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher("/data/indexes")
hits = searcher.search("Acme")
assert hits and hits[0].docid == "p1", hits
raw = searcher.doc("p1").raw()
for field in ("catalog_candidate", "catalog_source_variant", "catalog_price"):
    assert f"\"{field}\"" in raw, raw
print("Index-builder smoke passed: Lucene index is searchable and catalog fields are stored")
'
