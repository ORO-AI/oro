"""Deterministic BM25 contract exposed by the search-server."""

import hashlib
import json
from pathlib import Path
from typing import Any

MAX_K = 50
BACKEND_ID = "pyserini_lucene_bm25_v1"
QUERY_CONFIG = {
    "analyzer": "io.anserini.analysis.DefaultEnglishAnalyzer",
    "bm25": {"b": 0.4, "k1": 0.9},
    "fields": ["contents"],
    "max_k": MAX_K,
    "query_generator": "io.anserini.search.query.BagOfWordsQueryGenerator",
    "tie_breaker": "io.anserini.rerank.lib.ScoreTiesAdjusterReranker",
}

_LUCENE_SPECIAL_CHARS = str.maketrans(
    {character: " " for character in '+-&|!(){}[]^"~*?:\\/'}
)


class SearchContractError(ValueError):
    """A query or baked index manifest violates the search contract."""


def normalize_query(query: str | None) -> str:
    """Treat input as plain text and normalize whitespace."""

    if not query:
        return ""
    return " ".join(query.translate(_LUCENE_SPECIAL_CHARS).split())


def parse_k(value: str | int | None) -> int:
    if value is None or value == "":
        return 10
    try:
        requested = int(value)
    except (TypeError, ValueError) as exc:
        raise SearchContractError("k must be an integer") from exc
    return max(1, min(requested, MAX_K))


def identity_from_manifest(path: Path) -> dict[str, Any]:
    """Read and validate the public identity baked into the image."""

    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SearchContractError("Lucene manifest is not valid JSON") from exc

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise SearchContractError("Lucene manifest files must be a non-empty object")

    total_size = 0
    for name in sorted(files):
        receipt = files[name]
        digest = receipt.get("sha256") if isinstance(receipt, dict) else None
        size = receipt.get("size_bytes") if isinstance(receipt, dict) else None
        if (
            Path(name).name != name
            or not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(size, int)
            or size < 0
        ):
            raise SearchContractError(
                "Lucene manifest contains an invalid file receipt"
            )
        total_size += size

    index_sha256 = manifest.get("index_sha256")
    if (
        not isinstance(index_sha256, str)
        or len(index_sha256) != 64
        or any(character not in "0123456789abcdef" for character in index_sha256)
    ):
        raise SearchContractError("Lucene manifest is missing a valid index identity")

    metadata = {
        name: manifest.get(name)
        for name in ("documents_sha256", "java_version", "pyserini_version")
    }
    if not all(isinstance(value, str) and value for value in metadata.values()):
        raise SearchContractError("Lucene manifest is missing compatibility metadata")

    identity = {
        "backend_id": BACKEND_ID,
        **metadata,
        "index_file_count": len(files),
        "index_sha256": index_sha256,
        "index_size_bytes": total_size,
        "lucene_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "query": QUERY_CONFIG,
    }
    if manifest.get("index_contract"):
        identity["index_contract"] = manifest["index_contract"]
    return identity


def bm25(
    searcher: Any, query: str | None, k: str | int | None = None
) -> list[dict[str, str]]:
    """Return only canonical identities in Lucene rank order."""

    normalized = normalize_query(query)
    if not normalized:
        return []

    results = []
    for hit in searcher.search(q=normalized, k=parse_k(k), remove_dups=False):
        document = searcher.doc(hit.docid)
        try:
            product = json.loads(document.raw())["product"]
            raw_product_id = product["product_id"]
            raw_sku = product["sku"]
            if raw_product_id is None or raw_sku is None:
                raise KeyError("missing canonical identity")
            product_id = str(raw_product_id)
            sku = str(raw_sku)
        except (AttributeError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SearchContractError(
                "Lucene hit does not contain canonical product_id and sku"
            ) from exc
        if not product_id or not sku:
            raise SearchContractError(
                "Lucene hit does not contain canonical product_id and sku"
            )
        results.append({"product_id": product_id, "sku": sku})
    return results
