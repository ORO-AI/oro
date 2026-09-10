import json

import pytest

from src.search_engine.search_contract import (
    SearchContractError,
    bm25,
    identity_from_manifest,
    normalize_query,
    parse_k,
)


class _Hit:
    def __init__(self, docid):
        self.docid = docid


class _Document:
    def __init__(self, product):
        self._product = product

    def raw(self):
        return json.dumps({"product": self._product})


class _Searcher:
    def __init__(self, products):
        self.products = products
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return [_Hit(index) for index in range(min(kwargs["k"], len(self.products)))]

    def doc(self, docid):
        return _Document(self.products[docid])


def test_normalize_query_treats_lucene_syntax_as_plain_text():
    assert normalize_query('  travel +(mug):"steel"  ') == "travel mug steel"


@pytest.mark.parametrize(
    "value,expected",
    [(None, 10), ("", 10), ("0", 1), ("4", 4), (100, 50)],
)
def test_parse_k_clamps_to_contract(value, expected):
    assert parse_k(value) == expected


def test_parse_k_rejects_non_integer():
    with pytest.raises(SearchContractError, match="k must be an integer"):
        parse_k("many")


def test_identity_comes_from_baked_manifest(tmp_path):
    manifest = tmp_path / "lucene_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "documents_sha256": "d" * 64,
                "index_sha256": "e" * 64,
                "java_version": "21",
                "pyserini_version": "1.0.0",
                "files": {
                    "segments_1": {"sha256": "a" * 64, "size_bytes": 12},
                    "_0.cfs": {"sha256": "b" * 64, "size_bytes": 34},
                },
            }
        )
    )

    identity = identity_from_manifest(manifest)

    assert identity["documents_sha256"] == "d" * 64
    assert identity["index_sha256"] == "e" * 64
    assert identity["index_file_count"] == 2
    assert identity["index_size_bytes"] == 46
    assert identity["pyserini_version"] == "1.0.0"


@pytest.mark.parametrize("value", [None, "short", "A" * 64, "z" * 64])
def test_identity_rejects_invalid_declared_index_identity(tmp_path, value):
    manifest = tmp_path / "lucene_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "documents_sha256": "d" * 64,
                "index_sha256": value,
                "java_version": "21",
                "pyserini_version": "1.0.0",
                "files": {
                    "segments_1": {"sha256": "a" * 64, "size_bytes": 12}
                },
            }
        )
    )

    with pytest.raises(SearchContractError, match="valid index identity"):
        identity_from_manifest(manifest)


def test_bm25_returns_only_ordered_canonical_identities():
    searcher = _Searcher(
        [
            {"product_id": 12, "sku": "blue", "hidden": "not returned"},
            {"product_id": "13", "sku": 7},
        ]
    )

    assert bm25(searcher, "travel+mug", 2) == [
        {"product_id": "12", "sku": "blue"},
        {"product_id": "13", "sku": "7"},
    ]
    assert searcher.calls == [{"q": "travel mug", "k": 2, "remove_dups": False}]


def test_bm25_rejects_documents_without_exact_sku_identity():
    searcher = _Searcher([{"product_id": "12"}])
    with pytest.raises(SearchContractError, match="product_id and sku"):
        bm25(searcher, "travel mug")


def test_bm25_empty_query_does_not_touch_lucene():
    searcher = _Searcher([])
    assert bm25(searcher, "   ") == []
    assert searcher.calls == []
