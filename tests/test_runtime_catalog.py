from types import SimpleNamespace

import pytest

from src.search_engine import runtime_catalog
from src.search_engine.runtime_catalog import (
    CatalogContractError,
    catalog_record,
    filter_candidates,
    products,
    search_candidates,
)


def _document(product_id="1", sku="sku-1", **overrides):
    product = {
        "admitted": True,
        "attributes": {"Color": "Blue"},
        "brand": "Acme",
        "category_path": ["Audio", "Headphones"],
        "currency": "PHP",
        "description": "Description",
        "in_stock": True,
        "main_image_url": "https://example.com/image.jpg",
        "options": "Blue",
        "price": 100.0,
        "product_id": product_id,
        "product_url": "https://example.com/product",
        "sku": sku,
        "source_listing_id": "a" * 64 + "0",
        "title": "Acme Headphones",
    }
    product.update(overrides)
    return {
        "catalog_candidate": "true",
        "catalog_source_variant": "true",
        "product": product,
    }


class _Stored:
    def __init__(self, value):
        import json

        self._raw = json.dumps(value)

    def raw(self):
        return self._raw


class _Searcher:
    def __init__(self, documents, hits=()):
        self.documents = documents
        self.hits = list(hits)

    def doc(self, product_id):
        value = self.documents.get(product_id)
        return _Stored(value) if value is not None else None

    def search(self, **_kwargs):
        return self.hits


def _query_stubs(monkeypatch):
    monkeypatch.setattr(
        runtime_catalog, "_term_query", lambda field, value: ("term", field, value)
    )
    monkeypatch.setattr(
        runtime_catalog, "_and_query", lambda queries: ("and", tuple(queries))
    )
    monkeypatch.setattr(
        runtime_catalog, "_or_query", lambda queries: ("or", tuple(queries))
    )
    monkeypatch.setattr(runtime_catalog, "_constant_query", lambda query: query)


def test_catalog_record_projects_the_runtime_shape():
    record = catalog_record(_document())

    assert record["product_id"] == "1"
    assert record["sku"] == "sku-1"
    assert record["in_stock"] is True


def test_catalog_record_rejects_a_missing_required_field():
    document = _document()
    document["product"].pop("category_path")

    with pytest.raises(CatalogContractError, match="category_path"):
        catalog_record(document)


def test_products_preserves_requested_order_and_skips_unknown_ids():
    searcher = _Searcher({"1": _document(), "2": _document("2", "sku-2")})

    assert [row["product_id"] for row in products(searcher, ["2", "missing", "1"])] == [
        "2",
        "1",
    ]


def test_search_candidates_deduplicates_exact_candidates():
    searcher = _Searcher({"1": _document()})

    hits = [
        {"product_id": "1", "sku": "sku-1"},
        {"product_id": "1", "sku": "sku-1"},
    ]
    assert [
        (row["product_id"], row["sku"]) for row in search_candidates(searcher, hits)
    ] == [("1", "sku-1")]


def test_search_candidates_expands_a_source_variant(monkeypatch):
    unavailable = _document(in_stock=False)
    unavailable["catalog_candidate"] = "false"
    replacement = catalog_record(_document("2", "sku-2"))
    searcher = _Searcher({"1": unavailable}, [SimpleNamespace(docid="1")])
    monkeypatch.setattr(
        runtime_catalog,
        "_source_candidates",
        lambda _searcher, sources: {sources[0]: [replacement]},
    )

    assert search_candidates(searcher, [{"product_id": "1", "sku": "sku-1"}]) == [
        replacement
    ]


def test_search_candidates_rejects_identity_drift():
    searcher = _Searcher({"1": _document()})

    with pytest.raises(CatalogContractError, match="does not match"):
        search_candidates(searcher, [{"product_id": "1", "sku": "other"}])


def test_search_candidates_rejects_missing_catalog_flags():
    document = _document()
    document.pop("catalog_candidate")
    searcher = _Searcher({"1": document})

    with pytest.raises(CatalogContractError, match="runtime catalog flags"):
        search_candidates(searcher, [{"product_id": "1", "sku": "sku-1"}])


def test_filter_brand_terms_are_words_of_one_brand(monkeypatch):
    _query_stubs(monkeypatch)
    first_word = "a" * 64 + "0"
    second_word = "b" * 64 + "0"

    def sorted_records(_searcher, query, *, limit):
        assert query == (
            "and",
            (
                ("term", "catalog_candidate", "true"),
                (
                    "and",
                    (
                        ("term", "catalog_brand", first_word),
                        ("term", "catalog_brand", second_word),
                    ),
                ),
            ),
        )
        assert limit == 30
        return []

    monkeypatch.setattr(runtime_catalog, "_sorted_records", sorted_records)

    assert filter_candidates(
        object(),
        category_terms=[],
        brand_terms=[first_word, second_word],
        max_price_term=None,
        limit=30,
    ) == []


def test_sorted_records_returns_the_global_product_id_prefix(monkeypatch):
    _query_stubs(monkeypatch)
    documents = {
        f"{index:03d}": _document(f"{index:03d}", f"sku-{index:03d}")
        for index in range(200)
    }

    class NativeSearcher:
        def __init__(self):
            self.internal_ids = list(reversed(documents))

        def search(self, query, limit, sort):
            assert query == "query"
            assert sort == "catalog-sort"
            ordered = sorted(range(len(self.internal_ids)), key=self.internal_ids.__getitem__)
            return SimpleNamespace(
                scoreDocs=[SimpleNamespace(doc=value) for value in ordered[:limit]]
            )

        def doc(self, internal_id):
            return SimpleNamespace(get=lambda field: self.internal_ids[internal_id])

    searcher = _Searcher(documents)
    searcher.object = SimpleNamespace(searcher=NativeSearcher())
    monkeypatch.setattr(runtime_catalog, "_catalog_sort", lambda: "catalog-sort")

    result = runtime_catalog._sorted_records(
        searcher,
        "query",
        limit=2,
    )

    assert [row["product_id"] for row in result] == ["000", "001"]


def test_sorted_records_allows_no_matches(monkeypatch):
    native = SimpleNamespace(
        search=lambda _query, _limit, _sort: SimpleNamespace(scoreDocs=[])
    )
    searcher = SimpleNamespace(object=SimpleNamespace(searcher=native))
    monkeypatch.setattr(runtime_catalog, "_catalog_sort", lambda: "catalog-sort")

    assert runtime_catalog._sorted_records(searcher, "query", limit=30) == []


def test_each_source_expansion_gets_its_own_budget(monkeypatch):
    _query_stubs(monkeypatch)
    source_a = "a" * 64 + "0"
    source_b = "b" * 64 + "0"
    documents = {}
    hits_by_source = {}
    for source, count in ((source_a, 50), (source_b, 40)):
        hits = []
        for index in range(count):
            product_id = f"{source[0]}-{index:02d}"
            documents[product_id] = _document(
                product_id,
                f"sku-{index:02d}",
                source_listing_id=source,
            )
            hits.append(SimpleNamespace(docid=product_id))
        hits_by_source[source] = hits

    class Searcher(_Searcher):
        def __init__(self):
            super().__init__(documents)
            self.queries = []

    searcher = Searcher()
    def sorted_records(_searcher, query, *, limit):
        assert limit == runtime_catalog.MAX_SOURCE_VARIANTS
        searcher.queries.append(query)
        source = query[1][1][2]
        return [catalog_record(documents[hit.docid]) for hit in hits_by_source[source]]

    monkeypatch.setattr(runtime_catalog, "_sorted_records", sorted_records)
    result = runtime_catalog._source_candidates(searcher, [source_a, source_b])

    assert len(searcher.queries) == 2
    assert len(result[source_a]) == 50
    assert len(result[source_b]) == 40
