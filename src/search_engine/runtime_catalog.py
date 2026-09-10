"""Lucene-backed immutable catalog operations for the generated environment."""

import json
import re
from typing import Any

INDEX_CONTRACT_VERSION = "oro.lucene.catalog.v2"
MAX_BATCH_PRODUCTS = 1000
MAX_FILTER_RESULTS = 150
MAX_SOURCE_VARIANTS = 50
_TERM_RE = re.compile(r"[0-9a-f]{64}0")
_PRICE_RE = re.compile(r"[0-9a-f]{16}0")


class CatalogContractError(ValueError):
    """An index document or internal catalog request is invalid."""


def _raw_document(searcher: Any, product_id: str) -> dict[str, Any] | None:
    document = searcher.doc(product_id)
    if document is None:
        return None
    try:
        value = json.loads(document.raw())
    except (AttributeError, TypeError, json.JSONDecodeError) as exc:
        raise CatalogContractError("Lucene catalog document is invalid") from exc
    if not isinstance(value, dict) or not isinstance(value.get("product"), dict):
        raise CatalogContractError("Lucene catalog document has no product")
    return value


def catalog_record(document: dict[str, Any]) -> dict[str, Any]:
    """Project one stored Lucene document onto the runtime catalog contract."""

    product = document["product"]
    required = {
        "admitted",
        "attributes",
        "brand",
        "category_path",
        "currency",
        "description",
        "in_stock",
        "main_image_url",
        "options",
        "price",
        "product_id",
        "product_url",
        "sku",
        "source_listing_id",
        "title",
    }
    missing = sorted(required - set(product))
    if missing:
        raise CatalogContractError(
            f"Lucene catalog product is missing fields: {', '.join(missing)}"
        )
    return {
        "admitted": product["admitted"] is True,
        "brand": product["brand"],
        "category_path": product["category_path"],
        "currency": product["currency"],
        "description": product["description"],
        "in_stock": product["in_stock"] is True,
        "main_image_url": product["main_image_url"],
        "options": product["options"],
        "price": product["price"],
        "product_id": str(product["product_id"]),
        "product_url": product["product_url"],
        "sku": str(product["sku"]),
        "source_listing_id": product["source_listing_id"],
        "specification": product["attributes"],
        "title": product["title"],
    }


def products(searcher: Any, product_ids: list[str]) -> list[dict[str, Any]]:
    if (
        not isinstance(product_ids, list)
        or not product_ids
        or len(product_ids) > MAX_BATCH_PRODUCTS
        or any(
            not isinstance(product_id, str) or not product_id
            for product_id in product_ids
        )
    ):
        raise CatalogContractError(
            f"product_ids must contain 1-{MAX_BATCH_PRODUCTS} values"
        )
    records = []
    for product_id in product_ids:
        document = _raw_document(searcher, str(product_id))
        if document is not None:
            records.append(catalog_record(document))
    return records


def _term_query(field: str, value: str) -> Any:
    from jnius import autoclass

    term = autoclass("org.apache.lucene.index.Term")
    term_query = autoclass("org.apache.lucene.search.TermQuery")
    return term_query(term(field, value))


def _and_query(queries: list[Any]) -> Any:
    from jnius import autoclass

    builder = autoclass("org.apache.lucene.search.BooleanQuery$Builder")()
    occur = autoclass("org.apache.lucene.search.BooleanClause$Occur")
    for query in queries:
        builder.add(query, occur.MUST)
    return builder.build()


def _or_query(queries: list[Any]) -> Any:
    from jnius import autoclass

    builder = autoclass("org.apache.lucene.search.BooleanQuery$Builder")()
    occur = autoclass("org.apache.lucene.search.BooleanClause$Occur")
    for query in queries:
        builder.add(query, occur.SHOULD)
    builder.setMinimumNumberShouldMatch(1)
    return builder.build()


def _constant_query(query: Any) -> Any:
    from jnius import autoclass

    return autoclass("org.apache.lucene.search.ConstantScoreQuery")(query)


def _price_query(maximum: str) -> Any:
    from jnius import autoclass

    bytes_ref = autoclass("org.apache.lucene.util.BytesRef")
    range_query = autoclass("org.apache.lucene.search.TermRangeQuery")
    return range_query(
        "catalog_price",
        bytes_ref(b"00000000000000000"),
        bytes_ref(maximum.encode()),
        True,
        True,
    )


def _validate_terms(values: list[str], *, name: str) -> list[str]:
    if not all(
        isinstance(value, str) and _TERM_RE.fullmatch(value) for value in values
    ):
        raise CatalogContractError(f"{name} contains an invalid term")
    return list(dict.fromkeys(values))


def _records_for_hits(searcher: Any, hits: list[Any]) -> list[dict[str, Any]]:
    records = []
    for hit in hits:
        document = _raw_document(searcher, hit.docid)
        if document is None:
            raise CatalogContractError("Lucene catalog hit cannot be loaded")
        records.append(catalog_record(document))
    return records


def _catalog_sort() -> Any:
    from jnius import autoclass

    sort = autoclass("org.apache.lucene.search.Sort")
    sort_field = autoclass("org.apache.lucene.search.SortField")
    sort_type = autoclass("org.apache.lucene.search.SortField$Type")
    # Anserini's JsonCollection indexes `id` as binary doc values. Product IDs
    # are unique per projected variant, so ordering by id is the catalog's
    # stable (product_id, sku) order without a secondary sort field.
    return sort(sort_field("id", sort_type.STRING_VAL))


def _sorted_records(
    searcher: Any, query: Any, *, limit: int
) -> list[dict[str, Any]]:
    top_docs = searcher.object.searcher.search(query, limit, _catalog_sort())
    product_ids = [
        searcher.object.searcher.doc(hit.doc).get("id")
        for hit in top_docs.scoreDocs
    ]
    if not product_ids:
        return []
    if any(not product_id for product_id in product_ids):
        raise CatalogContractError("Lucene catalog hit has no product id")
    return products(searcher, product_ids)


def filter_candidates(
    searcher: Any,
    *,
    category_terms: list[str],
    brand_terms: list[str],
    max_price_term: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    if not 1 <= limit <= MAX_FILTER_RESULTS:
        raise CatalogContractError(f"limit must be between 1 and {MAX_FILTER_RESULTS}")
    category_terms = _validate_terms(category_terms, name="category_terms")
    brand_terms = _validate_terms(brand_terms, name="brand_terms")
    queries = [_term_query("catalog_candidate", "true")]
    if category_terms:
        queries.append(
            _or_query(
                [_term_query("catalog_category", term) for term in category_terms]
            )
        )
    # These are normalized words from one requested brand (for example,
    # "Hewlett Packard"), not alternative brands. Require every word.
    if brand_terms:
        queries.append(
            _and_query([_term_query("catalog_brand", term) for term in brand_terms])
        )
    if max_price_term is not None:
        if not _PRICE_RE.fullmatch(max_price_term):
            raise CatalogContractError("max_price_term is invalid")
        queries.append(_price_query(max_price_term))
    return _sorted_records(
        searcher,
        _constant_query(_and_query(queries)),
        limit=limit,
    )


def _source_candidates(
    searcher: Any, source_ids: list[str]
) -> dict[str, list[dict[str, Any]]]:
    source_ids = _validate_terms(source_ids, name="source_ids")
    if not source_ids:
        return {}
    grouped = {source_id: [] for source_id in source_ids}
    for source_id in source_ids:
        query = _constant_query(
            _and_query(
                [
                    _term_query("catalog_candidate", "true"),
                    _term_query("catalog_source", source_id),
                ]
            )
        )
        grouped[source_id] = _sorted_records(
            searcher,
            query,
            limit=MAX_SOURCE_VARIANTS,
        )
    return grouped


def search_candidates(
    searcher: Any, hits: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Hydrate canonical BM25 hits and expand unavailable source variants."""

    expanded: list[list[dict[str, Any]] | str] = []
    source_ids: list[str] = []
    for hit in hits:
        product_id = hit.get("product_id")
        sku = hit.get("sku")
        if not product_id or not sku:
            raise CatalogContractError("BM25 hit has no canonical identity")
        document = _raw_document(searcher, product_id)
        if document is None:
            raise CatalogContractError("BM25 hit cannot be loaded")
        product = document["product"]
        if str(product.get("sku")) != sku:
            raise CatalogContractError("BM25 hit does not match its stored product")
        if not {"catalog_candidate", "catalog_source_variant"} <= document.keys():
            raise CatalogContractError("BM25 hit has no runtime catalog flags")
        if document.get("catalog_candidate") == "true":
            expanded.append([catalog_record(document)])
        elif document.get("catalog_source_variant") == "true":
            source = product["source_listing_id"]
            expanded.append(source)
            source_ids.append(source)
        else:
            expanded.append([])
    by_source = _source_candidates(searcher, source_ids)
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for value in expanded:
        candidates = by_source.get(value, []) if isinstance(value, str) else value
        for candidate in candidates:
            key = (candidate["product_id"], candidate["sku"])
            if key not in seen:
                seen.add(key)
                results.append(candidate)
    return results


__all__ = [
    "CatalogContractError",
    "INDEX_CONTRACT_VERSION",
    "catalog_record",
    "filter_candidates",
    "products",
    "search_candidates",
]
