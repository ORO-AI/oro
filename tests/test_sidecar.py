"""Parity tests for the find_product filter sidecar.

Verifies that filtering via the columnar sidecar produces exactly the same
accept/reject decision as the original per-product predicate logic, across a
grid of shop_id / price / service filters. Does not import server.py (which
loads a Lucene index at import); it re-states the reference predicates locally.
"""
import json

import pytest

np = pytest.importorskip("numpy")

from src.search_engine import build_sidecar, sidecar as sidecar_mod  # noqa: E402


# --- reference predicates: copied verbatim from server.is_filter_by_* ---
def ref_reject(product, shop_id, price, service):
    if shop_id and shop_id != product.get("shop_id"):
        return True
    low, high = price
    if low is not None and product["price"] < low:
        return True
    if high is not None and product["price"] > high:
        return True
    for serv in service:
        if serv not in product.get("service", []):
            return True
    return False


SERVICES = ["official", "freeShipping", "COD", "flashsale"]


def make_products(n=40):
    products = []
    for i in range(n):
        products.append(
            {
                "product_id": str(100000 + i),
                "shop_id": str(2000 + (i % 5)),
                "price": float(10 * (i % 13)) + 0.99,
                "sold_count": i % 7,
                "service": SERVICES[: (i % 4)],  # 0..3 services
            }
        )
    return products


@pytest.fixture
def built(tmp_path):
    products = make_products()
    docs = tmp_path / "documents.jsonl"
    with open(docs, "w") as f:
        for p in products:
            f.write(json.dumps({"id": p["product_id"], "contents": "x", "product": p}) + "\n")
    out = tmp_path / "sidecar"
    build_sidecar.build(str(docs), str(out))
    sc = sidecar_mod.load(str(out))
    assert sc is not None
    return products, sc


@pytest.mark.parametrize(
    "shop_id,price,service",
    [
        (None, (None, None), []),
        ("2003", (None, None), []),
        (None, (20.0, 80.0), []),
        (None, (None, 50.0), []),
        (None, (30.0, None), []),
        (None, (None, None), ["freeShipping"]),
        (None, (None, None), ["official", "COD"]),
        ("2001", (10.0, 100.0), ["freeShipping"]),
        ("9999", (None, None), []),          # shop that matches nothing
        (None, (None, None), ["unknown_svc"]),  # token absent from corpus
    ],
)
def test_filter_parity(built, shop_id, price, service):
    products, sc = built
    shop_int = int(shop_id) if shop_id else None
    reqmask = sc.reqmask(service)
    for p in products:
        i = sc.lookup(p["product_id"])
        assert i is not None
        got = sc.rejects(i, shop_int, price, reqmask)
        want = ref_reject(p, shop_id, price, service)
        assert got == want, (p["product_id"], shop_id, price, service, got, want)


def test_lookup_missing(built):
    _, sc = built
    assert sc.lookup("999999999") is None
    assert sc.lookup(None) is None
    assert sc.lookup("not-an-int") is None


def test_load_absent_dir_returns_none(tmp_path):
    assert sidecar_mod.load(str(tmp_path / "does-not-exist")) is None
    assert sidecar_mod.load(None) is None
