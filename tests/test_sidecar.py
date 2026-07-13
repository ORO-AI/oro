"""Parity tests for the find_product filter sidecar.

The oracle is the real predicate module (`product_filters`), the same code the
server's decode path runs -- so this is an independent check, not a copy kept in
sync by hand. Does not import server.py (which loads a Lucene index at import).
"""
import json

import pytest

np = pytest.importorskip("numpy")

from src.search_engine import build_sidecar, product_filters as pf, sidecar as sidecar_mod  # noqa: E402

SERVICES = ["official", "freeShipping", "COD", "flashsale"]


def oracle_reject(product, shop_id, price, service):
    if product_filters_shop(product, shop_id):
        return True
    if pf.is_filter_by_price(product, price):
        return True
    if pf.is_filter_by_service(product, service):
        return True
    return False


def product_filters_shop(product, shop_id):
    return pf.is_filter_by_shop_id(product, shop_id)


def make_products(n=40):
    products = []
    for i in range(n):
        products.append(
            {
                "product_id": str(100000 + i),
                "shop_id": str(2000 + (i % 5)),
                "price": float(10 * (i % 13)) + 0.99,
                "sold_count": i % 7,
                "service": SERVICES[: (i % 4)],
            }
        )
    # a null-price row and a null-shop row (both handled by build_sidecar)
    products.append({"product_id": "200001", "shop_id": "2001", "price": None, "sold_count": 0, "service": []})
    products.append({"product_id": "200002", "shop_id": None, "price": 50.0, "sold_count": 0, "service": []})
    return products


def _build(tmp_path, products):
    docs = tmp_path / "documents.jsonl"
    with open(docs, "w") as f:
        for p in products:
            f.write(json.dumps({"id": p["product_id"], "contents": "x", "product": p}) + "\n")
    out = tmp_path / "sidecar"
    build_sidecar.build(str(docs), str(out))
    return out


@pytest.fixture
def built(tmp_path):
    products = make_products()
    out = _build(tmp_path, products)
    sc = sidecar_mod.load(str(out), expected_num_docs=len(products))
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
        ("9999", (None, None), []),
        (None, (None, None), ["unknown_svc"]),
    ],
)
def test_filter_parity(built, shop_id, price, service):
    products, sc = built
    shop_int = int(shop_id) if shop_id else None
    reqmask = sc.reqmask(service)
    for p in products:
        i = sc.lookup(p["product_id"])
        assert i is not None
        verdict = sc.rejects(i, shop_int, price, reqmask)
        if verdict is sc.undecided:
            # only null price under an active price filter, where the decode
            # path raises; the server defers to it, so behavior is identical.
            assert p["price"] is None and (price[0] is not None or price[1] is not None)
            continue
        assert verdict == oracle_reject(p, shop_id, price, service)


def test_null_price_defers_only_under_price_filter(built):
    products, sc = built
    i = sc.lookup("200001")  # price=None
    assert sc.rejects(i, None, (None, None), 0) is False   # no price filter -> kept
    assert sc.rejects(i, None, (10.0, None), 0) is sc.undecided
    assert sc.rejects(i, None, (None, 99.0), 0) is sc.undecided


def test_null_shop_row(built):
    products, sc = built
    i = sc.lookup("200002")  # shop_id=None -> stored NULL_SHOP
    assert sc.rejects(i, 2003, (None, None), 0) is True    # any real shop filter drops it
    assert sc.rejects(i, None, (None, None), 0) is False   # no shop filter -> kept


@pytest.mark.parametrize(
    "shop_id,canonical",
    [("2003", True), ("0", True), ("02003", False), ("+2003", False), ("-1", False), (" 2003 ", False), ("", False)],
)
def test_shop_id_canonical(shop_id, canonical):
    assert pf.shop_id_is_canonical(shop_id) is canonical


def test_lookup_missing(built):
    _, sc = built
    assert sc.lookup("999999999") is None
    assert sc.lookup(None) is None
    assert sc.lookup("not-an-int") is None


def test_load_guards(tmp_path):
    assert sidecar_mod.load(str(tmp_path / "nope")) is None
    assert sidecar_mod.load(None) is None
    out = _build(tmp_path, make_products())
    assert sidecar_mod.load(str(out), expected_num_docs=999) is None  # index/sidecar drift
    assert sidecar_mod.load(str(out), expected_num_docs=len(make_products())) is not None
