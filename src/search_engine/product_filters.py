"""Post-filter predicates for find_product, shared by the server (decode path),
the sidecar fast path's parity, and the tests.

Single source of truth: `server.py` imports these, `sidecar.rejects` mirrors
them against columnar values, and the test suite uses them as the oracle -- so
there is no third copy to drift out of parity.
"""


def is_filter_by_price(product, price):
    low, high = price
    if low is not None and product["price"] < low:
        return True
    if high is not None and product["price"] > high:
        return True
    return False


def is_filter_by_service(product, service):
    for serv in service:
        if serv not in product.get("service", []):
            return True
    return False


def is_filter_by_shop_id(product, shop_id):
    if shop_id and shop_id != product.get("shop_id"):
        return True
    return False


def shop_id_is_canonical(shop_id):
    """True when a query shop_id can be compared as an integer with identical
    results to the original string comparison.

    The sidecar stores shop_id as an int, so it may only take the fast path for
    canonical non-negative decimal strings. Non-canonical numerics -- leading
    zeros ("02003"), signs ("+2003", "-1"), surrounding space (" 2003 ") -- would
    flip the decision or collide with the null-shop sentinel, so they fall back
    to the string-comparison decode path.
    """
    return bool(shop_id) and shop_id.isdigit() and shop_id == str(int(shop_id))
