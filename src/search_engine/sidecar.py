"""Columnar sidecar for fast find_product filtering.

The search hot path scores up to CAPACITY_FULL BM25 candidates and, for each,
must read three fields -- shop_id, price, service -- to apply the post-filters.
Doing that by decoding every candidate's full stored JSON
(`json.loads(searcher.doc(id).raw())`) is the dominant per-request cost once a
selective filter forces a deep scan.

This sidecar precomputes those fields into compact, memory-mapped columnar
arrays keyed by product_id. Filtering then reads them with an O(log n) binary
search instead of a full-document decode; only the <=50 survivors that make the
paginated response are decoded. The arrays are `numpy.memmap`, so a single
physical copy is shared across all gunicorn workers via the OS page cache
(~65 MB total for the full corpus, independent of worker count).

Built by `build_sidecar.py`; loaded here at server start. If the directory is
absent, fails to load, or its stamped doc count does not match the live index,
`load()` returns None and the server falls back to the original
decode-every-candidate path -- so shipping this code is a no-op until a base
image that bundles a matching sidecar is deployed.
"""
import json
import os
import sys

import numpy as np

NULL_SHOP = -1  # sentinel stored for products with no shop_id
_UNDECIDED = object()  # sentinel: sidecar cannot decide this row, use decode path


class Sidecar:
    def __init__(self, pid, price, shop, svc, vocab):
        self._pid = pid
        self._price = price
        self._shop = shop
        self._svc = svc
        self._vocab = vocab

    def lookup(self, docid):
        """Return the row index for a collection docid, or None if absent."""
        try:
            key = int(docid)
        except (TypeError, ValueError):
            return None
        i = int(np.searchsorted(self._pid, key))
        if i >= len(self._pid) or self._pid[i] != key:
            return None
        return i

    def reqmask(self, service_list):
        """Bitmask of required services; unknown tokens get a bit no product
        has, so a query for an absent service matches nothing (as the original
        `serv not in product.service` check would)."""
        m = 0
        for s in service_list or ():
            bit = self._vocab.get(s)
            m |= (1 << bit) if bit is not None else (1 << 63)
        return m

    def rejects(self, i, shop_id_int, price, reqmask):
        """Mirror of the is_filter_by_* predicates over the columns.

        Returns True to drop the row, False to keep it, or the sentinel
        `Sidecar.undecided` when it cannot match the decode path exactly -- a
        null (NaN) price under an active price filter, where the original code
        raises on `None < low`. The caller must decode-and-filter those rows so
        behavior stays identical.
        """
        if shop_id_int is not None and self._shop[i] != shop_id_int:
            return True
        low, high = price
        if low is not None or high is not None:
            pr = self._price[i]
            if np.isnan(pr):
                return _UNDECIDED
            if low is not None and pr < low:
                return True
            if high is not None and pr > high:
                return True
        if reqmask and (int(self._svc[i]) & reqmask) != reqmask:
            return True
        return False

    undecided = _UNDECIDED


def load(sidecar_dir, expected_num_docs=None):
    """Load the sidecar from a directory, or return None if unavailable.

    Never raises: a missing file, malformed array, or a stamped doc count that
    does not match `expected_num_docs` (the live index size) all yield None so
    the caller transparently falls back to the decode path.
    """
    try:
        if not sidecar_dir or not os.path.isdir(sidecar_dir):
            return None
        arrs = {}
        for name in ("pid", "price", "shop", "svc"):
            path = os.path.join(sidecar_dir, name + ".npy")
            if not os.path.isfile(path):
                return None
            arrs[name] = np.load(path, mmap_mode="r")
        with open(os.path.join(sidecar_dir, "meta.json")) as f:
            meta = json.load(f)
        vocab = meta["vocab"]
        n = len(arrs["pid"])
        if any(len(arrs[k]) != n for k in ("price", "shop", "svc")):
            return None
        if meta.get("num_docs") != n:
            print("Sidecar meta doc count mismatch; using decode path", file=sys.stderr)
            return None
        if expected_num_docs is not None and n != expected_num_docs:
            print(
                f"Sidecar/index mismatch ({n} vs {expected_num_docs}); using decode path",
                file=sys.stderr,
            )
            return None
        print(f"Sidecar loaded: {n} products from {sidecar_dir}", file=sys.stderr)
        return Sidecar(arrs["pid"], arrs["price"], arrs["shop"], arrs["svc"], vocab)
    except Exception as e:  # noqa: BLE001 - fallback must never break startup
        print(f"Sidecar unavailable ({e!r}); using decode path", file=sys.stderr)
        return None
