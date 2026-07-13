"""Columnar sidecar for fast find_product filtering.

The search hot path scores up to CAPACITY_FULL BM25 candidates and, for each,
must read four fields -- shop_id, price, service, sold_count -- to apply the
post-filters. Doing that by decoding every candidate's full stored JSON
(`json.loads(searcher.doc(id).raw())`) is the dominant per-request cost once a
selective filter forces a deep scan.

This sidecar precomputes those four fields into compact, memory-mapped columnar
arrays keyed by product_id. Filtering then reads them with an O(log n) binary
search instead of a full-document decode; only the <=50 survivors that make the
paginated response are decoded. The arrays are `numpy.memmap`, so a single
physical copy is shared across all gunicorn workers via the OS page cache
(~90 MB total for the full corpus, independent of worker count).

Built by `build_sidecar.py`; loaded here at server start. If the directory is
absent or fails to load, `load()` returns None and the server falls back to the
original decode-every-candidate path -- so shipping this code is a no-op until a
base image that bundles the sidecar is deployed.
"""
import json
import os
import sys

_SERVICE_KEYS = ("pid", "price", "shop", "sold", "svc", "vocab")
_IMPOSSIBLE_BIT = 1 << 63  # requested service token absent from corpus -> matches nothing


class Sidecar:
    def __init__(self, pid, price, shop, sold, svc, vocab):
        self._pid = pid
        self._price = price
        self._shop = shop
        self._sold = sold
        self._svc = svc
        self._vocab = vocab

    def lookup(self, docid):
        """Return the row index for a collection docid, or None if absent."""
        import numpy as np

        try:
            key = int(docid)
        except (TypeError, ValueError):
            return None
        i = int(np.searchsorted(self._pid, key))
        if i >= len(self._pid) or self._pid[i] != key:
            return None
        return i

    def reqmask(self, service_list):
        """Bitmask of required services; unknown tokens get an impossible bit."""
        m = 0
        for s in service_list or ():
            bit = self._vocab.get(s)
            m |= (1 << bit) if bit is not None else _IMPOSSIBLE_BIT
        return m

    def rejects(self, i, shop_id_int, price, reqmask):
        """Mirror of the is_filter_by_* predicates, read from the columns.

        Returns True when the row should be dropped. `price` is the
        (low, high) tuple; `shop_id_int` is the parsed query shop id or None.
        """
        if shop_id_int is not None and self._shop[i] != shop_id_int:
            return True
        low, high = price
        pr = self._price[i]
        if low is not None and pr < low:
            return True
        if high is not None and pr > high:
            return True
        if reqmask and (int(self._svc[i]) & reqmask) != reqmask:
            return True
        return False


def load(sidecar_dir):
    """Load the sidecar from a directory, or return None if unavailable.

    Never raises: any missing file, numpy import failure, or malformed array
    yields None so the caller transparently falls back to the decode path.
    """
    try:
        import numpy as np

        if not sidecar_dir or not os.path.isdir(sidecar_dir):
            return None
        arrs = {}
        for name in ("pid", "price", "shop", "sold", "svc"):
            path = os.path.join(sidecar_dir, name + ".npy")
            if not os.path.isfile(path):
                return None
            arrs[name] = np.load(path, mmap_mode="r")
        with open(os.path.join(sidecar_dir, "vocab.json")) as f:
            vocab = json.load(f)
        n = len(arrs["pid"])
        if any(len(arrs[k]) != n for k in ("price", "shop", "sold", "svc")):
            return None
        print(f"Sidecar loaded: {n} products from {sidecar_dir}", file=sys.stderr)
        return Sidecar(arrs["pid"], arrs["price"], arrs["shop"], arrs["sold"], arrs["svc"], vocab)
    except Exception as e:  # noqa: BLE001 - fallback must never break startup
        print(f"Sidecar unavailable ({e!r}); using decode path", file=sys.stderr)
        return None
