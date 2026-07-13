"""Build the find_product filter sidecar from the index source JSONL.

Reads the same `documents.jsonl` that feeds the Lucene index builder and emits
compact columnar arrays (keyed by product_id, sorted for binary search) plus a
service vocabulary. See sidecar.py for how the server consumes them.

    python -m src.search_engine.build_sidecar <documents.jsonl> <out_dir>

Runs as a post-step of the base-image build, alongside the Lucene index build,
so the two stay in lockstep. Prices are stored as float64 so filter comparisons
are bit-identical to the decode path (which reads the same JSON float).
"""
import json
import os
import sys

import numpy as np

# Bit width of the service bitmask column (int64); build fails loudly if the
# corpus ever exceeds this so we never silently drop a service token.
_MAX_SERVICES = 63


def build(documents_path, out_dir):
    pids, prices, shops, solds, svc_sets = [], [], [], [], []
    vocab = {}
    with open(documents_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            p = doc.get("product", doc)
            pids.append(int(p["product_id"]))
            prices.append(float(p["price"]) if p.get("price") is not None else float("nan"))
            shops.append(int(p["shop_id"]) if p.get("shop_id") is not None else -1)
            solds.append(int(p.get("sold_count") or 0))
            svs = p.get("service") or []
            for s in svs:
                if s not in vocab:
                    vocab[s] = len(vocab)
            svc_sets.append(svs)

    if len(vocab) > _MAX_SERVICES:
        raise ValueError(f"service vocab {len(vocab)} exceeds {_MAX_SERVICES}-bit mask")

    n = len(pids)
    pid = np.asarray(pids, dtype=np.int64)
    price = np.asarray(prices, dtype=np.float64)
    shop = np.asarray(shops, dtype=np.int64)
    sold = np.asarray(solds, dtype=np.int64)
    svc = np.zeros(n, dtype=np.int64)
    for idx, svs in enumerate(svc_sets):
        m = 0
        for s in svs:
            m |= 1 << vocab[s]
        svc[idx] = m

    # sort by product_id so the server can binary-search lookups
    order = np.argsort(pid, kind="stable")

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "pid.npy"), pid[order])
    np.save(os.path.join(out_dir, "price.npy"), price[order])
    np.save(os.path.join(out_dir, "shop.npy"), shop[order])
    np.save(os.path.join(out_dir, "sold.npy"), sold[order])
    np.save(os.path.join(out_dir, "svc.npy"), svc[order])
    with open(os.path.join(out_dir, "vocab.json"), "w") as fh:
        json.dump(vocab, fh)
    print(f"Sidecar built: {n} products, {len(vocab)} services -> {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m src.search_engine.build_sidecar <documents.jsonl> <out_dir>", file=sys.stderr)
        sys.exit(2)
    build(sys.argv[1], sys.argv[2])
