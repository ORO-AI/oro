"""World search: the BM25 indexes of world packs, installed at runtime.

A world is one index archive of a pack, addressed by its sha256: a gzipped tar whose members are only
``index/<file>`` (a Lucene index). The validator uploads it (``PUT /internal/worlds/<sha256>``, the index's settings
as JSON in the ``X-ORO-World-Settings`` header) and searches it (``POST /internal/worlds/<sha256>/search`` with
``{"query", "params"}``). Worlds are unpacked under a root the server's worker processes share; each process keeps
a bounded LRU of open searchers, and the root keeps the most recently used worlds.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
import zlib
from collections import Counter, OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO, Callable

from flask import Blueprint, jsonify, request

BM25 = {"k1": 0.9, "b": 0.4}  # the weights of an index whose settings name none
K_MAX = 1000
MAX_QUERY = 1024  # characters
MAX_TERMS = 1023  # distinct analyzed terms: under Lucene's 1024 clauses per query
STALE_S = 3600  # temp files older than this are left by a worker that died mid-install
MAX_BYTES = 4 << 30  # the most an archive may unpack to
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LUCENE_SPECIAL = str.maketrans({c: " " for c in '+-&|!(){}[]^"~*?:\\/'})


class SettingsConflict(ValueError):
    """The world is already installed with other settings."""


@dataclass
class World:
    id: str
    searcher: Any


class Worlds:
    """Installed worlds under ``root``: at most ``capacity`` open searchers and ``keep`` unpacked worlds."""

    def __init__(
        self,
        root: Path,
        searcher_factory: Callable[[str], Any],
        capacity: int,
        keep: int,
    ) -> None:
        self.root, self.factory, self.capacity, self.keep = (
            Path(root),
            searcher_factory,
            capacity,
            keep,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.open: OrderedDict[str, World] = OrderedDict()
        self.lock = threading.Lock()

    def install(self, world_id: str, settings: dict, body: BinaryIO) -> bool:
        """Verify, unpack and open the archive ``body`` as ``world_id``; False if it was already unpacked."""
        if not _SHA256.fullmatch(world_id) or not isinstance(settings, dict):
            raise ValueError("bad world id or settings")
        weights = _bm25(settings)
        fd, archive = tempfile.mkstemp(dir=self.root, prefix=".upload-")
        try:
            digest = hashlib.sha256()
            with os.fdopen(fd, "wb") as out:
                for chunk in iter(lambda: body.read(1 << 20), b""):
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != world_id:
                raise ValueError("archive sha256 does not match the world id")
            created = not (self.root / world_id).is_dir()
            if created:
                staging = Path(tempfile.mkdtemp(dir=self.root, prefix=".world-"))
                try:
                    _extract(Path(archive), staging / "index")
                    (staging / "settings.json").write_text(json.dumps(settings))
                    staging.rename(self.root / world_id)
                except OSError:
                    if not (
                        self.root / world_id
                    ).is_dir():  # not a concurrent install of the same world
                        raise
                finally:
                    shutil.rmtree(staging, ignore_errors=True)
        finally:
            os.unlink(archive)
        if (
            _bm25(json.loads((self.root / world_id / "settings.json").read_text()))
            != weights
        ):
            raise SettingsConflict("the world is installed with other settings")
        with self.lock:
            if world_id in self.open:
                self.open.move_to_end(world_id)
            else:
                try:
                    self._open(world_id)
                except Exception:
                    if created:
                        shutil.rmtree(self.root / world_id, ignore_errors=True)
                    raise
            self._prune(world_id)
        return created

    def get(self, world_id: str) -> World:
        with self.lock:
            if world_id in self.open:
                self.open.move_to_end(world_id)
                # _prune() orders worlds by on-disk mtime, but _open() is the
                # only other place that bumps it. A world that stays open
                # (hit here every time, never evicted from the in-memory
                # LRU) would otherwise keep the mtime from its last cold
                # open indefinitely, so _prune() can see it as the least
                # recently used and delete it -- a currently-open,
                # currently-serving world -- while unrelated, truly colder
                # worlds on disk survive. The root is shared by worker
                # processes, so another one's _prune() may already have
                # removed the directory; the searcher here is still open
                # and serves fine, so that is not an error.
                with suppress(FileNotFoundError):
                    os.utime(self.root / world_id, ns=(time.time_ns(),) * 2)
                return self.open[world_id]
            if (
                not _SHA256.fullmatch(world_id)
                or not (self.root / world_id / "settings.json").is_file()
            ):
                raise KeyError(world_id)
            return self._open(world_id)

    def _open(self, world_id: str) -> World:
        path = self.root / world_id
        searcher = self.factory(str(path / "index"))
        k1, b = _bm25(json.loads((path / "settings.json").read_text()))
        searcher.set_bm25(k1, b)
        os.utime(
            path, ns=(time.time_ns(),) * 2
        )  # the file system's clock is too coarse to order opens
        world = self.open[world_id] = World(world_id, searcher)
        while len(self.open) > self.capacity:
            _close(self.open.popitem(last=False)[1])
        return world

    def _prune(self, current: str) -> None:
        worlds = sorted((p for p in self.root.iterdir() if _SHA256.fullmatch(p.name) and p.name != current),
                        key=lambda p: p.stat().st_mtime_ns, reverse=True)  # fmt: skip
        for path in worlds[max(0, self.keep - 1) :]:
            if path.name in self.open:
                _close(self.open.pop(path.name))
            shutil.rmtree(path, ignore_errors=True)
        for path in self.root.iterdir():
            with suppress(FileNotFoundError):  # another process may sweep it first
                if (
                    path.name.startswith((".upload-", ".world-"))
                    and path.lstat().st_mtime < time.time() - STALE_S
                ):
                    shutil.rmtree(path) if path.is_dir() else path.unlink()


def _close(world: World) -> None:
    close = getattr(world.searcher, "close", None)
    if close:
        close()


def _bm25(settings: dict) -> tuple[float, float]:
    """The (k1, b) weights of ``settings``: k1 in [0, 10] (Lucene takes any finite k1 >= 0), b in [0, 1]."""
    weights = settings.get("bm25") or {}
    if not isinstance(weights, dict):
        raise ValueError("bad bm25 settings")
    k1, b = (BM25 | weights)["k1"], (BM25 | weights)["b"]
    if not all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in (k1, b)
    ) or not (0 <= k1 <= 10 and 0 <= b <= 1):  # also rejects NaN and infinities
        raise ValueError("bad bm25 settings")
    return float(k1), float(b)


def _extract(archive: Path, dest: Path) -> None:
    """Unpack ``archive`` into ``dest`` if its members are only regular ``index/<file>`` files."""
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            members = tar.getmembers()
            for m in members:
                folder, _, name = m.name.partition("/")
                if (
                    not m.isfile()
                    or folder != "index"
                    or name in ("", ".", "..")
                    or "/" in name
                ):
                    raise ValueError(f"unsafe archive member: {m.name!r}")
            if not members or sum(m.size for m in members) > MAX_BYTES:
                raise ValueError("archive is empty or too large")
            dest.mkdir()
            for m in members:
                with (
                    tar.extractfile(m) as src,
                    open(dest / m.name.partition("/")[2], "xb") as out,
                ):
                    shutil.copyfileobj(src, out)
    except (
        tarfile.TarError,
        EOFError,
        gzip.BadGzipFile,
        zlib.error,
        FileExistsError,
    ) as exc:
        raise ValueError(f"bad archive: {exc}") from exc


def search(world: World, query: str, params: dict) -> dict:
    """The top ``k`` documents for ``query`` by BM25 (with ``all_terms``, only those holding every query term);
    equal scores rank by index order (id order)."""
    if not isinstance(params, dict):
        raise ValueError("bad search params")
    k, all_terms = params.get("k", 10), params.get("all_terms", False)
    if not (isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= K_MAX and isinstance(all_terms, bool)
            and isinstance(query, str)):  # fmt: skip
        raise ValueError("bad search params")
    if len(query) > MAX_QUERY:
        raise ValueError(f"query is longer than {MAX_QUERY} characters")
    try:
        query.encode()
    except UnicodeEncodeError:
        raise ValueError("query is not valid Unicode") from None
    text = " ".join(query.translate(_LUCENE_SPECIAL).split())
    terms = Counter(_analyzer().analyze(text)) if text else Counter()
    if len(terms) > MAX_TERMS:
        raise ValueError(f"query has more than {MAX_TERMS} distinct terms")
    hits = (
        world.searcher.search(_all_terms_query(terms) if all_terms else text, k)
        if terms
        else []
    )
    return {"hits": [{"id": h.docid, "score": float(h.score)} for h in hits]}


@lru_cache(maxsize=1)
def _analyzer():
    from pyserini.analysis import Analyzer, get_lucene_analyzer

    return Analyzer(get_lucene_analyzer())


def _all_terms_query(terms: Counter):
    """A Lucene query requiring every analyzed term, each weighted by its count as in the bag-of-words query."""
    from pyserini.search.lucene import querybuilder as qb

    builder = qb.get_boolean_query_builder()
    for term, count in sorted(terms.items()):
        clause = qb.JTermQuery(qb.JTerm("contents", term))
        builder.add(
            qb.JBoostQuery(clause, float(count)) if count > 1 else clause,
            qb.JBooleanClauseOccur["must"].value,
        )
    return builder.build()


def from_env(searcher_factory: Callable[[str], Any]) -> Worlds:
    return Worlds(Path(os.environ.get("WORLDS_DIR", "/tmp/worlds")), searcher_factory,
                  capacity=int(os.environ.get("WORLDS_OPEN", "8")), keep=int(os.environ.get("WORLDS_KEEP", "32")))  # fmt: skip


def blueprint(store: Worlds) -> Blueprint:
    routes = Blueprint("worlds", __name__)

    @routes.put("/internal/worlds/<world_id>")
    def install(world_id: str):
        try:
            settings = json.loads(request.headers.get("X-ORO-World-Settings", "{}"))
            created = store.install(world_id, settings, request.stream)
        except SettingsConflict as exc:
            return jsonify({"error": str(exc)}), 409
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"id": world_id}), 201 if created else 200

    @routes.post("/internal/worlds/<world_id>/search")
    def search_world(world_id: str):
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "expected a JSON object"}), 400
        try:
            world = store.get(world_id)
        except KeyError:
            return jsonify({"error": "unknown world"}), 404
        try:
            return jsonify(search(world, body.get("query"), body.get("params", {})))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    return routes


def app():
    """The world search server alone (no baked catalog index): ``gunicorn 'src.search_engine.worlds:app()'``."""
    from flask import Flask
    from pyserini.search.lucene import LuceneSearcher

    server = Flask(__name__)
    server.register_blueprint(blueprint(from_env(LuceneSearcher)))
    server.get("/health")(
        lambda: jsonify({"status": "healthy", "service": "search-worlds"})
    )
    return server
