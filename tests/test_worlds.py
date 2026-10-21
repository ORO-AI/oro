"""World search store, params and routes, with a fake searcher (no Java)."""

import gzip
import hashlib
import importlib.util
import io
import sys
import tarfile
import types
from types import SimpleNamespace

import pytest

if (
    importlib.util.find_spec("flask") is None
):  # the store and search need Flask only for the routes
    sys.modules["flask"] = types.SimpleNamespace(
        Blueprint=None, jsonify=None, request=None
    )
from src.search_engine import worlds  # noqa: E402


class FakeSearcher:
    """Ranks documents (``id contents`` lines in ``index/docs``) by how many query words they hold."""

    def __init__(self, path):
        self.path, self.closed, self.bm25 = path, False, None
        lines = open(f"{path}/docs").read().splitlines()
        self.docs = [line.split(" ", 1) for line in lines]

    def set_bm25(self, k1, b):
        self.bm25 = (k1, b)

    def search(self, query, k):
        words = query[1].split() if isinstance(query, tuple) else query.split()
        scored = [
            (sum(w in text.split() for w in words), i, id_)
            for i, (id_, text) in enumerate(self.docs)
        ]
        if isinstance(query, tuple):
            scored = [
                s for s in scored if all(w in self.docs[s[1]][1].split() for w in words)
            ]
        ranked = sorted((s for s in scored if s[0]), key=lambda s: (-s[0], s[1]))
        return [
            SimpleNamespace(docid=id_, score=float(score))
            for score, _, id_ in ranked[:k]
        ]

    def close(self):
        self.closed = True


def archive(files):
    buf = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w") as tar,
    ):
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def world(text="a red shoe\nb blue shoe\nc red hat\n"):
    body = archive({"index/docs": text.encode()})
    return hashlib.sha256(body).hexdigest(), body


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(worlds, "_all_terms_query", lambda text: ("all", text))
    return worlds.Worlds(tmp_path / "worlds", FakeSearcher, capacity=2, keep=3)


def test_install_search_and_bm25_settings(store):
    wid, body = world()
    assert store.install(wid, {"bm25": {"k1": 1.2}}, io.BytesIO(body)) is True
    assert store.install(wid, {}, io.BytesIO(body)) is False
    w = store.get(wid)
    assert w.searcher.bm25 == (
        1.2,
        0.4,
    )  # the unpacked world keeps the settings it was installed with
    hits = worlds.search(w, "red: shoe!", {})["hits"]
    assert [h["id"] for h in hits] == [
        "a",
        "b",
        "c",
    ]  # a holds both words; b and c tie and rank by index order
    assert [h["id"] for h in worlds.search(w, "red shoe", {"k": 1})["hits"]] == ["a"]
    assert [
        h["id"] for h in worlds.search(w, "red shoe", {"all_terms": True})["hits"]
    ] == ["a"]
    assert worlds.search(w, " ~*? ", {}) == {"hits": []}


def test_default_and_custom_bm25(store, tmp_path):
    wid, body = world()
    store.install(wid, {"bm25": {"k1": 1.2, "b": 0.75}}, io.BytesIO(body))
    assert store.get(wid).searcher.bm25 == (1.2, 0.75)
    wid2, body2 = world("x y\n")
    store.install(wid2, {}, io.BytesIO(body2))
    assert store.get(wid2).searcher.bm25 == (0.9, 0.4)
    with pytest.raises(ValueError):
        store.install(wid2, {"bm25": {"k1": "1"}}, io.BytesIO(body2))


@pytest.mark.parametrize(
    "params",
    [
        {"k": 0},
        {"k": 1001},
        {"k": True},
        {"k": 2.0},
        {"k": "3"},
        {"all_terms": 1},
        {"all_terms": "yes"},
        None,
    ],
)
def test_bad_params(store, params):
    wid, body = world()
    store.install(wid, {}, io.BytesIO(body))
    with pytest.raises(ValueError):
        worlds.search(store.get(wid), "red", params)


def test_bad_query_and_bounds(store):
    wid, body = world()
    store.install(wid, {}, io.BytesIO(body))
    with pytest.raises(ValueError):
        worlds.search(store.get(wid), 7, {})
    assert len(worlds.search(store.get(wid), "shoe hat", {"k": 1000})["hits"]) == 3


def test_sha_mismatch_and_bad_id(store):
    wid, body = world()
    with pytest.raises(ValueError, match="sha256"):
        store.install(wid, {}, io.BytesIO(body + b"x"))
    with pytest.raises(ValueError):
        store.install("../" + wid[3:], {}, io.BytesIO(body))
    with pytest.raises(KeyError):
        store.get(wid)
    assert [p.name for p in store.root.iterdir()] == []


@pytest.mark.parametrize(
    "files",
    [
        {"index/../evil": b"x"},
        {"/etc/evil": b"x"},
        {"evil": b"x"},
        {"index/sub/file": b"x"},
        {"other/file": b"x"},
        {"index/": b""},
        {},
    ],
)
def test_unsafe_archives(store, files):
    body = archive(files)
    with pytest.raises(ValueError):
        store.install(hashlib.sha256(body).hexdigest(), {}, io.BytesIO(body))
    assert [p.name for p in store.root.iterdir()] == []


def test_links_and_garbage_rejected(store):
    buf = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buf, mode="wb") as gz,
        tarfile.open(fileobj=gz, mode="w") as tar,
    ):
        link = tarfile.TarInfo("index/docs")
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(link)
    for body in (buf.getvalue(), b"not a tar", gzip.compress(b"not a tar")):
        with pytest.raises(ValueError):
            store.install(hashlib.sha256(body).hexdigest(), {}, io.BytesIO(body))


def test_lru_open_and_on_disk(store):
    ids = []
    for i in range(4):
        wid, body = world(f"d{i} word\n")
        store.install(wid, {}, io.BytesIO(body))
        ids.append(wid)
    assert list(store.open) == ids[2:]  # capacity 2 open searchers
    on_disk = {p.name for p in store.root.iterdir()}
    assert on_disk == set(ids[1:])  # keep 3 unpacked worlds
    with pytest.raises(KeyError):
        store.get(ids[0])
    reopened = store.get(
        ids[1]
    )  # unpacked but closed: reopened from disk, evicting the oldest open one
    assert list(store.open) == [ids[3], ids[1]]
    assert worlds.search(reopened, "word", {})["hits"][0]["id"] == "d1"


def test_another_process_sees_installed_worlds(store):
    wid, body = world()
    store.install(wid, {"bm25": {"b": 0.5}}, io.BytesIO(body))
    other = worlds.Worlds(store.root, FakeSearcher, capacity=2, keep=3)
    assert other.get(wid).searcher.bm25 == (0.9, 0.5)


@pytest.fixture
def client(store):
    if not getattr(sys.modules["flask"], "Flask", None):
        pytest.skip("Flask is not installed")
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(worlds.blueprint(store))
    return app.test_client()


def test_routes(client):
    wid, body = world()
    settings = {"X-ORO-World-Settings": '{"bm25": {"k1": 1.0, "b": 0.5}}'}
    assert (
        client.put(f"/internal/worlds/{wid}", data=body, headers=settings).status_code
        == 201
    )
    assert (
        client.put(f"/internal/worlds/{wid}", data=body, headers=settings).status_code
        == 200
    )
    r = client.post(
        f"/internal/worlds/{wid}/search", json={"query": "red shoe", "params": {"k": 2}}
    )
    assert r.status_code == 200
    assert r.get_json() == {
        "hits": [{"id": "a", "score": 2.0}, {"id": "b", "score": 1.0}]
    }
    r = client.post(
        f"/internal/worlds/{wid}/search",
        json={"query": "red shoe", "params": {"all_terms": True}},
    )
    assert [h["id"] for h in r.get_json()["hits"]] == ["a"]
    assert (
        client.post(f"/internal/worlds/{wid}/search", json={"query": "red"}).status_code
        == 200
    )


def test_route_errors(client):
    wid, body = world()
    assert client.put(f"/internal/worlds/{wid}", data=body + b"x").status_code == 400
    assert client.put("/internal/worlds/nothex", data=body).status_code == 400
    assert (
        client.put(
            f"/internal/worlds/{wid}", data=body, headers={"X-ORO-World-Settings": "{"}
        ).status_code
        == 400
    )
    r = client.post(
        f"/internal/worlds/{wid}/search", json={"query": "red", "params": {}}
    )
    assert (r.status_code, r.get_json()) == (404, {"error": "unknown world"})
    assert client.put(f"/internal/worlds/{wid}", data=body).status_code == 201
    for payload in ({"query": "red", "params": {"k": 0}}, {"query": 3}, ["red"]):
        assert (
            client.post(f"/internal/worlds/{wid}/search", json=payload).status_code
            == 400
        )
