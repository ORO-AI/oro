"""Transport fixture migrated in pytest-owned temporary files only.

The checked-in generator archive and catalog retain their original SHA checks; the
test-owned copy is rewritten as composed situation rows for the installed runtime.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from oro_env_runtime import contracts
from oro_env_runtime.grading import Grading
from oro_env_runtime.delivery import DELIVERY_SUBSET_VERSION
from oro_env_runtime.pack import (
    COMPILED_EPOCH_VERSION,
    fingerprint,
    sha256_file,
    task_set_fingerprint,
)
from oro_env_runtime.reward import REWARD_VERSION
from oro_env_runtime.search_index import source_listing_id
from oro_env_runtime.situation import LINE_SLOTS
from oro_env_runtime.schema import CandidateRef, TaskSpec
from validator.env_pack_loader import LoadedPack

COMPAT_ARCHIVE = (
    Path(__file__).with_name("fixtures") / "oro_env_runtime_compat_v1.tar.gz"
)
COMPAT_METADATA = COMPAT_ARCHIVE.with_name("oro_env_runtime_compat_v1.json")
COMPAT_PRODUCTS = COMPAT_ARCHIVE.with_name("oro_env_runtime_compat_v1_products.jsonl")


# Legacy task fields the checked-in archive carries; composed rows have none of them.
_LEGACY_FIELDS = ("event_rule", "event_rules", "interventions", "preferred", "latent_prefs")
# Every stock line a situation must word: the line's name followed by its slots.
_LINES = {
    key: " ".join((key, *(f"{{{slot}}}" for slot in slots)))
    for key, slots in LINE_SLOTS.items()
}


def _migrate_to_composed(epoch: Path, manifest: dict) -> None:
    """Rewrite the test-owned copy of the legacy archive as composed situation rows: each
    task keeps its goal, hard constraints and accepted set, and states them as a category
    requirement over its accepted keys plus its budget."""

    task_path = epoch / "data/tasks/private_tasks.jsonl"
    task_rows = [json.loads(line) for line in task_path.read_text().splitlines()]
    seed_start = min(row["task"]["seed"] for row in task_rows)
    for index, row in enumerate(task_rows):
        task = {k: v for k, v in row["task"].items() if k not in _LEGACY_FIELDS}
        keys = list(task["acceptance"]["acceptable_keys"])
        hard = task["hard"]
        task.update(
            family="composed",
            seed=seed_start + index,
            # The source row's id keeps the rewritten rows semantically distinct.
            family_payload={"world": {"regime": "qualifying"}, "compat_source": row["task_id"]},
            situation={
                "preset": "compat",
                "lines": _LINES,
                "requirements": [
                    {"id": "cat", "predicate": {"kind": "category", "value": "*"}},
                    {
                        "id": "budget",
                        "predicate": {
                            "kind": "budget",
                            "max": hard["budget"],
                            "currency": hard["currency"],
                        },
                    },
                ],
            },
            grading=Grading(
                family="composed",
                decoupled=True,
                situation_tables={"cat": dict.fromkeys(keys, True)},
            ).model_dump(mode="json"),
        )
        row["task_id"] = f"TF8-composed-{task['seed']}"
        # Qualifying deliveries keep the accepted keys beside the runtime's task model.
        row["task"] = {
            **TaskSpec.model_validate(task).model_dump(mode="json"),
            "acceptance": {"acceptable_keys": keys},
        }
    task_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in task_rows
        )
    )
    manifest["pack_version"] = COMPILED_EPOCH_VERSION
    manifest["reward"] = {"default": REWARD_VERSION}
    manifest["epoch"].update(
        families=["composed"],
        family_counts={"composed": len(task_rows)},
        count_per_family=len(task_rows),
        seed_range={"start": seed_start, "end_exclusive": seed_start + len(task_rows)},
        task_set_fingerprint=fingerprint(
            [
                {"task_id": row["task_id"], "task_fingerprint": fingerprint(row["task"])}
                for row in task_rows
            ]
        ),
    )


def _write_checksums(epoch: Path) -> None:
    (epoch / "checksums.sha256").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(epoch).as_posix()}\n"
            for path in sorted(epoch.rglob("*"))
            if path.is_file() and path.name != "checksums.sha256"
        )
    )


def delivery_archive(
    epoch: Path,
    task_ids: list[str],
    *,
    scope: str,
    parent_pack_sha256: str,
    scope_id: str | None = None,
    runtime_contract: int | None = contracts.RUNTIME_CONTRACT,
) -> bytes:
    """Delivery bytes for ``task_ids`` of ``epoch``, declared as the Backend declares
    them; ``runtime_contract=None`` leaves the contract out, as older deliveries did."""

    out = epoch.parent / f"delivery-{fingerprint([scope, scope_id, task_ids])[:12]}"
    shutil.copytree(epoch, out / "epoch")
    task_path = out / "epoch/data/tasks/private_tasks.jsonl"
    rows = [
        row
        for line in task_path.read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row["task_id"] in task_ids
    ]
    task_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    manifest = json.loads((out / "epoch/manifest.json").read_text())
    manifest["epoch"].update(
        family_counts={"composed": len(rows)},
        task_set_fingerprint=task_set_fingerprint(rows),
    )
    manifest["delivery"] = {
        "version": DELIVERY_SUBSET_VERSION,
        "scope": scope,
        "parent_pack_sha256": parent_pack_sha256,
        "task_ids": [row["task_id"] for row in rows],
        **({"scope_id": scope_id} if scope == "race" else {}),
        **({} if runtime_contract is None else {"runtime_contract": runtime_contract}),
    }
    (out / "epoch/manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    _write_checksums(out / "epoch")
    return Path(shutil.make_archive(str(out / "delivery"), "gztar", out, "epoch")).read_bytes()


def accepted_ref(task: TaskSpec) -> CandidateRef:
    """The first listing the compat task accepts (its rows grade a category table)."""

    product_id, _, sku = next(iter(task.grading.situation_tables["cat"])).partition("::")
    return CandidateRef(product_id=product_id, sku=sku)


@pytest.fixture
def compiled_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Extract the trusted compatibility archive without importing the generator."""

    metadata = json.loads(COMPAT_METADATA.read_text())
    artifact = COMPAT_ARCHIVE.read_bytes()
    assert len(artifact) == metadata["artifact_size_bytes"]
    assert hashlib.sha256(artifact).hexdigest() == metadata["pack_sha256"]
    local_archive = tmp_path / "epoch.tar.gz"
    shutil.copy2(COMPAT_ARCHIVE, local_archive)
    shutil.unpack_archive(local_archive, tmp_path)
    epoch = tmp_path / "epoch"
    assert epoch.is_dir()

    manifest = json.loads((epoch / "manifest.json").read_text())
    products = {
        row["product_id"]: row
        for line in COMPAT_PRODUCTS.read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    assert (
        hashlib.sha256(COMPAT_PRODUCTS.read_bytes()).hexdigest()
        == manifest["source_sha256"]["products_schema_b.jsonl"]
    )
    manifest["contracts"]["runtime"] = contracts.RUNTIME_VERSION
    _migrate_to_composed(epoch, manifest)
    (epoch / "tf4_hybrid_release_gate.json").unlink(missing_ok=True)
    (epoch / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    _write_checksums(epoch)
    shutil.make_archive(str(epoch), "gztar", root_dir=tmp_path, base_dir="epoch")

    def catalog_record(product_id: str) -> dict:
        product = products[product_id]
        variant = product["variants"][0]
        url = (product.get("source") or {}).get("url") or ""
        return {
            "admitted": True,
            "brand": product.get("brand"),
            "category_path": product.get("category_path") or [],
            "currency": product["pricing"]["currency"],
            "description": (product.get("descriptions") or {}).get("long") or "",
            "in_stock": variant["in_stock"] is True,
            "main_image_url": "",
            "options": variant.get("options"),
            "price": variant["price"],
            "product_id": product_id,
            "product_url": url,
            "sku": variant["sku"],
            "source_listing_id": source_listing_id(product_id, url),
            "specification": product.get("specification") or {},
            "title": product.get("title") or "",
        }

    records = {product_id: catalog_record(product_id) for product_id in products}

    class StubSearch:
        identity = manifest["search"]

        def bm25(self, query: str, k: int = 10) -> list[dict[str, str]]:
            if not query.strip():
                return []
            return [
                {
                    "product_id": record["product_id"],
                    "sku": record["sku"],
                }
                for record in records.values()
            ][:k]

        def search_catalog(self, query: str, k: int = 10) -> list[dict]:
            return [records[hit["product_id"]] for hit in self.bm25(query, k)]

        def catalog_products(self, product_ids: list[str]) -> list[dict]:
            return [
                records[product_id]
                for product_id in product_ids
                if product_id in records
            ]

        def filter_catalog(
            self,
            *,
            category: str | None,
            brand: str | None,
            max_price: float | None,
            limit: int,
        ) -> list[dict]:
            return [
                record
                for record in records.values()
                if (
                    category is None
                    or category.casefold()
                    in " ".join(record["category_path"]).casefold()
                )
                and (
                    brand is None or brand.casefold() == str(record["brand"]).casefold()
                )
                and (max_price is None or record["price"] <= max_price)
            ][:limit]

    def client(*_args, expected_identity=None, **_kwargs):  # noqa: ANN001, ANN202
        if expected_identity is not None:
            assert expected_identity == StubSearch.identity
        return StubSearch()

    monkeypatch.setattr("oro_env_runtime.runtime.SearchServerClient", client)
    return epoch


@pytest.fixture
def loaded_pack(compiled_epoch: Path, tmp_path: Path) -> LoadedPack:
    """Load the shared sealed epoch through the validator-facing pack handle."""

    rows = [
        json.loads(line)
        for line in (compiled_epoch / "data/tasks/private_tasks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    archive = compiled_epoch.parent / f"{compiled_epoch.name}.tar.gz"
    return LoadedPack(
        pack_dir=compiled_epoch,
        manifest=json.loads(
            (compiled_epoch / "manifest.json").read_text(encoding="utf-8")
        ),
        task_specs=[TaskSpec.model_validate(row["task"]) for row in rows],
        task_ids=[row["task_id"] for row in rows],
        pack_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        metadata={},
        _scratch_dir=tmp_path / "loader-owned-elsewhere",
    )


__all__ = ["COMPAT_ARCHIVE", "COMPAT_METADATA", "accepted_ref", "delivery_archive"]
