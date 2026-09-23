"""Transport fixture migrated in pytest-owned temporary files only.

The checked-in generator archive and catalog retain their original SHA checks.
Family stubs keep this fixture scoped to transport tests, not real-task QA.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from oro_env_runtime import contracts, verify as runtime_verify
from oro_env_runtime.pack import write_checksums
from oro_env_runtime.reward import (
    TF4_RELEASE_GATE_PATH,
    tf4_release_gate_fingerprint,
    tf4_release_gate_passed,
)
from oro_env_runtime.search_index import source_listing_id
from oro_env_runtime.schema import TaskSpec
from oro_env_runtime.tf4_judge_contract import judge_contract
from validator.env_pack_loader import LoadedPack

COMPAT_ARCHIVE = (
    Path(__file__).with_name("fixtures") / "oro_env_runtime_compat_v1.tar.gz"
)
COMPAT_METADATA = COMPAT_ARCHIVE.with_name("oro_env_runtime_compat_v1.json")
COMPAT_PRODUCTS = COMPAT_ARCHIVE.with_name("oro_env_runtime_compat_v1_products.jsonl")


class StubRuntimeFamily:
    """Runtime hooks matching the generator's deliberately minimal fixture tasks."""

    agent_system = "Use the local shopping tools and complete the task."
    metric_name = "stub_metric"

    def __init__(self, name: str) -> None:
        self.name = name

    def configure_environment(self, env, task: TaskSpec) -> None:  # noqa: ANN001
        return None

    def on_write(self, env, ref) -> bool:  # noqa: ANN001
        return False

    def user_sim_context(self, task: TaskSpec) -> None:
        return None

    def user_sim_allow_pushback(self, task: TaskSpec) -> bool:
        return False

    def verify_extra(
        self,
        task,  # noqa: ANN001
        ledger,  # noqa: ANN001
        catalog,  # noqa: ANN001
        checks,  # noqa: ANN001
        order,  # noqa: ANN001
        order_seq,  # noqa: ANN001
        *,
        observed=None,  # noqa: ANN001
    ) -> dict:
        correct = all(
            checks.get(name) is True
            for name in (
                "final_in_gold",
                "final_in_stock",
                "within_budget",
                "no_illegal_side_effects",
            )
        )
        return {
            "construct_success": correct,
            "family_metric": 1.0 if correct else 0.0 if order is not None else None,
        }


def _migrate_to_sealed_grading(epoch: Path, manifest: dict) -> None:
    """Upgrade the test-owned fixture copy when the installed runtime uses v6."""

    from oro_env_runtime.grading import GRADING_SCHEMA_VERSION
    from oro_env_runtime.pack import COMPILED_EPOCH_VERSION, fingerprint

    manifest["pack_version"] = COMPILED_EPOCH_VERSION
    task_path = epoch / "data/tasks/private_tasks.jsonl"
    task_rows = [json.loads(line) for line in task_path.read_text().splitlines()]
    for row in task_rows:
        task = row["task"]
        family = task["family"]
        task.setdefault("family_payload", {})["world"] = {"regime": "qualifying"}
        keys = list(task["acceptance"]["acceptable_keys"])
        grading = {
            "schema_version": GRADING_SCHEMA_VERSION,
            "family": family,
            "gold_keys": keys,
            "hard_satisfied": {key: True for key in keys},
            "preference_scores": {key: 1.0 for key in keys},
            "event_commitment_keys": keys if task.get("event_rule") else None,
            "uses_judge": family in {"preference_reasoning", "justification"},
            "decoupled": True,
        }
        grading.update(
            {
                "retrieval_recall": {
                    "retrieval": {
                        "eligible_pool": keys,
                        "relevance_clusters": [[key] for key in keys],
                        "required_observed_clusters": min(1, len(keys)),
                    }
                },
                "ranking": {
                    "ranking": {
                        "candidate_scores": {key: 1.0 for key in keys},
                        "pre_event_top_keys": keys,
                        "post_event_top_keys": keys,
                        "post_event_candidate_scores": {key: 1.0 for key in keys},
                    }
                },
                "recovery": {
                    "recovery": {
                        "relaxation_level": {key: 0 for key in keys},
                    }
                },
            }.get(family, {})
        )
        task["grading"] = grading
        row["task"] = TaskSpec.model_validate(task).model_dump(mode="json")
    task_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in task_rows
        )
    )
    manifest["epoch"]["task_set_fingerprint"] = fingerprint(
        [
            {
                "task_id": row["task_id"],
                "task_fingerprint": fingerprint(row["task"]),
            }
            for row in task_rows
        ]
    )


@pytest.fixture
def compiled_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Extract the trusted compatibility archive without importing the generator."""

    lookup = lambda name: StubRuntimeFamily(name)  # noqa: E731
    monkeypatch.setattr("oro_env_runtime.families.get_family", lookup)
    monkeypatch.setattr(runtime_verify, "get_family", lookup, raising=False)
    monkeypatch.setattr("validator.session_registry.get_family", lookup)
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
    manifest["contracts"]["tools"] = contracts.TOOL_CONTRACT_VERSION
    manifest["contracts"]["verifier"] = contracts.VERIFIER_VERSION
    if "grading" in TaskSpec.model_fields:
        _migrate_to_sealed_grading(epoch, manifest)
    gate_path = epoch / "tf4_hybrid_release_gate.json"
    shutil.copy2(TF4_RELEASE_GATE_PATH, gate_path)
    gate = json.loads(gate_path.read_text())
    tf4_contract = judge_contract(manifest["models"]["judge"])
    tf4_reward = manifest["reward"]["preference_reasoning"]
    tf4_reward["release_gate_fingerprint"] = tf4_release_gate_fingerprint(gate)
    tf4_reward["judge_contract"] = tf4_contract
    gate_active = tf4_release_gate_passed(gate_path, judge_contract=tf4_contract)
    tf4_reward["active"] = gate_active
    tf4_reward["status"] = "active" if gate_active else "shadow"
    (epoch / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    write_checksums(epoch)
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
    monkeypatch.setattr("oro_env_runtime.validation.SearchServerClient", client)
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


__all__ = ["COMPAT_ARCHIVE", "COMPAT_METADATA"]
