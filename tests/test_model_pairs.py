"""Validates docker/proxy/model_pairs.json — the cross-provider model-name
mapping used by the inference proxy. The proxy njs runtime isn't
exercisable in this Python CI, so this test focuses on the data file: it's
the most likely failure mode (typos, missing entries, drifted from the
Backend allowlist), and well-formedness is enough for the rewrite logic in
validate_model.js to behave correctly given the data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PAIRS_PATH = Path(__file__).resolve().parent.parent / "docker" / "proxy" / "model_pairs.json"


@pytest.fixture(scope="module")
def pairs() -> list[dict[str, str]]:
    doc = json.loads(PAIRS_PATH.read_text())
    return doc["pairs"]


def test_pairs_file_parses_with_pairs_array(pairs: list[dict[str, str]]) -> None:
    assert isinstance(pairs, list)
    assert len(pairs) > 0


def test_each_pair_has_both_sides_non_empty(pairs: list[dict[str, str]]) -> None:
    for p in pairs:
        assert set(p.keys()) == {"chutes", "openrouter"}, p
        assert isinstance(p["chutes"], str) and p["chutes"].strip() == p["chutes"] and p["chutes"]
        assert (
            isinstance(p["openrouter"], str)
            and p["openrouter"].strip() == p["openrouter"]
            and p["openrouter"]
        )


def test_no_duplicate_chutes_or_openrouter_ids(pairs: list[dict[str, str]]) -> None:
    chutes_ids = [p["chutes"] for p in pairs]
    openrouter_ids = [p["openrouter"] for p in pairs]
    assert len(chutes_ids) == len(set(chutes_ids))
    assert len(openrouter_ids) == len(set(openrouter_ids))


def test_chutes_and_openrouter_namespaces_dont_overlap(pairs: list[dict[str, str]]) -> None:
    """If a string appeared on both columns the rewriter would loop ambiguously.
    Chutes IDs end in `-TEE`; OpenRouter slugs are lowercase and never do."""
    chutes_ids = {p["chutes"] for p in pairs}
    openrouter_ids = {p["openrouter"] for p in pairs}
    assert chutes_ids.isdisjoint(openrouter_ids)


def test_chutes_ids_use_tee_inference_variant(pairs: list[dict[str, str]]) -> None:
    """Backend's Chutes allowlist only includes the `-TEE` inference variants
    today. Catches accidental copy-paste of bare HF repo names."""
    for p in pairs:
        assert p["chutes"].endswith("-TEE"), p


def test_each_id_has_org_slash_model_shape(pairs: list[dict[str, str]]) -> None:
    """Both Chutes and OpenRouter use `<org>/<model>`; an entry without a slash
    is almost certainly a typo."""
    for p in pairs:
        assert "/" in p["chutes"], p
        assert "/" in p["openrouter"], p
