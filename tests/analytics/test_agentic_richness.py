"""Unit tests for the agentic_richness calculator (ORO-1372).

Coverage:
  1. Axis-A: all dispatches valid -> 1.0
  2. Axis-B: all dispatches missing nonce_status -> 0.0, has_nonce_stamps=True
  3. Mixed: 1 valid + 1 missing -> 0.5
  4. Pre-enforcement: no nonce_status field anywhere -> agentic_richness=None
  5. Empty bundle -> agentic_richness=None, total_dispatch_count=0
  6. Non-catalogue paths ignored (inference, /search/get_product_raw)
  7. Mismatched/expired/replayed counted in total but NOT in valid
  8. Tool breakdown reflects only catalogue dispatches
"""

from __future__ import annotations

from src.analytics.agentic_richness import calc_agentic_richness


def _dispatch(path: str, nonce_status: str | None = ...) -> dict:
    """Build a proxy_calls summary entry. Use nonce_status=None to omit the field."""
    call: dict = {"kind": "summary", "path": path}
    if nonce_status is not ...:
        call["nonce_status"] = nonce_status
    return call


def _step(*calls) -> dict:
    return {"extra_info": {"proxy_calls": list(calls)}}


# -----------------------------------------------------------------------
# 1. Axis-A: all valid
# -----------------------------------------------------------------------
def test_all_valid_returns_1_0() -> None:
    bundle = [
        _step(
            _dispatch("/search/find_product", "valid"),
            _dispatch("/search/view_product_information", "valid"),
        )
    ]
    result = calc_agentic_richness(bundle)
    assert result.agentic_richness == 1.0
    assert result.valid_count == 2
    assert result.total_dispatch_count == 2
    assert result.has_nonce_stamps is True


# -----------------------------------------------------------------------
# 2. Axis-B: nonce_status present but "missing" (not valid)
# -----------------------------------------------------------------------
def test_all_missing_returns_0_0() -> None:
    bundle = [
        _step(
            _dispatch("/search/find_product", "missing"),
            _dispatch("/search/find_product", "missing"),
        )
    ]
    result = calc_agentic_richness(bundle)
    assert result.agentic_richness == 0.0
    assert result.valid_count == 0
    assert result.total_dispatch_count == 2
    assert result.has_nonce_stamps is True


# -----------------------------------------------------------------------
# 3. Mixed: 1 valid + 1 missing
# -----------------------------------------------------------------------
def test_mixed_returns_0_5() -> None:
    bundle = [
        _step(
            _dispatch("/search/find_product", "valid"),
            _dispatch("/search/find_product", "missing"),
        )
    ]
    result = calc_agentic_richness(bundle)
    assert abs(result.agentic_richness - 0.5) < 1e-9
    assert result.valid_count == 1
    assert result.total_dispatch_count == 2


# -----------------------------------------------------------------------
# 4. Pre-enforcement: no nonce_status field at all
# -----------------------------------------------------------------------
def test_pre_enforcement_returns_none() -> None:
    bundle = [
        _step(
            _dispatch("/search/find_product"),   # no nonce_status key
            _dispatch("/search/find_product"),
        )
    ]
    result = calc_agentic_richness(bundle)
    assert result.agentic_richness is None
    assert result.has_nonce_stamps is False
    assert result.total_dispatch_count == 2


# -----------------------------------------------------------------------
# 5. Empty bundle
# -----------------------------------------------------------------------
def test_empty_bundle_returns_none() -> None:
    result = calc_agentic_richness([])
    assert result.agentic_richness is None
    assert result.total_dispatch_count == 0
    assert result.n_steps == 0
    assert result.has_nonce_stamps is False


# -----------------------------------------------------------------------
# 6. Non-catalogue paths are ignored
# -----------------------------------------------------------------------
def test_non_catalogue_paths_ignored() -> None:
    bundle = [
        _step(
            # inference path — not a catalogue dispatch
            {"kind": "summary", "path": "/inference/chat/completions", "nonce_status": "valid"},
            # unknown /search/* endpoint
            _dispatch("/search/get_product_raw", "valid"),
            # the only real catalogue call
            _dispatch("/search/find_product", "valid"),
        )
    ]
    result = calc_agentic_richness(bundle)
    assert result.total_dispatch_count == 1
    assert result.valid_count == 1
    assert result.agentic_richness == 1.0


# -----------------------------------------------------------------------
# 7. Mismatch / expired / replayed counted in total but not valid
# -----------------------------------------------------------------------
def test_non_valid_statuses_not_credited() -> None:
    bundle = [
        _step(
            _dispatch("/search/find_product", "mismatch"),
            _dispatch("/search/find_product", "expired"),
            _dispatch("/search/find_product", "replayed"),
            _dispatch("/search/find_product", "valid"),
        )
    ]
    result = calc_agentic_richness(bundle)
    assert result.total_dispatch_count == 4
    assert result.valid_count == 1
    assert abs(result.agentic_richness - 0.25) < 1e-9


# -----------------------------------------------------------------------
# 8. Tool breakdown reflects only catalogue dispatches
# -----------------------------------------------------------------------
def test_tool_breakdown_counts_catalogue_only() -> None:
    bundle = [
        _step(
            _dispatch("/search/find_product", "valid"),
            _dispatch("/search/find_product", "valid"),
            _dispatch("/search/view_product_information", "missing"),
            {"kind": "summary", "path": "/inference/chat/completions"},  # not counted
        )
    ]
    result = calc_agentic_richness(bundle)
    assert result.tool_breakdown == {"find_product": 2, "view_product_information": 1}
    assert result.total_dispatch_count == 3
