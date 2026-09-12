"""Deterministic acceptance-membership event-oracle regression.

Zero-inference black-box probe: fresh sealed sessions, add one visible candidate
each through the same public ``add_to_cart`` path a miner sees, then observe
miner-visible response fields for a validator-known accepted and rejected
candidate. The runtime binds each event to its public, compiler-selected target
instead of ``task.acceptance.acceptable_keys``. This probe commits an accepted
and a rejected candidate in otherwise fresh sessions and proves that both
follow the same public event path. Acceptance membership therefore
cannot be inferred from whether the event fires, changes the overlay, or
produces a delayed shopper message.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from oro_env_runtime.schema import CandidateRef, EventRule
from oro_env_runtime.runtime import TOOL_CONTRACT_VERSION
from validator.env_pack_loader import LoadedPack
from validator.session_registry import SessionRegistry

pytest_plugins = ("tests.compat_fixture",)


_ACCEPTED_SESSION = "oracle-session-accepted"
_REJECTED_SESSION = "oracle-session-rejected"
_EVIDENCE_ENV = "ORO_1868_EVIDENCE_PATH"


class _OracleFamily:
    """Family stub mirroring real acceptance-triggered ``on_write`` behavior."""

    agent_system = "acceptance-event baseline probe"
    metric_name = "oracle"

    def __init__(self, name: str) -> None:
        self.name = name

    def configure_environment(self, env, task) -> None:
        return None

    def user_sim_context(self, task):
        return None

    def user_sim_allow_pushback(self, task) -> bool:
        return False

    def on_write(self, env, ref) -> bool:
        if env.applied_event is None and ref.key() in env._event_trigger_keys:
            env.phase = "committed"
            env._apply_event(ref)
            return True
        return False

    def event_targets(self, _task, _catalog, ref) -> list[str]:
        return [ref.key()]

    def verify_extra(self, *_a, **_kw) -> dict:
        return {"construct_success": False, "family_metric": 0.0}


class _StubSimulator:
    """Deterministic simulator; only invoked when SessionRegistry event_ready fires."""

    def __init__(self) -> None:
        self.signals: list[dict | None] = []

    async def respond(self, _transcript, signal):
        self.signals.append(signal)
        content = "shopper-noted-event" if signal else "shopper-neutral"
        return {"content": content, "action": "continue", "reason": "oracle-stub"}

    def ensure_react(self, decision, _signal):
        return decision


def _install_oracle_family(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _OracleFamily
    monkeypatch.setattr("oro_env_runtime.families.get_family", factory)
    monkeypatch.setattr("oro_env_runtime.verify.get_family", factory)
    monkeypatch.setattr("validator.session_registry.get_family", factory)


def _envelope(session_id: str, call_id: str, turn: int, action: dict) -> dict:
    return {
        "session_id": session_id,
        "tool_contract_version": TOOL_CONTRACT_VERSION,
        "call_id": call_id,
        "idempotency_key": call_id,
        "turn": turn,
        "action": action,
    }


def _run_probe(pack: LoadedPack, candidate: CandidateRef, session_id: str) -> dict:
    simulator = _StubSimulator()
    with SessionRegistry(pack, simulator_factory=lambda _s: simulator) as registry:
        bootstrap = registry.start(
            evaluation_run_id="acceptance-event-baseline",
            agent_version_id="acceptance-event-baseline-policy",
            task_id=pack.task_ids[0],
            session_id=session_id,
        )
        session = registry._sessions[session_id].session
        session.env.task.event_rule = EventRule(kind="stockout")
        turn_one = registry.call(
            _envelope(
                session_id,
                call_id="add-1",
                turn=1,
                action={
                    "name": "add_to_cart",
                    "args": {
                        "product_id": candidate.product_id,
                        "sku": candidate.sku,
                    },
                },
            )
        )
        turn_two = registry.call(
            _envelope(
                session_id,
                call_id="inspect-1",
                turn=2,
                action={"name": "inspect_cart", "args": {}},
            )
        )
    return {
        "bootstrap": bootstrap,
        "turn_one": turn_one,
        "turn_two": turn_two,
        "simulator_signals": simulator.signals,
    }


def _resolve_git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _write_evidence(payload: dict) -> Path | None:
    destination = os.environ.get(_EVIDENCE_ENV)
    if not destination:
        return None
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_public_event_path_does_not_depend_on_acceptance_membership(
    loaded_pack: LoadedPack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted and rejected public event targets follow the same protocol."""

    _install_oracle_family(monkeypatch)
    task = loaded_pack.task_specs[0]
    accepted_key = task.acceptance.acceptable_keys[0]
    accepted_product, accepted_sku = accepted_key.split("::", 1)
    accepted = CandidateRef(product_id=accepted_product, sku=accepted_sku)

    probe_session = loaded_pack.open_session(loaded_pack.task_ids[0])
    accepted_set = set(task.acceptance.acceptable_keys)
    rejected = next(
        ref
        for ref in probe_session.catalog.purchasable(
            max_price=task.hard.budget,
            limit=150,
        )
        if ref.key() not in accepted_set
    )

    accepted_out = _run_probe(loaded_pack, accepted, _ACCEPTED_SESSION)
    rejected_out = _run_probe(loaded_pack, rejected, _REJECTED_SESSION)

    accepted_msg = accepted_out["turn_two"]["user_message"]
    rejected_msg = rejected_out["turn_two"]["user_message"]
    accepted_added = accepted_out["turn_one"]["observation"]["observation"]["added"]
    rejected_added = rejected_out["turn_one"]["observation"]["observation"]["added"]

    channels = {
        "same_next_turn_user_message": accepted_msg == rejected_msg,
        "both_cart_overlays_stockout": accepted_added.get("in_stock") is False
        and rejected_added.get("in_stock") is False,
        "simulator_signal_present_for_both": all(
            any(signal and signal.get("kind") == "stockout" for signal in signals)
            for signals in (
                accepted_out["simulator_signals"],
                rejected_out["simulator_signals"],
            )
        ),
    }
    membership_independent = all(channels.values())

    evidence = {
        "schema_version": "oro.acceptance_event_oracle_regression.v2",
        "test_case": "acceptance-membership-event-oracle",
        "commit": _resolve_git_commit(),
        "pack_sha256": loaded_pack.pack_sha256,
        "task_id": loaded_pack.task_ids[0],
        "task_family_recorded": task.family,
        "event_rule_injected": {"kind": "stockout"},
        "candidates": {
            "accepted": {
                "product_id": accepted.product_id,
                "sku": accepted.sku,
                "acceptance_membership": True,
            },
            "rejected": {
                "product_id": rejected.product_id,
                "sku": rejected.sku,
                "acceptance_membership": False,
            },
        },
        "miner_visible_observations": {
            "accepted": {
                "turn_one_added": accepted_added,
                "turn_two_user_message": accepted_msg,
                "simulator_signals_seen_by_registry": accepted_out[
                    "simulator_signals"
                ],
            },
            "rejected": {
                "turn_one_added": rejected_added,
                "turn_two_user_message": rejected_msg,
                "simulator_signals_seen_by_registry": rejected_out[
                    "simulator_signals"
                ],
            },
        },
        "membership_independence_checks": channels,
        "vulnerable_baseline": not membership_independent,
        "no_hidden_state_returned_directly": all(
            "acceptable_keys" not in json.dumps(out["bootstrap"])
            and "acceptance" not in json.dumps(out["bootstrap"])
            for out in (accepted_out, rejected_out)
        ),
    }

    hash_payload = json.dumps(
        {k: v for k, v in evidence.items() if k != "commit"},
        sort_keys=True,
    )
    evidence["evidence_fingerprint"] = hashlib.sha256(
        hash_payload.encode()
    ).hexdigest()
    evidence_path = _write_evidence(evidence)
    if evidence_path is not None:
        print(f"[acceptance-event] wrote regression evidence to {evidence_path}")

    assert channels["same_next_turn_user_message"], (
        "shopper push behavior changed with private acceptance membership"
    )
    assert channels["both_cart_overlays_stockout"], (
        "the public event target was not stocked out consistently"
    )
    assert channels["simulator_signal_present_for_both"], (
        "simulator event signaling changed with private acceptance membership"
    )
    assert membership_independent
