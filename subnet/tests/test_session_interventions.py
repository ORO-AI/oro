"""Adapter parity: real TaskSession actions, deterministic sealed shopper turns.

These compatibility-fixture checks are transport regressions, not a substitute
for the release pack's public-solution and verifier audit.
"""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from oro_env_runtime.schema import Event, InterventionRule
from oro_env_runtime.catalog import Catalog
from oro_env_runtime.environment import Environment
from oro_env_runtime.runtime import TOOL_CONTRACT_VERSION, TaskSession
from oro_env_runtime.user_sim import UserSim
from validator.session_registry import SessionRegistry

pytest_plugins = ("tests.compat_fixture",)


def _rule(turn=3, *, trigger=None, budget=50):
    return InterventionRule(
        trigger=trigger or {"kind": "solver_turn", "turn": turn},
        action="add_constraint",
        public_delta={"budget": budget},
        expected_effect={"private_canary": "never disclose"},
        utterance=f"My budget is now {budget}. Please confirm before ordering.",
    )


class _Simulator:
    ensure_intervention = UserSim.ensure_intervention

    def __init__(self):
        self._budget_cap = 1000
        self.calls = 0

    async def respond(self, transcript, signal):
        self.calls += 1
        self.last_signal = signal
        return {"action": "no_op", "content": "", "reason": "test"}

    def ensure_react(self, decision, signal):
        return {
            "action": "react_to_event",
            "content": "Check the change.",
            "reason": "test",
        }


@contextmanager
def _registry(pack, rules):
    from tests.compat_fixture import COMPAT_PRODUCTS

    sim = _Simulator()
    # Build a transport-unit session from the compatibility fixture's separately
    # SHA-verified catalog. The current portable archive excludes catalog rows.
    # This is not sealed release-pack admission.
    session = TaskSession.__new__(TaskSession)
    session.task = pack.task_specs[0].model_copy(update={"interventions": rules})
    session.task_id = pack.task_ids[0]
    session.catalog = Catalog.load(COMPAT_PRODUCTS)
    session.search = None
    session.env = Environment(session.task, session.catalog)
    session.state_blind = False
    session.max_steps = 30
    session._steps = session._solver_turns = 0
    session._run_started = session._run_finished = False
    session._judge_lock = asyncio.Lock()
    session._judged_result = None
    session._render_budget = None
    transport_pack = SimpleNamespace(
        pack_sha256=pack.pack_sha256,
        open_session=lambda *args, **kwargs: session,
        close=lambda: None,
    )
    with SessionRegistry(
        transport_pack, simulator_factory=lambda session: sim
    ) as registry:
        registry.start(
            evaluation_run_id="audit",
            agent_version_id="policy",
            task_id=pack.task_ids[0],
            session_id="session",
        )
        state = registry._sessions["session"]
        yield registry, state, sim


def _call(registry, turn, actions=None):
    actions = actions or [{"name": "inspect_cart", "args": {}}]
    envelope = {
        "session_id": "session",
        "tool_contract_version": TOOL_CONTRACT_VERSION,
        "turn": turn,
        "call_id": f"turn-{turn}",
        "idempotency_key": f"turn-{turn}",
        "calls": [
            {"call_id": f"turn-{turn}-{i}", "action": action}
            for i, action in enumerate(actions)
        ],
    }
    return registry.call(envelope), envelope


@pytest.mark.parametrize("turn", [3, 4, 5])
def test_due_constraint_surfaces_at_end_of_trigger_turn_once(loaded_pack, turn):
    rule = _rule(turn)
    with _registry(loaded_pack, [rule]) as (registry, state, sim):
        for step in range(1, turn):
            response, _ = _call(registry, step)
            assert response["user_message"] is None
        response, envelope = _call(registry, turn)
        assert response["user_message"] == {"content": rule.utterance}
        assert "private_canary" not in str(response)
        assert sim.calls == 0
        assert sim._budget_cap == 50
        assert state.delivered_interventions == {0}
        assert registry.call(envelope)["replayed"] is True
        assert (
            len(
                [
                    e
                    for e in state.session.env.ledger.entries()
                    if e.kind == "user_message"
                ]
            )
            == 1
        )
        assert state.session.env.ledger.last().payload["env_signal"] == {
            "kind": "intervention",
            "index": 0,
            "action": "add_constraint",
        }
        assert _call(registry, turn + 1)[0]["user_message"] is None


def test_parallel_actions_do_not_advance_intervention_clock(loaded_pack):
    with _registry(loaded_pack, [_rule(3)]) as (registry, state, sim):
        actions = [{"name": "inspect_cart", "args": {}}] * 4
        response, _ = _call(registry, 1, actions)
        assert response["user_message"] is None
        assert response["action_count"] == 4
        assert response["solver_turn_count"] == 1
        assert _call(registry, 2)[0]["user_message"] is None
        assert _call(registry, 3)[0]["user_message"] is not None


def test_simultaneously_due_rules_preserve_task_order(loaded_pack):
    rules = [_rule(3, budget=75), _rule(3, budget=50)]
    with _registry(loaded_pack, rules) as (registry, state, sim):
        _call(registry, 1)
        _call(registry, 2)
        assert _call(registry, 3)[0]["user_message"]["content"] == "\n".join(
            rule.utterance for rule in rules
        )
        assert _call(registry, 4)[0]["user_message"] is None
        assert state.delivered_interventions == {0, 1}
        assert sim._budget_cap == 50


def test_event_notice_and_due_intervention_share_boundary(loaded_pack):
    with _registry(loaded_pack, [_rule(2)]) as (registry, state, sim):
        _call(registry, 1)
        state.session.env.applied_events.append(
            Event(
                kind="stockout",
                target=state.session.task.gold_set[0],
                currency=state.session.task.hard.currency,
            )
        )
        state.event_fired_turn = 1
        assert _call(registry, 2)[0]["user_message"] == {
            "content": "Check the change.\n" + _rule().utterance
        }
        assert state.delivered_interventions == {0}
        assert _call(registry, 3)[0]["user_message"] is None


def test_after_event_intervention_waits_one_turn(loaded_pack):
    rule = _rule(trigger={"kind": "after_event"})
    with _registry(loaded_pack, [rule]) as (registry, state, sim):
        _call(registry, 1)
        state.session.env.applied_events.append(
            Event(
                kind="stockout",
                target=state.session.task.gold_set[0],
                currency=state.session.task.hard.currency,
            )
        )
        state.event_fired_turn = 1
        assert _call(registry, 2)[0]["user_message"] == {"content": "Check the change."}
        assert state.event_surfaced_turn == 2
        assert _call(registry, 3)[0]["user_message"] == {"content": rule.utterance}


def test_price_notice_preserves_event_currency(loaded_pack):
    with _registry(loaded_pack, []) as (registry, state, sim):
        _call(registry, 1)
        state.session.env.applied_events.append(
            Event(
                kind="price_change",
                target=state.session.task.gold_set[0],
                old_price=100,
                new_price=130,
                currency=state.session.task.hard.currency,
            )
        )
        state.event_fired_turn = 1
        _call(registry, 2)
        assert sim.last_signal == {
            "kind": "price_change",
            "old_price": 100,
            "new_price": 130,
            "currency": state.session.task.hard.currency,
        }


@pytest.mark.parametrize("trigger_turn", [1, 3])
def test_same_group_terminal_gets_no_retroactive_constraint(loaded_pack, trigger_turn):
    with _registry(loaded_pack, [_rule(trigger_turn)]) as (registry, state, sim):
        candidate = state.session.task.gold_set[0]
        response, envelope = _call(
            registry,
            1,
            [
                {
                    "name": "add_to_cart",
                    "args": {"product_id": candidate.product_id, "sku": candidate.sku},
                },
                {"name": "place_test_order", "args": {}},
                {"name": "inspect_cart", "args": {}},
            ],
        )
        assert response["calls"][1]["observation"]["done"] is True
        assert response["calls"][2]["observation"]["error"] == "skipped_after_terminal"
        assert response["user_message"] is None
        assert not state.delivered_interventions
        assert sim.calls == 0
        assert registry.call(envelope)["replayed"] is True


def test_cart_trigger_uses_successful_add_count(loaded_pack):
    rule = _rule(trigger={"kind": "cart_add_count", "count": 1})
    with _registry(loaded_pack, [rule]) as (registry, state, sim):
        candidate = state.session.task.gold_set[0]
        response, _ = _call(
            registry,
            1,
            [
                {
                    "name": "add_to_cart",
                    "args": {"product_id": candidate.product_id, "sku": candidate.sku},
                }
            ],
        )
        assert response["user_message"] == {"content": rule.utterance}
        assert sim.calls == 0
