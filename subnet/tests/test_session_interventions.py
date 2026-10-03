"""Adapter parity: real TaskSession actions, deterministic sealed shopper turns.

These compatibility-fixture checks are transport regressions, not a substitute
for the release pack's public-solution and verifier audit.
"""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


from oro_env_runtime import loop
from oro_env_runtime.schema import Event
from oro_env_runtime.situation import (
    AfterTransition,
    Budget,
    Commitment,
    Facet,
    MarketChange,
    Obligation,
    OnQuestion,
    Requirement,
    RequirementChange,
    Situation,
    SolverTurn,
    Transition,
    Utterance,
)
from oro_env_runtime.observations import event_observed_or_signaled
from oro_env_runtime.situation_eval import evaluate
from oro_env_runtime.catalog import CandidateMeta
from oro_env_runtime.environment import Environment
from oro_env_runtime.runtime import TaskSession
from oro_env_runtime.schema import CandidateRef
from oro_env_runtime.user_sim import UserSim
from validator.session_registry import SessionRegistry

from tests.compat_fixture import accepted_ref

pytest_plugins = ("tests.compat_fixture",)


class _Simulator:
    def __init__(self):
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


class _FixtureCatalog:
    """The runtime's catalog interface over the compatibility fixture's products."""

    currency = "USD"

    def __init__(self, path):
        with open(path, encoding="utf-8") as rows:
            self.products = {r["product_id"]: r for r in map(json.loads, rows)}

    def contains_product(self, product_id):
        return product_id in self.products

    def variants_of(self, product_id):
        p = self.products[product_id]
        return [
            CandidateMeta(
                ref=CandidateRef(product_id=product_id, sku=v["sku"]),
                price=v["price"],
                in_stock=v["in_stock"],
                brand=p.get("brand"),
                category_path=tuple(p["category_path"]),
                options=v.get("options"),
                currency=p["pricing"]["currency"],
            )
            for v in p["variants"]
        ]

    def meta(self, ref):
        return next(m for m in self.variants_of(ref.product_id) if m.ref == ref)

    def price(self, ref):
        return self.meta(ref).price

    def exists(self, ref):
        return self.contains_product(ref.product_id) and any(
            m.ref == ref for m in self.variants_of(ref.product_id)
        )





@contextmanager
def _registry(pack, situation=None, make_sim=lambda task: _Simulator()):
    from tests.compat_fixture import COMPAT_PRODUCTS

    # Build a transport-unit session from the compatibility fixture's separately
    # SHA-verified catalog. The current portable archive excludes catalog rows.
    # This is not sealed release-pack admission.
    session = TaskSession.__new__(TaskSession)
    session.task = pack.task_specs[0].model_copy(update={"situation": situation})
    sim = make_sim(session.task)
    session.task_id = pack.task_ids[0]
    session.catalog = _FixtureCatalog(COMPAT_PRODUCTS)
    session.search = None
    session.env = Environment(session.task, session.catalog)
    session.state_blind = False
    session.max_steps = 30
    session._steps = session._solver_turns = 0
    session._run_started = session._run_finished = False
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
        "turn": turn,
        "call_id": f"turn-{turn}",
        "idempotency_key": f"turn-{turn}",
        "calls": [
            {"call_id": f"turn-{turn}-{i}", "action": action}
            for i, action in enumerate(actions)
        ],
    }
    return registry.call(envelope), envelope


def test_price_notice_preserves_event_currency(loaded_pack):
    with _registry(loaded_pack) as (registry, state, sim):
        _call(registry, 1)
        state.session.env.applied_events.append(
            Event(
                kind="price_change",
                target=accepted_ref(state.session.task),
                old_price=100,
                new_price=130,
                currency=state.session.task.hard.currency,
            )
        )
        state.event_fired_turns[0] = 1
        _call(registry, 2)
        assert sim.last_signal == {
            "kind": "price_change",
            "old_price": 100,
            "new_price": 130,
            "currency": state.session.task.hard.currency,
        }


class _DecisionsClient:
    """A simulator client whose decisions model reads a question about the use."""

    async def decide(self, model, state, questions):
        return {
            "model": model,
            "answers": {
                key: {"noul": 0.9 if key == "facet:use" else 0.1} for key in questions
            },
        }


def test_timeline_line_and_facet_disclosure_reach_the_ledger(loaded_pack):
    hard = loaded_pack.task_specs[0].hard
    situation = Situation(
        preset="boundary",
        requirements=[
            Requirement(
                id="budget", predicate=Budget(max=hard.budget, currency=hard.currency)
            )
        ],
        facets=[
            Facet(id="use", reveal_when="how it will be used", statement="it's for travel.")
        ],
        timeline=[
            Transition(
                id="opening",
                trigger=SolverTurn(turn=1),
                effects=[Utterance(text="i need it by friday.")],
            ),
            Transition(
                id="asked",
                trigger=OnQuestion(facet="use"),
                effects=[Utterance(text="thanks for asking.")],
            ),
        ],
    )

    def make_sim(task):
        return UserSim(
            task,
            model="simulator",
            surface_events=True,
            completion=_DecisionsClient(),
        )

    with _registry(loaded_pack, situation, make_sim) as (registry, state, sim):
        assert _call(registry, 1)[0]["user_message"] == {
            "content": "i need it by friday."
        }
        question = {"name": "message", "args": {"content": "how will you use it?"}}
        assert _call(registry, 2, [question])[0]["user_message"] == {
            "content": "it's for travel.\nthanks for asking."
        }
        assert state.session.env.disclosed_facets == {"use"}
        assert [
            entry.payload.get("env_signal")
            for entry in state.session.env.ledger.entries()
            if entry.kind == "user_message"
        ] == [
            {"kind": "transition", "id": "opening"},
            {"kind": "disclosure", "facet": "use"},
            {"kind": "transition", "id": "asked"},
        ]


def test_a_budget_cut_is_said_at_its_boundary_and_a_silent_change_never_announced(
    loaded_pack,
):
    """End to end through the session boundary: a silent reprice on the first add fires a
    budget cut in the same call group. The cut is recorded as it fires, said once at the end
    of that turn, and graded; the silent change is never announced (no notice, no shopper
    reaction), so only the agent's re-check notices it."""
    task = loaded_pack.task_specs[0]
    hard, item = task.hard, accepted_ref(task).model_dump()
    cut_line = "sorry, i can only spend half of that now."
    situation = Situation(
        preset="boundary",
        requirements=[
            Requirement(id="budget", predicate=Budget(max=hard.budget, currency=hard.currency)),
            Requirement(
                id="budget:cut",
                predicate=Budget(max=hard.budget / 2, currency=hard.currency),
                visibility="on_transition",
                active=False,
            ),
        ],
        timeline=[
            Transition(
                id="market:0",
                trigger=Commitment(n=1),
                effects=[
                    MarketChange(
                        change="price_change", price_multiplier=1.5, presentation="silent"
                    )
                ],
            ),
            Transition(
                id="cut",
                trigger=AfterTransition(transition="market:0"),
                effects=[
                    Utterance(text=cut_line),
                    RequirementChange(op="replace", requirement="budget:cut", replaces="budget"),
                ],
            ),
        ],
        obligations=[
            Obligation(id="noticed:market:0", kind="noticed_market_change", after="market:0")
        ],
    )

    def make_sim(task):
        return UserSim(
            task,
            model="simulator",
            surface_events=True,
            completion=_DecisionsClient(),
        )

    with _registry(loaded_pack, situation, make_sim) as (registry, state, sim):
        env = state.session.env
        add = {"name": "add_to_cart", "args": item}
        first, _ = _call(registry, 1, [add])
        # The silent change hides from the add; the cut it fired is said at this boundary.
        assert "market_event" not in first["calls"][0]["observation"]
        assert first["user_message"] == {"content": cut_line}
        assert env.announced_events() == []
        # Later turns say nothing more: no repeat of the line, no notice of the change.
        for turn in (2, 3, 4):
            assert _call(registry, turn)[0].get("user_message") is None
        recheck = {"name": "inspect_stock", "args": item}
        _call(registry, 5, [recheck])
        _call(registry, 6, [{"name": "place_test_order", "args": item}])
        ledger = env.ledger.entries()

    said = [e for e in ledger if e.kind == "user_message"]
    assert [e.payload.get("env_signal") for e in said] == [
        {"kind": "transition", "id": "cut"}
    ]
    assert all(e.payload.get("action") != "react_to_event" for e in said)
    # Recorded when it fired (the add's turn), before the line was said.
    fired = next(e for e in ledger if e.kind == "verifier_signal" and "env_signal" in e.payload)
    assert fired.payload["env_signal"] == {"kind": "transition", "id": "cut"}
    assert fired.seq < said[0].seq and fired.turn == said[0].turn == 1
    graded = evaluate(situation, ledger, {})
    assert graded.fired == ("market:0", "cut") and "budget:cut" in graded.active
    # The re-check is the agent's evidence it saw the change; the shopper never told it.
    assert event_observed_or_signaled(ledger, None, index=0, agent=True)
    # Ordering the item the change priced out fails the cut and the notice alike.
    assert graded.checks["requirement:budget:cut"] is False
    assert graded.checks["obligation:noticed:market:0"] is False


def test_a_turn_3_cut_is_heard_before_an_order_on_turn_3(loaded_pack):
    """As loop.run says it: a cut due at turn 3's boundary is said at the end of turn 2's
    response, so an order on turn 3 is graded against a cut the agent has heard."""
    task = loaded_pack.task_specs[0]
    hard, item = task.hard, accepted_ref(task).model_dump()
    cut_line = "my budget just dropped."
    situation = Situation(
        preset="boundary",
        requirements=[
            Requirement(id="budget", predicate=Budget(max=hard.budget, currency=hard.currency)),
            Requirement(
                id="budget:cut",
                predicate=Budget(max=0.01, currency=hard.currency),
                visibility="on_transition",
                active=False,
            ),
        ],
        timeline=[
            Transition(
                id="cut",
                trigger=SolverTurn(turn=3),
                effects=[
                    Utterance(text=cut_line),
                    RequirementChange(op="replace", requirement="budget:cut", replaces="budget"),
                ],
            )
        ],
    )

    def make_sim(task):
        return UserSim(
            task, model="simulator", surface_events=True, completion=_DecisionsClient()
        )

    with _registry(loaded_pack, situation, make_sim) as (registry, state, _):
        env = state.session.env
        assert _call(registry, 1, [{"name": "add_to_cart", "args": item}])[0][
            "user_message"
        ] is None
        assert _call(registry, 2)[0]["user_message"] == {"content": cut_line}
        order = {"name": "place_test_order", "args": item}
        third, _ = _call(registry, 3, [order])
        assert third["calls"][0]["observation"]["done"] and third["user_message"] is None
        ledger = env.ledger.entries()

    fired = next(e for e in ledger if e.payload.get("env_signal") == {"kind": "transition", "id": "cut"})
    said = next(e for e in ledger if e.kind == "user_message")
    placed = next(e for e in ledger if e.kind == "verifier_signal" and "order" in e.payload)
    # Fired and said at turn 3's boundary, both before the turn-3 order, as the reference loop.
    assert fired.seq < said.seq < placed.seq
    assert fired.turn == said.turn == placed.turn == 3
    assert said.payload["content"] == cut_line
    graded = evaluate(situation, ledger, {})
    assert graded.fired == ("cut",) and graded.checks["requirement:budget:cut"] is False


@pytest.mark.parametrize(
    ("band", "inspect_turn", "noticed_events", "noticed_turns"),
    [
        ("standard", None, [0, 2], [3, 7]),
        ("hard", None, [0, 2], [4, 8]),
        # On the hard band an event the agent already saw needs no notice.
        ("hard", 2, [2], [8]),
    ],
)
def test_each_announced_market_change_gets_its_own_notice_in_order(
    loaded_pack, band, inspect_turn, noticed_events, noticed_turns
):
    """As loop.run tracks them: every announced change is noticed once, one turn after
    it fires (two on the hard band) and in firing order; a silent change in between
    never gets a notice."""
    task = loaded_pack.task_specs[0]
    hard, item = task.hard, accepted_ref(task)

    def change(turn, presentation="announced"):
        return Transition(
            id=f"market:{turn}",
            trigger=SolverTurn(turn=turn),
            effects=[
                MarketChange(
                    change="price_change",
                    price_multiplier=1.1,
                    target="keys",
                    keys=[item.key()],
                    presentation=presentation,
                )
            ],
        )

    situation = Situation(
        preset="boundary",
        requirements=[
            Requirement(id="budget", predicate=Budget(max=hard.budget, currency=hard.currency))
        ],
        timeline=[change(2), change(3, "silent"), change(6)],
    )
    with _registry(loaded_pack, situation) as (registry, state, sim):
        state.session.task = state.session.task.model_copy(update={"difficulty_band": band})
        signals = []

        async def respond(transcript, signal):
            signals.append(signal)
            return {"action": "no_op", "content": "", "reason": "test"}

        sim.respond = respond
        inspect = {"name": "inspect_stock", "args": item.model_dump()}
        for turn in range(1, 10):
            _call(registry, turn, [inspect] if turn == inspect_turn else None)
        env = state.session.env

    assert env.announced_events() == [0, 2]
    assert signals == [loop._event_signal(env.applied_events[i]) for i in noticed_events]
    noticed = [
        entry.turn
        for entry in env.ledger.entries()
        if entry.kind == "user_message" and entry.payload.get("env_signal")
    ]
    assert noticed == noticed_turns
