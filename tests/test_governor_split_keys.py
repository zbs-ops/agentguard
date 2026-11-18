"""The same target written another way (agent-governance PR #18).

0.16.0 matched a target by the exact argument name, so the same order written another way looked
like a different target and the second part of a split ran on one approval: ``order`` then
``order_id``, camelCase names (``orderId``, which also fell back to the exact match so a new reason
slipped through), a capitalised name, the order inside a nested object or a list, the same id with
a look-alike hyphen, an invisible character or a full-width digit, or an email's +tag. A name is
now read by what it names, nested arguments are read too, and values are compared after Unicode
normalisation. The fixture was written by agent-governance main with a frozen clock: the kind it
reads for each tricky name, its verdict on the second part for each tricky pair, and a ledger.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentguard.gate import Governor, State, load_entries, verify_chain
from agentguard.gate.governor import target_kind

HUMAN = "you@company.com"
FINANCE = "finance@company.com"
KEYS = {HUMAN: "demo-human-reviewer-key", FINANCE: "demo-finance-reviewer-key"}
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "governance-split-keys.json").read_text(encoding="utf-8"))


class Clock:
    def __init__(self, start: float = 1_793_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def second_part(first: dict, second: dict) -> State:
    g = Governor(reviewer_keys=KEYS, clock=Clock())

    def pay(args: dict, cents: int):
        a = g.propose("billing-agent", "refund.create", {**args, "amount_cents": cents}, estimated_cost_cents=cents)
        return g.decide(a.id, True, HUMAN, note="ok", token=g.mint_token(a.id, HUMAN))

    assert pay(first, 20_000).state is State.EXECUTED
    return pay(second, 10_000).state


@pytest.mark.parametrize(("name", "kind"), FIXTURE["kinds"], ids=lambda x: repr(x))
def test_each_name_is_read_as_the_kind_agent_governance_reads(name, kind):
    assert target_kind(name) == kind


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"order": "T-1"}, {"order_id": "T-1"}),
        ({"order_id": "T-1"}, {"order": "T-1"}),
        ({"customer": "acme"}, {"customer_id": "acme"}),
        ({"order": "T-1"}, {"orderId": "T-1"}),
        ({"orderId": "T-1", "reason": "a"}, {"orderId": "T-1", "reason": "b"}),
        ({"order": "T-1"}, {"ORDER_ID": "T-1"}),
        ({"payment_intent": "pi_1"}, {"paymentIntentId": "pi_1"}),
        ({"order": "T-1"}, {"orderNumber": "T-1"}),
        ({"email": "a@b.com"}, {"customer_email": "a@b.com"}),
    ],
)
def test_the_same_target_under_another_name_still_needs_a_second_person(first, second):
    assert second_part(first, second) is State.PENDING_APPROVAL


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"order": "T-1"}, {"target": {"order": "T-1"}}),
        ({"customer": "acme"}, {"customer": {"id": "acme", "name": "Acme"}}),
        ({"order": "T-1"}, {"order": ["T-1"]}),
        ({"order": "T-1"}, {"items": [{"orderId": "T-1", "sku": "x"}]}),
        ({"order": "T-1"}, {"order": ("T-1",)}),
    ],
)
def test_the_same_target_inside_nested_arguments_still_needs_a_second_person(first, second):
    assert second_part(first, second) is State.PENDING_APPROVAL


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"order": "T-1"}, {"order": "T‑1"}),
        ({"order": "T-1"}, {"order": "T-1​"}),
        ({"invoice": "8842"}, {"invoice": "884２"}),
        ({"email": "a@b.com"}, {"email": "A@b.com", "note": "x"}),
        ({"email": "a@b.com"}, {"email": "a+refund2@b.com"}),
    ],
)
def test_the_same_id_with_look_alike_characters_still_needs_a_second_person(first, second):
    assert second_part(first, second) is State.PENDING_APPROVAL


@pytest.mark.parametrize(("first", "second", "first_state", "second_state"), FIXTURE["pairs"], ids=lambda x: repr(x))
def test_every_tricky_pair_gets_agent_governances_verdict(first, second, first_state, second_state):
    """The second part's state for each pair, as agent-governance decided it: look-alike dashes,
    invisible and full-width characters, JavaScript's trim (which keeps U+001C and U+0085, unlike
    Python's strip), case rules (final sigma, dotted I), numbers, booleans, emails and nesting."""
    assert first_state == "EXECUTED"
    assert second_part(first, second).value == second_state


def _replay_typescript_split_keys_run() -> Governor:
    """The calls agent-governance's store made for the fixture: its six seed actions; $200 to
    orderId K-1 and customer {id: acme} on one yes; $100 through stripe.refund to order_id "k‑1"
    (a look-alike hyphen) with a note (you, then finance); $100 to customer_email pay+2@globex.com
    on one yes; $60 naming customerId " ACME\\u200b " inside a list, which waits and finance
    rejects; $50 to email PAY@globex.com, the same mailbox, on one yes."""
    g = Governor(reviewer_keys=KEYS, clock=Clock(datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc).timestamp()))
    g.propose("support-agent", "crm.read", {"customer": "acme-industries"})
    g.propose("support-agent", "ticket.create", {"subject": "Duplicate charge on invoice 8842", "priority": "high"})
    g.propose(
        "billing-agent",
        "refund.issue",
        {"invoice": "8842", "amount_cents": 12_400, "reason": "duplicate charge"},
        estimated_cost_cents=12_400,
    )
    g.propose("growth-agent", "email.send", {"to": "2,300 customers", "template": "winback-q3"}, estimated_cost_cents=900)
    g.propose("rogue-agent", "db.drop", {"table": "customers"})
    g.propose("rogue-agent", "crypto.transfer", {"to": "0xbeef…", "amount": "all"})

    def pay(tool: str, args: dict, cents: int):
        return g.propose("billing-agent", tool, {**args, "amount_cents": cents}, estimated_cost_cents=cents)

    def decide(action_id: str, yes: bool, who: str, note: str):
        g.decide(action_id, yes, who, note=note, token=g.mint_token(action_id, who))

    a = pay("refund.create", {"orderId": "K-1", "customer": {"id": "acme"}}, 20_000)
    decide(a.id, True, HUMAN, "ok")
    b = pay("stripe.refund", {"order_id": "k‑1", "note": "x"}, 10_000)
    decide(b.id, True, HUMAN, "ok")
    decide(b.id, True, FINANCE, "second yes")
    c = pay("refund.create", {"customer_email": "pay+2@globex.com"}, 10_000)
    decide(c.id, True, HUMAN, "ok")
    d = pay("refund.create", {"items": [{"customerId": " ACME​ "}]}, 6_000)
    decide(d.id, True, HUMAN, "ok")
    decide(d.id, False, FINANCE, "no")
    e = pay("refund.create", {"email": "PAY@globex.com", "reason": "dup"}, 5_000)
    decide(e.id, True, HUMAN, "ok")
    return g


def test_the_split_keys_ledger_matches_agent_governance_byte_for_byte():
    g = _replay_typescript_split_keys_run()
    assert [e.to_dict() for e in g.ledger()] == FIXTURE["chain"]
    assert g.spent_cents == FIXTURE["budget"]["spentCents"]
    assert [[a.id, a.state.value, a.decision.decided_by, a.approvals] for a in g.actions()] == FIXTURE["states"]
    assert verify_chain(load_entries(FIXTURE)).valid


@pytest.mark.parametrize("depth", [30, 31, 32, 33])
def test_nesting_is_read_as_deep_as_agent_governance_reads_it(depth):
    """agent-governance reads an order nested 30 or 31 levels down and stops below that."""
    nested: dict = {"order": "T-1"}
    for _ in range(depth):
        nested = {"n": nested}
    expected = State.PENDING_APPROVAL if depth <= 31 else State.EXECUTED
    assert second_part({"order": "T-1"}, nested) is expected


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"order": "T-1"}, {"orderId": "T-2"}),
        ({"order": "42"}, {"invoice": "42"}),
        ({"order": "T-1"}, {"paid": "T-1"}),
        ({"to": "acme"}, {"photo": "acme"}),
        ({"order": "T-1"}, {"product_code": "T-1"}),
        ({"order": "T-1"}, {"meta": {"batch": "T-1"}}),
        ({"customer": "acme"}, {"product": {"id": "acme"}}),
    ],
)
def test_different_targets_stay_separate(first, second):
    assert second_part(first, second) is State.EXECUTED
