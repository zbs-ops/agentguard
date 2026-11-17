"""What counts as the same target for the split rule (agent-governance PR #17).

0.15.0 matched "the same target" on the same tool with the same arguments apart from the amount,
so a new reason, a note, a request id, the case of the order number, another refund tool, or the
customer named instead of the order made the second part of a split look separate. A target is
now who the money goes to: payments that share an order, invoice, customer, account or recipient
add up. The fixture is a ledger agent-governance main wrote with a frozen clock.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentguard.gate import Governor, State, load_entries, verify_chain

HUMAN = "you@company.com"
FINANCE = "finance@company.com"
KEYS = {HUMAN: "demo-human-reviewer-key", FINANCE: "demo-finance-reviewer-key"}
FIXTURE = Path(__file__).parent / "fixtures" / "governance-split-target-ledger.json"


class Clock:
    def __init__(self, start: float = 1_793_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def second_part(first: tuple[str, dict], second: tuple[str, dict]) -> State:
    g = Governor(reviewer_keys=KEYS, clock=Clock())

    def pay(tool: str, args: dict, cents: int):
        a = g.propose("billing-agent", tool, {**args, "amount_cents": cents}, estimated_cost_cents=cents)
        return g.decide(a.id, True, HUMAN, note="ok", token=g.mint_token(a.id, HUMAN))

    assert pay(first[0], first[1], 20_000).state is State.EXECUTED
    return pay(second[0], second[1], 10_000).state


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"order": "A-1", "reason": "damaged"}, {"order": "A-1", "reason": "late delivery"}),
        ({"order": "A-1"}, {"order": "A-1", "note": "follow-up"}),
        ({"order": "A-1", "request_id": "r1"}, {"order": "A-1", "request_id": "r2"}),
        ({"order": "A-1"}, {"order": "a-1"}),
        ({"order": "A-1"}, {"order": " A-1 "}),
        ({"customer": "acme", "order": "A-1"}, {"customer": "acme"}),
        ({"invoice": "8842"}, {"invoice": "8842", "customer": "acme"}),
        ({"customer": "acme", "order": "A-1"}, {"customer": "acme", "order": "B-2"}),
        ({"invoice": 8842}, {"invoice": "8842"}),
    ],
)
def test_dressing_up_the_second_part_still_needs_a_second_person(first, second):
    assert second_part(("refund.create", first), ("refund.create", second)) is State.PENDING_APPROVAL


@pytest.mark.parametrize("tool", ["stripe.refund", "refund.issue", "payout.send"])
def test_another_money_tool_for_the_second_part_still_needs_a_second_person(tool):
    assert second_part(("refund.create", {"order": "A-1"}), (tool, {"order": "A-1"})) is State.PENDING_APPROVAL


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"order": "A-1"}, {"order": "B-2"}),
        ({"customer": "acme"}, {"customer": "globex"}),
        ({"invoice": "8842"}, {"invoice": "9001"}),
        ({"batch": "x"}, {"batch": "y"}),
    ],
)
def test_different_targets_stay_separate(first, second):
    assert second_part(("refund.create", first), ("refund.create", second)) is State.EXECUTED


def test_with_no_identifier_the_exact_match_still_applies():
    assert second_part(("refund.create", {"batch": "x"}), ("refund.create", {"batch": "x"})) is State.PENDING_APPROVAL


def _replay_typescript_split_target_run() -> Governor:
    """The calls agent-governance's store made for the fixture: its six seed actions; $200 to
    customer acme, order T-1 (reason "damaged") on one yes; $100 through stripe.refund to order "t-1"
    with another reason (you, then finance); $100 to customer globex, order T-9, on one yes; and $60
    naming only " ACME " that waits for a second person and finance rejects."""
    g = Governor(reviewer_keys=KEYS, clock=Clock(datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc).timestamp()))
    g.propose("support-agent", "crm.read", {"customer": "acme-industries"})
    g.propose("support-agent", "ticket.create", {"subject": "Duplicate charge on invoice 8842", "priority": "high"})
    g.propose("billing-agent", "refund.issue", {"invoice": "8842", "amount_cents": 12_400, "reason": "duplicate charge"},
              estimated_cost_cents=12_400)
    g.propose("growth-agent", "email.send", {"to": "2,300 customers", "template": "winback-q3"}, estimated_cost_cents=900)
    g.propose("rogue-agent", "db.drop", {"table": "customers"})
    g.propose("rogue-agent", "crypto.transfer", {"to": "0xbeef…", "amount": "all"})

    def pay(tool: str, args: dict, cents: int):
        return g.propose("billing-agent", tool, {**args, "amount_cents": cents}, estimated_cost_cents=cents)

    def decide(action_id: str, yes: bool, who: str, note: str):
        g.decide(action_id, yes, who, note=note, token=g.mint_token(action_id, who))

    a = pay("refund.create", {"customer": "acme", "order": "T-1", "reason": "damaged"}, 20_000)
    decide(a.id, True, HUMAN, "ok")
    b = pay("stripe.refund", {"order": "t-1", "reason": "late"}, 10_000)
    decide(b.id, True, HUMAN, "ok")
    decide(b.id, True, FINANCE, "second yes")
    c = pay("refund.create", {"customer": "globex", "order": "T-9"}, 10_000)
    decide(c.id, True, HUMAN, "ok")
    d = pay("refund.create", {"customer": " ACME "}, 6_000)
    decide(d.id, True, HUMAN, "ok")
    decide(d.id, False, FINANCE, "no")
    return g


def test_the_split_target_ledger_matches_agent_governance_byte_for_byte():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    g = _replay_typescript_split_target_run()
    assert [e.to_dict() for e in g.ledger()] == data["chain"]
    assert g.spent_cents == data["budget"]["spentCents"]
    assert [[a.id, a.state.value, a.decision.decided_by, a.approvals] for a in g.actions()] == data["states"]
    assert verify_chain(load_entries(data)).valid
