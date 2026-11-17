"""Splitting a large payment doesn't avoid the two-person rule (agent-governance PR #16).

A $300 refund needed two reviewers, but the same $300 sent to the same order as $200 then $100
needed one: each part is under $250 and they aren't repeats. Payments of the same kind to the same
target now add up within the approval window. The fixture is a ledger agent-governance itself
wrote with a frozen clock; the Python replay has to match it byte for byte.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentguard.gate import APPROVAL_TTL_SECONDS, Governor, SameApproverError, State, load_entries, verify_chain

HUMAN = "you@company.com"
FINANCE = "finance@company.com"
KEYS = {HUMAN: "demo-human-reviewer-key", FINANCE: "demo-finance-reviewer-key"}
SPLIT_FIXTURE = Path(__file__).parent / "fixtures" / "governance-split-ledger.json"


class Clock:
    def __init__(self, start: float = 1_793_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def governor(**kw: object) -> tuple[Governor, Clock]:
    clock = Clock()
    return Governor(reviewer_keys=KEYS, clock=clock, **kw), clock  # type: ignore[arg-type]


def refund(g: Governor, cents: int, order: str = "A-1", agent: str = "billing-agent"):
    return g.propose(agent, "refund.create", {"order": order, "amount_cents": cents}, estimated_cost_cents=cents)


def approve(g: Governor, action_id: str, who: str = HUMAN):
    return g.decide(action_id, True, who, note="ok", token=g.mint_token(action_id, who))


def test_a_split_refund_needs_two_people_for_the_part_that_crosses_the_line():
    g, _ = governor()
    assert approve(g, refund(g, 20_000).id).state is State.EXECUTED
    second = refund(g, 10_000)
    assert approve(g, second.id).state is State.PENDING_APPROVAL
    assert second.approvals == [HUMAN] and g.spent_cents == 20_000
    with pytest.raises(SameApproverError):
        approve(g, second.id)
    assert approve(g, second.id, FINANCE).state is State.EXECUTED and g.spent_cents == 30_000


def test_three_parts_are_caught_on_the_one_that_crosses():
    g, _ = governor()
    assert approve(g, refund(g, 10_000).id).state is State.EXECUTED
    assert approve(g, refund(g, 9_999).id).state is State.EXECUTED
    assert approve(g, refund(g, 5_002).id).state is State.PENDING_APPROVAL


def test_other_agents_count_and_other_orders_do_not():
    g, _ = governor()
    approve(g, refund(g, 20_000, "A-1").id)
    assert approve(g, refund(g, 10_000, "A-1", agent="support-agent").id).state is State.PENDING_APPROVAL
    assert approve(g, refund(g, 10_000, "B-2").id).state is State.EXECUTED


def test_only_money_that_moved_within_the_window_counts():
    g, clock = governor()
    rejected = refund(g, 20_000)
    g.decide(rejected.id, False, HUMAN, token=g.mint_token(rejected.id, HUMAN))
    refund(g, 15_000)  # still waiting
    assert approve(g, refund(g, 10_000).id).state is State.EXECUTED
    clock.now += APPROVAL_TTL_SECONDS + 1
    assert approve(g, refund(g, 20_001).id).state is State.EXECUTED


def test_up_to_the_line_exactly_one_approval_is_enough():
    g, _ = governor()
    approve(g, refund(g, 20_000).id)
    assert approve(g, refund(g, 5_000).id).state is State.EXECUTED


def test_switching_the_two_person_rule_off_switches_this_off_too():
    g, _ = governor(two_person_above_cents=None)
    approve(g, refund(g, 20_000).id)
    assert approve(g, refund(g, 10_000).id).state is State.EXECUTED


def _replay_typescript_split_run() -> Governor:
    """The calls agent-governance's store made for the split fixture: its six seed actions, $200 to
    order SP-1 on one yes, $100 more to SP-1 (you, then finance), $150 to SP-2 on one yes, and $50
    more to SP-1 from another agent that finance rejects after your yes."""
    clock = Clock(datetime(2026, 11, 17, 9, 0, tzinfo=timezone.utc).timestamp())
    g = Governor(reviewer_keys=KEYS, clock=clock)
    g.propose("support-agent", "crm.read", {"customer": "acme-industries"})
    g.propose("support-agent", "ticket.create", {"subject": "Duplicate charge on invoice 8842", "priority": "high"})
    g.propose("billing-agent", "refund.issue", {"invoice": "8842", "amount_cents": 12_400, "reason": "duplicate charge"},
              estimated_cost_cents=12_400)
    g.propose("growth-agent", "email.send", {"to": "2,300 customers", "template": "winback-q3"}, estimated_cost_cents=900)
    g.propose("rogue-agent", "db.drop", {"table": "customers"})
    g.propose("rogue-agent", "crypto.transfer", {"to": "0xbeef…", "amount": "all"})

    def part(cents: int, order: str, agent: str = "billing-agent"):
        return g.propose(agent, "refund.create", {"customer": "acme", "amount_cents": cents, "order": order},
                         estimated_cost_cents=cents)

    def decide(action_id: str, yes: bool, who: str, note: str):
        g.decide(action_id, yes, who, note=note, token=g.mint_token(action_id, who))

    a = part(20_000, "SP-1")
    decide(a.id, True, HUMAN, "ok")
    b = part(10_000, "SP-1")
    decide(b.id, True, HUMAN, "ok")
    decide(b.id, True, FINANCE, "second yes")
    c = part(15_000, "SP-2")
    decide(c.id, True, HUMAN, "ok")
    d = part(5_000, "SP-1", agent="support-agent")
    decide(d.id, True, HUMAN, "ok")
    decide(d.id, False, FINANCE, "no")
    return g


def test_the_split_ledger_matches_agent_governance_byte_for_byte():
    data = json.loads(SPLIT_FIXTURE.read_text(encoding="utf-8"))
    g = _replay_typescript_split_run()
    assert [e.to_dict() for e in g.ledger()] == data["chain"]
    assert g.spent_cents == data["budget"]["spentCents"]
    assert [[a.id, a.state.value, a.decision.decided_by, a.approvals] for a in g.actions()] == data["states"]
    assert verify_chain(load_entries(data)).valid

