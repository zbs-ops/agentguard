"""Amounts written any other way (agent-governance PRs #20 and #21).

The gate priced a call from ``args["amount_cents"]`` only, so a $300 refund with a $0 estimate ran
on one approval, charged nothing, when the amount was written as ``amountCents``, ``cents``,
``amount_in_cents``, ``amountcents``, ``amount: 300``, ``total``, nested, or as line items. Now
every field whose last word ends in cents is priced, at any depth, a list of line items counts as
their sum, and the call is charged the largest of these and the estimate; on a tool that moves
money, a number named like an amount with no unit is refused (``unpriced-amount``); arguments
nested past 32 levels are refused. The fixture was written by agent-governance main with a frozen
clock: ``evaluate``'s verdict, tier, rule, charge and reasons for 103 argument shapes, and a ledger.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentguard.gate import Governor, load_entries, verify_chain
from agentguard.gate.policy import Budget, evaluate

HUMAN = "you@company.com"
FINANCE = "finance@company.com"
KEYS = {HUMAN: "demo-human-reviewer-key", FINANCE: "demo-finance-reviewer-key"}
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "governance-amounts.json").read_text(encoding="utf-8"))
BUDGET = Budget(spent_cents=FIXTURE["budget"]["spentCents"], limit_cents=FIXTURE["budget"]["limitCents"])


class Clock:
    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def _id(case: dict) -> str:
    return f"{case['tool']} {json.dumps(case['args'], ensure_ascii=False)[:60]}"


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=_id)
def test_every_amount_gets_agent_governances_decision(case):
    d = evaluate(case["tool"], estimated_cost_cents=case["estimate"], args=case["args"], budget=BUDGET)
    assert [d.verdict.value, d.tier, d.decided_by, d.cost_cents, list(d.reasons)] == [
        case["verdict"], case["tier"], case["decidedBy"], case["costCents"], case["reasons"],
    ]


@pytest.mark.parametrize(
    "args",
    [
        {"amountCents": 30_000},
        {"amountcents": 30_000},
        {"cents": 30_000},
        {"money": {"amount_cents": 30_000}},
        {"items": [{"amount_cents": 20_000}, {"amount_cents": 10_000}]},
    ],
)
def test_a_300_dollar_refund_under_another_name_needs_a_second_person(args):
    g = Governor(reviewer_keys=KEYS, clock=Clock(1_793_000_000.0))
    a = g.propose("billing-agent", "refund.issue", {"invoice": "R4-1", **args}, estimated_cost_cents=0)
    after = g.decide(a.id, True, HUMAN, note="ok", token=g.mint_token(a.id, HUMAN))
    assert after.state.value == "PENDING_APPROVAL"
    assert g.spent_cents == 0


@pytest.mark.parametrize("args", [{"amount": 300}, {"total": "300.00"}, {"totalamount": 300}, {"amount_cents": 100, "amount": 300}])
def test_an_amount_with_no_unit_on_a_money_tool_is_refused(args):
    d = evaluate("payout.send", args=args, budget=BUDGET)
    assert d.decided_by == "unpriced-amount"
    assert "amount_cents" in d.reasons[-1]


def test_python_only_values_are_read_like_their_json():
    # A tuple is a list, a float that is whole is a whole number, and a bool is never a number.
    assert evaluate("refund.issue", args={"items": ({"cents": 200}, {"cents": 100})}, budget=BUDGET).cost_cents == 300
    assert evaluate("refund.issue", args={"cents": 300.0}, budget=BUDGET).cost_cents == 300
    assert evaluate("refund.issue", args={"amount": True}, budget=BUDGET).decided_by == "money-movement"


def _replay_typescript_amounts_run() -> Governor:
    """The calls agent-governance's store made for the fixture: its six seed actions; $300 as
    amountCents with a $0 estimate (you, then finance); $300 as amount (refused); $120 as line items
    with a $1 estimate on one yes; a negative nested cents value (refused); 800 fee_cents on an
    internal write, rejected; $200 as cents with the budget nearly spent (refused)."""
    g = Governor(reviewer_keys=KEYS, clock=Clock(datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc).timestamp()))
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

    def decide(action_id: str, yes: bool, who: str, note: str):
        g.decide(action_id, yes, who, note=note, token=g.mint_token(action_id, who))

    a = g.propose("billing-agent", "refund.issue", {"invoice": "L-1", "amountCents": 30_000}, estimated_cost_cents=0)
    decide(a.id, True, HUMAN, "ok")
    decide(a.id, True, FINANCE, "second yes")
    g.propose("billing-agent", "refund.issue", {"invoice": "L-2", "amount": 300}, estimated_cost_cents=0)
    c = g.propose("billing-agent", "stripe.refund", {"invoice": "L-3", "items": [{"amount_cents": 7_000}, {"amount_cents": 5_000}]}, estimated_cost_cents=100)
    decide(c.id, True, HUMAN, "ok")
    g.propose("billing-agent", "payout.send", {"vendor": "v_9", "money": {"cents": -1}}, estimated_cost_cents=0)
    e = g.propose("billing-agent", "crm.update", {"note": "x", "fee_cents": 800}, estimated_cost_cents=0)
    decide(e.id, False, HUMAN, "no")
    g.propose("billing-agent", "refund.issue", {"invoice": "L-6", "cents": 20_000}, estimated_cost_cents=0)
    return g


def test_the_amounts_ledger_matches_agent_governance_byte_for_byte():
    g = _replay_typescript_amounts_run()
    ledger = FIXTURE["ledger"]
    assert [[a.id, a.state.value, a.decision.decided_by, a.decision.cost_cents, a.approvals] for a in g.actions()] == ledger["states"]
    assert g.spent_cents == ledger["budget"]["spentCents"]
    assert [e.to_dict() for e in g.ledger()] == ledger["chain"]
    assert verify_chain(load_entries(ledger)).valid
