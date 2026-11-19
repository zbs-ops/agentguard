"""Payouts to the same payee (agent-governance PR #19).

0.17.0's target kinds spoke refund (order, invoice, customer), so a split payout to the same vendor,
IBAN or wallet with a new reason ran on one approval. Payout destinations are targets now (vendor,
merchant, supplier, beneficiary, employee, contractor, counterparty, member, destination, iban,
card, wallet, phone, paypal); every name for who gets paid shares one namespace, so the same value
under any two of them is the same payee; account, card and phone numbers are compared without
spaces, dashes, dots or brackets. The fixture was written by agent-governance main with a frozen
clock: the kind it reads for each payout name, its verdict on the second part of a split payout for
each tricky pair, and a ledger.
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
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "governance-split-payees.json").read_text(encoding="utf-8"))


class Clock:
    def __init__(self, start: float = 1_793_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def second_payout(first: dict, second: dict) -> State:
    g = Governor(reviewer_keys=KEYS, clock=Clock())

    def pay(args: dict, cents: int):
        a = g.propose("billing-agent", "payout.send", {**args, "amount_cents": cents}, estimated_cost_cents=cents)
        return g.decide(a.id, True, HUMAN, note="ok", token=g.mint_token(a.id, HUMAN))

    assert pay({**first, "reason": "a"}, 20_000).state is State.EXECUTED
    return pay({**second, "reason": "b"}, 10_000).state


@pytest.mark.parametrize(("name", "kind"), FIXTURE["kinds"], ids=lambda x: repr(x))
def test_each_payout_name_is_read_as_agent_governance_reads_it(name, kind):
    assert target_kind(name) == kind


@pytest.mark.parametrize(("first", "second", "first_state", "second_state"), FIXTURE["pairs"], ids=lambda x: repr(x))
def test_every_payout_pair_gets_agent_governances_verdict(first, second, first_state, second_state):
    assert first_state == "EXECUTED"
    assert second_payout(first, second).value == second_state


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"vendor": "v_1"}, {"vendor": "v_1"}),
        ({"iban": "DE89370400440532013000"}, {"bank_account": "DE89 3704 0044 0532 0130 00"}),
        ({"payee": "acme"}, {"vendor": "ACME"}),
        ({"to": "0xBEEF"}, {"wallet": "0xbeef"}),
        ({"phone": "+1 (555) 0100"}, {"phone_number": "+15550100"}),
    ],
)
def test_a_split_payout_to_the_same_payee_needs_a_second_person(first, second):
    assert second_payout(first, second) is State.PENDING_APPROVAL


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"vendor": "v_1"}, {"vendor": "v_2"}),
        ({"client_id": "app_1", "vendor": "v_1"}, {"client_id": "app_1", "vendor": "v_2"}),
        ({"order": "42"}, {"invoice": "42"}),
        ({"order": "T-1"}, {"order": "T1"}),
    ],
)
def test_different_targets_stay_separate(first, second):
    assert second_payout(first, second) is State.EXECUTED


def _replay_typescript_payout_run() -> Governor:
    """The calls agent-governance's store made for the fixture: its six seed actions; $200 payout
    to vendor acme by IBAN on one yes; $100 through stripe.refund to the same IBAN written with
    spaces as a bank account, with a reason (you, then finance); $100 to vendor_name globex by
    wallet on one yes; $60 to payee " ACME ", which waits and finance rejects; $50 through
    refund.issue to the wallet as `to` in another case, on one yes."""
    g = Governor(reviewer_keys=KEYS, clock=Clock(datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc).timestamp()))
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

    a = pay("payout.send", {"vendor": "acme", "iban": "DE89370400440532013000"}, 20_000)
    decide(a.id, True, HUMAN, "ok")
    b = pay("stripe.refund", {"bank_account": "DE89 3704 0044 0532 0130 00", "reason": "late"}, 10_000)
    decide(b.id, True, HUMAN, "ok")
    decide(b.id, True, FINANCE, "second yes")
    c = pay("payout.send", {"vendor_name": "globex", "wallet": "0xG"}, 10_000)
    decide(c.id, True, HUMAN, "ok")
    d = pay("payout.send", {"payee": " ACME "}, 6_000)
    decide(d.id, True, HUMAN, "ok")
    decide(d.id, False, FINANCE, "no")
    e = pay("refund.issue", {"to": "0xg"}, 5_000)
    decide(e.id, True, HUMAN, "ok")
    return g


def test_the_payout_ledger_matches_agent_governance_byte_for_byte():
    g = _replay_typescript_payout_run()
    assert [e.to_dict() for e in g.ledger()] == FIXTURE["chain"]
    assert g.spent_cents == FIXTURE["budget"]["spentCents"]
    assert [[a.id, a.state.value, a.decision.decided_by, a.approvals] for a in g.actions()] == FIXTURE["states"]
    assert verify_chain(load_entries(FIXTURE)).valid
