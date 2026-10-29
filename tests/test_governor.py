"""The approval workflow: who may approve, when an approval expires, and what gets charged.

Ported from the agent-governance store, approval, budget-at-approval, expiry and rate-limit
suites, plus a byte-for-byte check against a ledger the TypeScript project wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentguard.gate import (
    APPROVAL_TTL_SECONDS,
    UNATTENDED_WRITE_LIMIT,
    ApprovalAuthError,
    Governor,
    SelfApprovalError,
    State,
    mint_approval_token,
    proposal_digest,
)
from agentguard.gate.ledger import Entry, load_entries, verify_chain

HUMAN = "you@company.com"
KEYS = {HUMAN: "demo-human-reviewer-key"}
FIXTURE = Path(__file__).parent / "fixtures" / "governance-ledger.json"


class Clock:
    def __init__(self, start: float = 1_793_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def governor(**kw: object) -> tuple[Governor, Clock]:
    clock = Clock()
    return Governor(reviewer_keys=KEYS, clock=clock, **kw), clock  # type: ignore[arg-type]


def refund(g: Governor, cents: int, tool: str = "refund.create"):
    return g.propose("billing-agent", tool, {"amount_cents": cents}, estimated_cost_cents=cents)


def approve(g: Governor, action_id: str, note: str | None = "ok"):
    return g.decide(action_id, True, HUMAN, note=note, token=g.mint_token(action_id, HUMAN))


# -- who may approve --------------------------------------------------------------------


def test_an_agent_cannot_approve_its_own_proposal_even_with_odd_spacing():
    g, _ = governor()
    a = refund(g, 5_000)
    for name in ("billing-agent", "  Billing-Agent "):
        with pytest.raises(SelfApprovalError):
            g.decide(a.id, True, name, token="x")


def test_a_refused_self_approval_leaves_no_trace():
    g, _ = governor()
    a = refund(g, 5_000)
    before = (g.spent_cents, len(g.ledger()))
    with pytest.raises(SelfApprovalError):
        g.decide(a.id, True, "billing-agent")
    assert a.state is State.PENDING_APPROVAL
    assert (g.spent_cents, len(g.ledger())) == before


def test_an_agent_can_withdraw_its_own_proposal():
    g, _ = governor()
    assert g.decide(refund(g, 5_000).id, False, "billing-agent").state is State.REJECTED


@pytest.mark.parametrize("token", [None, "not-a-real-token"])
def test_an_approval_needs_a_real_token(token):
    g, _ = governor()
    a = refund(g, 5_000)
    before = (g.spent_cents, len(g.ledger()))
    with pytest.raises(ApprovalAuthError):
        g.decide(a.id, True, HUMAN, token=token)
    assert (g.spent_cents, len(g.ledger())) == before


def test_a_token_for_one_action_cannot_be_replayed_on_another():
    g, _ = governor()
    a, b = refund(g, 5_000), refund(g, 5_000)
    with pytest.raises(ApprovalAuthError):
        g.decide(a.id, True, HUMAN, token=g.mint_token(b.id, HUMAN))


def test_an_unregistered_reviewer_cannot_approve_or_get_a_token():
    g, _ = governor()
    a = refund(g, 5_000)
    with pytest.raises(ApprovalAuthError):
        g.decide(a.id, True, "stranger", token="anything")
    with pytest.raises(ApprovalAuthError):
        g.mint_token(a.id, "stranger")


def test_an_authenticated_no_and_an_authenticated_yes_both_work():
    g, _ = governor()
    a = refund(g, 5_000)
    assert g.decide(a.id, False, HUMAN, token=g.mint_token(a.id, HUMAN)).state is State.REJECTED
    assert approve(g, refund(g, 5_000).id).state is State.EXECUTED


def test_a_reject_in_a_persons_name_needs_their_token_and_writes_nothing_without_it():
    # Otherwise the ledger would record "REJECTED by you@company.com" for a decision that person
    # never made, in an entry nobody can edit afterwards.
    g, _ = governor()
    a = refund(g, 5_000)
    before = len(g.ledger())
    with pytest.raises(ApprovalAuthError):
        g.decide(a.id, False, HUMAN)
    with pytest.raises(ApprovalAuthError):
        g.decide(a.id, False, HUMAN, token="f" * 64)
    with pytest.raises(ApprovalAuthError):
        g.decide(a.id, False, "support-agent")  # another agent can't cancel this one's work
    assert g.get(a.id).state is State.PENDING_APPROVAL
    assert len(g.ledger()) == before


def test_the_agent_that_asked_can_still_withdraw_without_a_token():
    g, _ = governor()
    a = refund(g, 5_000)
    assert g.decide(a.id, False, " Billing-Agent").state is State.REJECTED
    assert g.ledger()[-1].event == "REJECTED"


def test_deciding_an_unknown_or_finished_action_returns_none():
    g, _ = governor()
    a = refund(g, 5_000)
    approve(g, a.id)
    assert g.decide(a.id, False, HUMAN) is None
    assert g.decide("act_9999", False, HUMAN) is None


# -- what gets charged ------------------------------------------------------------------


def test_an_understated_refund_is_charged_its_real_amount():
    g, _ = governor()
    a = g.propose("billing-agent", "stripe.refund", {"amount_cents": 9_000}, estimated_cost_cents=100)
    assert a.state is State.PENDING_APPROVAL and a.decision.decided_by == "cost-understated"
    approve(g, a.id)
    assert g.spent_cents == 9_000


def test_unattended_writes_are_charged_their_estimate_and_denials_nothing():
    g, _ = governor()
    g.propose("a", "ticket.create", {"subject": "x"}, estimated_cost_cents=250)
    g.propose("a", "db.drop", {"table": "customers"})
    g.propose("a", "crypto.transfer", {"amount": "all"})
    assert g.spent_cents == 250


# -- the budget at approval time --------------------------------------------------------


def test_the_second_of_two_refunds_that_only_fit_alone_is_blocked_at_approval():
    g, _ = governor()
    each = int(g.budget_limit_cents * 0.6)
    first, second = refund(g, each), refund(g, each)
    assert approve(g, first.id).state is State.EXECUTED
    late = approve(g, second.id)
    assert late.state is State.BLOCKED
    assert late.decision.decided_by == "budget-exceeded-at-approval"
    assert g.spent_cents == each
    events = [e.event for e in g.ledger() if e.proposal_id == second.id]
    assert "APPROVED" in events and "BLOCKED" in events and "EXECUTED" not in events
    assert g.verify().valid


def test_spend_lands_exactly_on_the_cap_and_no_further():
    g, _ = governor()
    limit = g.budget_limit_cents
    big, small, over = refund(g, limit - 100, "stripe.refund"), refund(g, 100), refund(g, 1)
    assert approve(g, big.id).state is State.EXECUTED
    assert approve(g, small.id).state is State.EXECUTED
    assert g.spent_cents == limit
    assert approve(g, over.id).state is State.BLOCKED


# -- approvals expire -------------------------------------------------------------------


def test_an_approval_inside_and_exactly_at_the_window_runs():
    for offset in (APPROVAL_TTL_SECONDS - 60, APPROVAL_TTL_SECONDS):
        g, clock = governor()
        a = refund(g, 2_000)
        clock.now += offset
        assert approve(g, a.id).state is State.EXECUTED


def test_a_late_approval_is_blocked_charges_nothing_and_says_why():
    g, clock = governor()
    a = refund(g, 2_000)
    clock.now += APPROVAL_TTL_SECONDS + 1
    late = approve(g, a.id)
    assert late.state is State.BLOCKED and late.decision.decided_by == "approval-expired"
    assert g.spent_cents == 0
    last = [e for e in g.ledger() if e.proposal_id == a.id][-1]
    assert last.event == "BLOCKED" and "expired" in last.detail["why"]
    assert g.verify().valid


def test_a_stale_request_can_still_be_rejected():
    g, clock = governor()
    a = refund(g, 2_000)
    clock.now += APPROVAL_TTL_SECONDS * 3
    assert g.decide(a.id, False, HUMAN, note="too old", token=g.mint_token(a.id, HUMAN)).state is State.REJECTED


def test_identity_is_checked_before_the_clock():
    g, clock = governor()
    a = refund(g, 2_000)
    clock.now += APPROVAL_TTL_SECONDS + 1
    with pytest.raises(ApprovalAuthError):
        g.decide(a.id, True, HUMAN, token="forged")
    assert a.state is State.PENDING_APPROVAL


def test_the_window_is_configurable():
    g, clock = governor(approval_ttl_seconds=60)
    a = refund(g, 2_000)
    clock.now += 61
    assert approve(g, a.id).decision.decided_by == "approval-expired"


# -- runaway loops ----------------------------------------------------------------------


def test_a_loop_of_unattended_writes_is_slowed_to_ask_a_person():
    g, clock = governor()
    states = []
    for _ in range(UNATTENDED_WRITE_LIMIT + 1):
        states.append(g.propose("loop-agent", "ticket.create", {"subject": "x"}).state)
        clock.now += 1
    assert states[:-1] == [State.EXECUTED] * UNATTENDED_WRITE_LIMIT
    assert states[-1] is State.PENDING_APPROVAL
    # Another agent, and reads, are not slowed.
    assert g.propose("other-agent", "ticket.create", {}).state is State.EXECUTED
    assert g.propose("loop-agent", "crm.read", {}).state is State.EXECUTED


def test_the_rate_window_slides():
    g, clock = governor()
    for _ in range(UNATTENDED_WRITE_LIMIT):
        g.propose("loop-agent", "ticket.create", {})
    clock.now += 61
    assert g.propose("loop-agent", "ticket.create", {}).state is State.EXECUTED


# -- the ledger -------------------------------------------------------------------------


def _chain(n: int) -> list[Entry]:
    from agentguard.gate.ledger import append_entry

    chain: list[Entry] = []
    for i in range(1, n + 1):
        append_entry(chain, at=f"2026-01-0{i}T00:00:00.000Z", event="PROPOSED", actor="agent",
                     proposal_id=f"act_{i}", detail={"n": i})
    return chain


def test_ledger_links_and_verifies():
    chain = _chain(3)
    assert chain[0].prev_hash == "GENESIS" and chain[1].prev_hash == chain[0].hash
    assert verify_chain(_chain(5)).valid and verify_chain([]).valid


def test_ledger_names_the_first_edited_entry():
    chain = _chain(5)
    chain[2].detail = {"n": "quietly changed"}
    assert verify_chain(chain).broken_at == 3
    chain = _chain(4)
    chain[1].actor = "someone-else"
    assert verify_chain(chain).broken_at == 2


def test_ledger_catches_deletion_reordering_and_a_rehashed_forgery():
    from agentguard.gate.ledger import hash_entry

    chain = _chain(5)
    del chain[2]
    assert not verify_chain(chain).valid
    chain = _chain(4)
    chain[1], chain[2] = chain[2], chain[1]
    assert not verify_chain(chain).valid
    chain = _chain(4)
    chain[1].detail = {"n": "rewritten"}
    chain[1].hash = hash_entry(chain[1])
    assert verify_chain(chain).broken_at == 3


# -- parity with the TypeScript project -------------------------------------------------


def _replay_typescript_run() -> Governor:
    """The same calls the TypeScript store made (its six seed actions, then the fixture's)."""
    from datetime import datetime, timezone

    clock = Clock(datetime(2026, 10, 26, 2, 17, tzinfo=timezone.utc).timestamp())
    g = Governor(reviewer_keys=KEYS, clock=clock)
    g.propose("support-agent", "crm.read", {"customer": "acme-industries"})
    g.propose("support-agent", "ticket.create", {"subject": "Duplicate charge on invoice 8842", "priority": "high"})
    g.propose("billing-agent", "refund.issue", {"invoice": "8842", "amount_cents": 12_400, "reason": "duplicate charge"},
              estimated_cost_cents=12_400)
    g.propose("growth-agent", "email.send", {"to": "2,300 customers", "template": "winback-q3"}, estimated_cost_cents=900)
    g.propose("rogue-agent", "db.drop", {"table": "customers"})
    g.propose("rogue-agent", "crypto.transfer", {"to": "0xbeef…", "amount": "all"})
    r = refund(g, 20_000)
    approve(g, r.id)
    s = refund(g, 1_000)
    g.decide(s.id, False, HUMAN, token=g.mint_token(s.id, HUMAN))  # tokens never reach the ledger
    t = refund(g, 500)
    clock.now = datetime(2026, 10, 27, 3, 0, tzinfo=timezone.utc).timestamp()
    approve(g, t.id, note="late")
    return g


def test_a_ledger_written_by_the_typescript_project_verifies_here():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    chain = load_entries(data)
    assert len(chain) == 25
    assert verify_chain(chain).valid
    chain[4].detail["tier"] = 1
    assert verify_chain(chain).broken_at == 5


def test_the_same_calls_produce_the_same_ledger_byte_for_byte():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    g = _replay_typescript_run()
    assert [e.to_dict() for e in g.ledger()] == data["chain"]
    assert g.spent_cents == data["budget"]["spentCents"]
    assert [[a.id, a.state.value, a.decision.decided_by] for a in reversed(g.actions())] == data["states"]


# Reference values printed by agent-governance's proposalDigest / mintApprovalToken (PR #10).
TS_REFERENCE = [
    (
        ("act_0001", "billing-agent", "stripe.refund", {"customer": "acme", "amount_cents": 500}, 500,
         "2026-10-28T05:00:00.000Z"),
        "e977ac4bbc68be50d51fed5b8ee0e4355a940626f71ad09f7e1bd7e9e9f9bb1f",
        "e84399c9cb692b2496ba9605b7487ba83883d43cc1f40b3c984e1e63715b8ddd",
    ),
    (
        ("act_0042", "ops", "crm.update",
         {"z": [1, 2.5, True, None], "a": {"\u00fc": 'caf\u00e9 "x"\n', "n": 1.0, "k": -0.0}}, 0,
         "2026-10-28T05:00:00.123Z"),
        "de5a8e0b80e945ab6a123f9107df81a874ce71a69913ea3dc08efd8990aaf21e",
        "b17d53dfbfa446262249003512b4c592cbdf058fdd68ab0ff15ba817e3514d31",
    ),
]


@pytest.mark.parametrize(("proposal", "digest", "token"), TS_REFERENCE)
def test_digest_and_token_match_the_typescript_bytes(proposal, digest, token):
    assert proposal_digest(*proposal) == digest
    assert mint_approval_token(KEYS[HUMAN], proposal[0], HUMAN, digest=digest) == token
    assert mint_approval_token(KEYS[HUMAN], proposal[0], " You@Company.com ", digest=digest) == token


def test_a_token_does_not_carry_over_to_a_new_action_with_the_same_id():
    # Ids are a counter, so a fresh Governor (a restart) hands out act_0001 again.
    first, _ = governor()
    small = refund(first, 2_000, tool="stripe.refund")
    old_token = first.mint_token(small.id, HUMAN)

    second, _ = governor()
    big = second.propose("billing-agent", "stripe.refund", {"amount_cents": 40_000, "customer": "someone-else"},
                         estimated_cost_cents=40_000)
    assert big.id == small.id
    with pytest.raises(ApprovalAuthError):
        second.decide(big.id, True, HUMAN, token=old_token)
    assert second.spent_cents == 0
    assert second.get(big.id).state is State.PENDING_APPROVAL


@pytest.mark.parametrize(
    "change",
    [
        {"args": {"amount_cents": 2_001}},
        {"args": {"amount_cents": 2_000, "note": "and the rest"}},
        {"tool": "stripe.payout"},
        {"agent": "other-agent"},
        {"estimated_cost_cents": 1},
        {"created_at": 1_793_000_001.0},
    ],
)
def test_a_token_is_refused_if_anything_it_covered_changed(change):
    g, _ = governor()
    a = refund(g, 2_000, tool="stripe.refund")
    token = g.mint_token(a.id, HUMAN)
    for k, v in change.items():
        setattr(a, k, v)
    with pytest.raises(ApprovalAuthError):
        g.decide(a.id, True, HUMAN, token=token)


def test_argument_order_does_not_change_the_digest():
    one = proposal_digest("act_0001", "a", "t", {"x": 1, "y": {"b": 2, "a": 1}}, 0, "2026-10-28T05:00:00.000Z")
    two = proposal_digest("act_0001", "a", "t", {"y": {"a": 1, "b": 2}, "x": 1}, 0, "2026-10-28T05:00:00.000Z")
    assert one == two


def test_cli_verify_ledger(tmp_path, capsys):
    from agentguard.cli import main

    assert main(["verify-ledger", str(FIXTURE)]) == 0
    assert "All 25 entries verify" in capsys.readouterr().out
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data["chain"][9]["actor"] = "someone-else"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    assert main(["verify-ledger", str(bad)]) == 1
    assert "Entry #10" in capsys.readouterr().out
    junk = tmp_path / "junk.json"
    junk.write_text('{"nope": 1}', encoding="utf-8")
    assert main(["verify-ledger", str(junk)]) == 2


def test_reads_the_live_demo_audit_api_shape():
    # agent-governance's GET /api/audit answers {"entries": [...]}; that file should verify as-is.
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    chain = load_entries({"entries": data["chain"]})
    assert len(chain) == 25 and verify_chain(chain).valid


def test_cli_verifies_a_saved_audit_api_response(tmp_path, capsys):
    from agentguard.cli import main

    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    saved = tmp_path / "audit.json"
    saved.write_text(json.dumps({"entries": data["chain"]}), encoding="utf-8")
    assert main(["verify-ledger", str(saved)]) == 0
    assert "All 25 entries verify" in capsys.readouterr().out
