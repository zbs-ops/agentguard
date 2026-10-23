"""The gate's answers on the cases that would actually hurt."""

from __future__ import annotations

import pytest

from agentguard.gate import (
    UNATTENDED_WRITE_LIMIT,
    Budget,
    History,
    Verdict,
    evaluate,
)

BUDGET = Budget(spent_cents=0, limit_cents=50_000)


def test_reads_run_unattended() -> None:
    assert evaluate("crm.read", budget=BUDGET).verdict is Verdict.ALLOW


def test_internal_write_runs_but_tier_2() -> None:
    d = evaluate("ticket.create", budget=BUDGET)
    assert d.verdict is Verdict.ALLOW
    assert d.tier == 2


def test_money_movement_waits_for_a_human() -> None:
    d = evaluate("refund.create", estimated_cost_cents=8_400, budget=BUDGET)
    assert d.verdict is Verdict.REQUIRE_APPROVAL
    assert d.tier == 3


@pytest.mark.parametrize("tool", ["stripe", "STRIPE.refund", "search.x/../refund.create", "a.b\nrefund.create", "refund.create\n"])
def test_malformed_tool_name_is_refused_before_matching(tool: str) -> None:
    d = evaluate(tool, budget=BUDGET)
    assert d.verdict is Verdict.DENY
    assert d.decided_by == "malformed-tool"


def test_unknown_tool_fails_closed() -> None:
    d = evaluate("vault.exfiltrate", budget=BUDGET)
    assert d.verdict is Verdict.DENY
    assert d.decided_by == "unknown-tool"


@pytest.mark.parametrize("bad", [-1, 1.5, float("nan"), float("inf"), True])
def test_untrustworthy_cost_is_refused_before_the_budget(bad: object) -> None:
    d = evaluate("ticket.create", estimated_cost_cents=bad, budget=BUDGET)  # type: ignore[arg-type]
    assert d.verdict is Verdict.DENY
    assert d.decided_by == "malformed-cost"


def test_budget_is_checked_before_the_tier() -> None:
    # A tier-1 read is still refused when the money has run out, because budget comes first.
    spent = Budget(spent_cents=50_000, limit_cents=50_000)
    d = evaluate("crm.read", estimated_cost_cents=1, budget=spent)
    assert d.verdict is Verdict.DENY
    assert d.decided_by == "budget-exceeded"


def test_spend_exactly_on_the_cap_is_allowed_one_cent_over_is_not() -> None:
    near = Budget(spent_cents=49_900, limit_cents=50_000)
    assert evaluate("ticket.create", estimated_cost_cents=100, budget=near).verdict is Verdict.ALLOW
    assert evaluate("crm.read", estimated_cost_cents=101, budget=near).decided_by == "budget-exceeded"


def test_understated_cost_is_escalated_and_shows_the_real_amount() -> None:
    d = evaluate("ticket.create", estimated_cost_cents=100, args={"amount_cents": 9_000}, budget=BUDGET)
    assert d.verdict is Verdict.REQUIRE_APPROVAL
    assert d.decided_by == "cost-understated"
    assert d.cost_cents == 9_000
    assert "$90.00" in d.reason


def test_one_unusually_large_spend_is_escalated_even_when_it_fits() -> None:
    d = evaluate("refund.create", estimated_cost_cents=20_000, budget=BUDGET)
    assert d.verdict is Verdict.REQUIRE_APPROVAL
    assert d.decided_by in {"large-single-spend", "money-movement"}


def test_a_runaway_loop_of_writes_is_paused() -> None:
    under = History(recent_unattended_writes=UNATTENDED_WRITE_LIMIT - 1)
    at = History(recent_unattended_writes=UNATTENDED_WRITE_LIMIT)
    assert evaluate("ticket.create", budget=BUDGET, history=under).verdict is Verdict.ALLOW
    paused = evaluate("ticket.create", budget=BUDGET, history=at)
    assert paused.verdict is Verdict.REQUIRE_APPROVAL
    assert paused.decided_by == "unattended-rate"


def test_reads_are_never_slowed_by_the_rate_limit() -> None:
    hammering = History(recent_unattended_writes=UNATTENDED_WRITE_LIMIT * 5)
    assert evaluate("crm.read", budget=BUDGET, history=hammering).verdict is Verdict.ALLOW


def test_same_inputs_give_the_same_verdict() -> None:
    a = evaluate("refund.create", estimated_cost_cents=5_000, budget=BUDGET)
    b = evaluate("refund.create", estimated_cost_cents=5_000, budget=BUDGET)
    assert a == b
