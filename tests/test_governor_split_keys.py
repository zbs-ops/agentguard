"""The same target written another way (agent-governance PR #18).

0.16.0 matched a target by the exact argument name, so the same order written another way looked
like a different target and the second part of a split ran on one approval: ``order`` then
``order_id``, camelCase names (``orderId``, which also fell back to the exact match so a new reason
slipped through), a capitalised name, or the order inside a nested object or a list. A name is
now read by what it names, and nested arguments are read too. The fixture was written by
agent-governance main with a frozen clock: the kind it reads for each tricky name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentguard.gate import Governor, State
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
