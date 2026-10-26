"""A deterministic policy gate for agent tool calls.

Every tool call is a *proposal*. ``evaluate`` returns one of three verdicts: run it, hold it
for a person, or refuse it. The gate is a pure function of the proposal, the budget, and a
small history object, so the same inputs always give the same verdict, which is what makes it
testable. Ported from the TypeScript ``agent-governance`` project; the reason strings match it
word for word, because they end up in ledger entries whose hashes must agree across the two.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "POLICY",
    "UNATTENDED_WINDOW_SECONDS",
    "UNATTENDED_WRITE_LIMIT",
    "Budget",
    "Decision",
    "History",
    "Verdict",
    "evaluate",
    "money",
]


class Verdict(str, Enum):
    """What the gate decided."""

    ALLOW = "ALLOW"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    DENY = "DENY"


@dataclass(frozen=True)
class _Rule:
    id: str
    describe: str
    tier: int
    match: tuple[str, ...]


# Policy is data, not conditionals buried in a service. Each rule is a row you can read and
# reorder. Tier 1 reads run unattended, tier 2 internal writes run but are recorded, tier 3
# (money, customers, production, deletes) waits for a human.
POLICY: tuple[_Rule, ...] = (
    _Rule("read-only", "Read-only lookups cannot change anything, so they run unattended.", 1,
          ("crm.read", "db.query", "search.*", "calendar.read")),
    _Rule("internal-write", "Writes to internal records are reversible and stay inside the company.", 2,
          ("crm.update", "ticket.create", "notify.internal", "db.insert")),
    _Rule("money-movement", "Anything that moves money needs a person to sign off.", 3,
          ("stripe.*", "payout.*", "invoice.void", "refund.*")),
    _Rule("customer-contact", "A message to a customer cannot be unsent.", 3,
          ("email.send", "sms.send", "whatsapp.send")),
    _Rule("destructive", "Deletes and production deploys need a person to sign off.", 3,
          ("db.delete", "db.drop", "deploy.production", "s3.delete")),
)

# Lowercase ASCII segments joined by dots, at least two of them: ``stripe.refund``. Matched
# with ``fullmatch``: Python's ``$`` also matches just before a trailing newline, so a name like
# ``"refund.create\n"`` would otherwise smuggle itself past this check.
_TOOL_NAME = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+")

# A single write is routine; a loop creating hundreds of tickets is a runaway agent. Past this
# many unattended tier-2 writes from one agent inside the window, the next one waits for a person.
UNATTENDED_WRITE_LIMIT = 10
UNATTENDED_WINDOW_SECONDS = 60


@dataclass(frozen=True)
class Budget:
    """A spend cap, in cents. ``spent_cents`` is what has been charged so far."""

    spent_cents: int
    limit_cents: int


@dataclass(frozen=True)
class History:
    """What the gate knows about recent activity. The caller supplies it; the gate stays pure."""

    recent_unattended_writes: int = 0


@dataclass(frozen=True)
class Decision:
    """The gate's answer, with every reason it is safe to show a human reviewer."""

    verdict: Verdict
    tier: int
    reasons: tuple[str, ...]
    decided_by: str
    #: What the gate charged against the budget: the larger of the estimate and any amount the
    #: arguments move. 0 when the gate refused before it could trust either number.
    cost_cents: int

    @property
    def reason(self) -> str:
        """The final reason, i.e. the one that produced the verdict."""
        return self.reasons[-1] if self.reasons else ""


def money(cents: int) -> str:
    """Render cents as a dollar string: ``money(8400)`` -> ``"$84.00"``."""
    return f"${cents / 100:.2f}"


def _is_cents(value: object) -> bool:
    """A whole, non-negative amount the budget arithmetic can trust (and not a bool)."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    ) or (isinstance(value, float) and value.is_integer() and value >= 0 and math.isfinite(value))


def _matches(pattern: str, tool: str) -> bool:
    if pattern.endswith("*"):
        return tool.startswith(pattern[:-1])
    return pattern == tool


def _rule_for(tool: str) -> _Rule | None:
    for rule in POLICY:
        if any(_matches(pattern, tool) for pattern in rule.match):
            return rule
    return None


def evaluate(
    tool: str,
    *,
    estimated_cost_cents: int = 0,
    args: dict[str, object] | None = None,
    budget: Budget,
    history: History | None = None,
) -> Decision:
    """Decide whether a proposed tool call may run, in seven ordered checks.

    The order is load-bearing and asserted by the tests: a malformed tool name is refused
    before it can borrow another rule's tier, the budget is checked before the tier, and an
    understated cost is escalated even on a tier that would otherwise run unattended.
    """
    args = args or {}
    history = history or History()
    reasons: list[str] = []

    # 0. A tool name has to look like a tool name before it is matched. Wildcards match by
    #    prefix, so without this a name like "search.x/../refund.create" would ride "search.*".
    if not _TOOL_NAME.fullmatch(tool):
        return Decision(Verdict.DENY, 3, (f"{json.dumps(tool, ensure_ascii=False)} is not a well-formed tool name.",), "malformed-tool", 0)

    # 1. Unknown tools fail closed.
    rule = _rule_for(tool)
    if rule is None:
        return Decision(Verdict.DENY, 3, (f'"{tool}" is not a tool this agent is allowed to call.',), "unknown-tool", 0)
    reasons.append(rule.describe)

    # 2. Both numbers must be ones the math can trust: the estimate and any amount the
    #    arguments move. A negative or NaN cost would slip past the cap.
    if not _is_cents(estimated_cost_cents):
        reasons.append(f"An estimated cost of {estimated_cost_cents} cents is not a whole, non-negative amount.")
        return Decision(Verdict.DENY, rule.tier, tuple(reasons), "malformed-cost", 0)
    arg_amount = args.get("amount_cents")
    if arg_amount is not None and not _is_cents(arg_amount):
        reasons.append(f"The arguments ask to move {arg_amount} cents, which is not a whole, non-negative amount.")
        return Decision(Verdict.DENY, rule.tier, tuple(reasons), "malformed-cost", 0)

    # The estimate is the agent's word; amount_cents is what the call will do. Charge the
    # larger, so understating the estimate can't buy room under the cap.
    estimate = int(estimated_cost_cents)
    moves = int(arg_amount) if isinstance(arg_amount, (int, float)) and _is_cents(arg_amount) else 0
    cost = max(estimate, moves)
    understated = cost > estimate

    # 3. Budget is checked before anything else, because a runaway spend is the failure that
    #    costs real money while everyone is asleep.
    projected = budget.spent_cents + cost
    if projected > budget.limit_cents:
        reasons.append(f"This would put spend at {money(projected)} against a {money(budget.limit_cents)} cap.")
        return Decision(Verdict.DENY, rule.tier, tuple(reasons), "budget-exceeded", cost)

    # 4. An agent that estimates one number and moves a bigger one gets a human's eyes, even on
    #    a tier that would normally run unattended, and the human sees the real amount.
    if understated:
        reasons.append(f"The agent estimated {money(estimate)}, but the arguments move {money(cost)} — escalated for review.")
        return Decision(Verdict.REQUIRE_APPROVAL, 3, tuple(reasons), "cost-understated", cost)

    # 5. One action asking for more than a quarter of the whole budget is worth a second look
    #    even when it fits.
    if cost > budget.limit_cents / 4:
        reasons.append(f"{money(cost)} in one action is unusually large — escalated for review.")
        return Decision(Verdict.REQUIRE_APPROVAL, 3, tuple(reasons), "large-single-spend", cost)

    if rule.tier == 3:
        reasons.append("Tier 3 always waits for a human.")
        return Decision(Verdict.REQUIRE_APPROVAL, 3, tuple(reasons), rule.id, cost)

    if rule.tier == 2:
        # A runaway loop of individually harmless writes is still a runaway. Reads are never slowed.
        recent = history.recent_unattended_writes
        if recent >= UNATTENDED_WRITE_LIMIT:
            reasons.append(
                f"This agent has already run {recent} unattended writes in the last "
                f"{UNATTENDED_WINDOW_SECONDS}s (limit {UNATTENDED_WRITE_LIMIT}), so the next one waits for a person."
            )
            return Decision(Verdict.REQUIRE_APPROVAL, 2, tuple(reasons), "unattended-rate", cost)
        reasons.append("Tier 2 runs unattended but is recorded in full.")
        return Decision(Verdict.ALLOW, 2, tuple(reasons), rule.id, cost)

    return Decision(Verdict.ALLOW, 1, tuple(reasons), rule.id, cost)
