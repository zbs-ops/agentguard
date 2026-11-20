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
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "POLICY",
    "TWO_PERSON_ABOVE_CENTS",
    "UNATTENDED_WINDOW_SECONDS",
    "UNATTENDED_WRITE_LIMIT",
    "Budget",
    "Decision",
    "History",
    "Verdict",
    "approvals_required",
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

# Above this amount one person's yes isn't enough: two different reviewers have to approve before
# the action runs. It's the amount the gate charges, not the agent's estimate, so understating a
# refund doesn't skip the second person. Same value as agent-governance.
TWO_PERSON_ABOVE_CENTS = 25_000


@dataclass(frozen=True)
class Budget:
    """A spend cap, in cents. ``spent_cents`` is what has been charged so far."""

    spent_cents: int
    limit_cents: int


@dataclass(frozen=True)
class History:
    """What the gate knows about recent activity. The caller supplies it; the gate stays pure."""

    recent_unattended_writes: int = 0
    #: An earlier request with the same agent, tool and arguments that is still waiting, was
    #: approved, or already ran within the approval window: its id and state.
    repeat_of: tuple[str, str] | None = None


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


#: JavaScript's ``Number.MAX_SAFE_INTEGER``. Above it agent-governance can't count exactly and
#: refuses the amount, so the port does too.
_MAX_SAFE_INTEGER = 2**53 - 1

#: What JavaScript's ``String.prototype.trim`` removes (WhiteSpace and LineTerminator). Python's
#: ``str.strip()`` also removes U+001C-U+001F and U+0085, which agent-governance keeps.
_JS_SPACE = "\t\n\v\f\r                  　﻿"


def _is_cents(value: object) -> bool:
    """A whole, non-negative amount the budget arithmetic can trust (and not a bool), no larger
    than JavaScript's ``Number.isSafeInteger`` allows."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0 <= value <= _MAX_SAFE_INTEGER
    return isinstance(value, float) and math.isfinite(value) and value.is_integer() and 0 <= value <= _MAX_SAFE_INTEGER


_MAX_ARG_DEPTH = 32
# A name is often one run-together word (``totalamount``, ``unitprice``), so these count inside a
# word. ``value`` and ``sum`` only count whole: inside a word they are usually something else
# (``valueset``, ``summary``, ``consumer``).
_AMOUNT_WORD_PARTS = ("amount", "total", "price")
_AMOUNT_WORDS = frozenset({"value", "sum"})
# ASCII digits only, as JavaScript's ``\d``; Python's ``\d`` also matches other scripts' digits.
_NUMERIC_TEXT = re.compile(r"[-+]?\$?[0-9][0-9,]*(\.[0-9]+)?")
_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_NOT_WORD = re.compile(r"[^a-z0-9]+")
# A JavaScript object lists keys that are array indices first, in ascending order.
_ARRAY_INDEX = re.compile(r"0|[1-9][0-9]*")


def _key_words(key: str) -> list[str]:
    """``amountCents``, ``AMOUNT_CENTS``, ``amount-in-cents`` -> lower-case words."""
    return [w for w in _NOT_WORD.split(_CAMEL.sub(r"\1 \2", key).lower()) if w]


def _is_cents_key(key: str) -> bool:
    """``amount_cents``, ``amountCents``, and run together, ``amountcents``."""
    words = _key_words(key)
    return bool(words) and words[-1].endswith("cents")


def _is_amount_key(key: str) -> bool:
    return any(w in _AMOUNT_WORDS or any(part in w for part in _AMOUNT_WORD_PARTS) for w in _key_words(key))


def _js_string(value: object) -> str:
    """``String(value)`` in JavaScript, for values parsed from JSON, so reasons match word for word."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e21:
        return str(int(value))
    if isinstance(value, (list, tuple)):
        return ",".join("" if v is None else _js_string(v) for v in value)
    if isinstance(value, Mapping):
        return "[object Object]"
    return str(value)


def _js_entries(obj: Mapping[object, object]) -> list[tuple[str, object]]:
    """``Object.entries`` order: array-index keys ascending, then the rest as inserted."""
    items = [(str(k), v) for k, v in obj.items()]
    index = sorted(
        ((k, v) for k, v in items if _ARRAY_INDEX.fullmatch(k) and int(k) < 2**32 - 1), key=lambda kv: int(kv[0])
    )
    rest = [(k, v) for k, v in items if not (_ARRAY_INDEX.fullmatch(k) and int(k) < 2**32 - 1)]
    return index + rest


@dataclass
class _Amounts:
    #: The largest amount the arguments state in cents: one field, or one list of line items summed.
    cents: int = 0
    #: Paths of numbers named like an amount but with no unit.
    unpriced: list[str] = field(default_factory=list)
    #: Why the arguments can't be priced at all, if they can't.
    malformed: str | None = None


def _read_amounts(args: Mapping[str, object]) -> _Amounts:
    """Every amount the arguments state, at any depth, read exactly as agent-governance's
    ``readAmounts`` reads them (including which problem is reported when there are several). A
    field counts as cents when the last word of its name ends in "cents" (``amount_cents``,
    ``amountCents``, ``amountcents``, ``cents``), and a list of objects that each carry one counts
    as their sum, so line items can't be priced at their largest line. Arguments nested past
    ``_MAX_ARG_DEPTH`` are refused rather than half read."""
    out = _Amounts()

    def walk(value: object, path: str, depth: int) -> None:
        if out.malformed is not None:
            return
        if depth > _MAX_ARG_DEPTH:
            out.malformed = f"The arguments are nested more than {_MAX_ARG_DEPTH} levels deep, too deep to price."
            return
        if isinstance(value, (list, tuple)):
            total = 0
            for i, item in enumerate(value):
                walk(item, f"{path}[{i}]", depth + 1)
                if isinstance(item, Mapping):
                    for k, v in _js_entries(item):
                        if _is_cents_key(k) and _is_cents(v):
                            total += int(v)  # type: ignore[call-overload]
            if total > _MAX_SAFE_INTEGER:
                out.malformed = f"The line items under {path} add up to more cents than can be counted."
            else:
                out.cents = max(out.cents, total)
            return
        if not isinstance(value, Mapping):
            return
        for k, v in _js_entries(value):
            at = f"{path}.{k}" if path else k
            if _is_cents_key(k):
                if not _is_cents(v):
                    where = "" if at == "amount_cents" else f" ({at})"
                    out.malformed = (
                        f"The arguments ask to move {_js_string(v)} cents{where}, which is not a whole, non-negative amount."
                    )
                    return
                out.cents = max(out.cents, int(v))  # type: ignore[call-overload]
                continue
            numeric = (isinstance(v, (int, float)) and not isinstance(v, bool)) or (
                isinstance(v, str) and _NUMERIC_TEXT.fullmatch(v.strip(_JS_SPACE)) is not None
            )
            if numeric and _is_amount_key(k):
                out.unpriced.append(at)
            walk(v, at, depth + 1)

    walk(args, "", 0)
    return out


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
    amounts = _read_amounts(args)
    if amounts.malformed is not None:
        reasons.append(amounts.malformed)
        return Decision(Verdict.DENY, rule.tier, tuple(reasons), "malformed-cost", 0)

    # 2b. An amount with no unit can't be priced: ``amount: 300`` is $3.00 to Stripe and $300
    #     to most other APIs. On a tool that moves money, guessing wrong misprices the call a
    #     hundredfold, so the agent is told to say it in cents instead.
    if rule.id == "money-movement" and amounts.unpriced:
        reasons.append(
            f"The arguments name an amount without a unit ({', '.join(amounts.unpriced)}). State what the "
            "call will move in cents, as amount_cents, so it can be priced."
        )
        return Decision(Verdict.DENY, rule.tier, tuple(reasons), "unpriced-amount", 0)

    # The estimate is the agent's word; the cents in the arguments are what the call will do.
    # Charge the larger, so understating the estimate can't buy room under the cap.
    estimate = int(estimated_cost_cents)
    cost = max(estimate, amounts.cents)
    understated = cost > estimate

    # 3. Budget is checked before anything else, because a runaway spend is the failure that
    #    costs real money while everyone is asleep.
    projected = budget.spent_cents + cost
    if projected > budget.limit_cents:
        reasons.append(f"This would put spend at {money(projected)} against a {money(budget.limit_cents)} cap.")
        return Decision(Verdict.DENY, rule.tier, tuple(reasons), "budget-exceeded", cost)

    # 3b. An agent that timed out waiting for an answer often just asks again. If the same
    #     action is already waiting, or already went through, a second copy would pay or contact
    #     the customer twice once both are approved. Only actions that move money or are tier 3
    #     are refused; the caller only reports copies that are still live.
    if history.repeat_of is not None and (rule.tier == 3 or cost > 0):
        first, state = history.repeat_of
        was = "is still waiting for a person" if state == "PENDING_APPROVAL" else "already went through"
        reasons.append(
            f"The same request ({first}) {was}. A repeat would do it twice; to really do it again, "
            "change the arguments (for example a new request id)."
        )
        return Decision(Verdict.DENY, rule.tier, tuple(reasons), "repeated-request", cost)

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


def approvals_required(decision: Decision, above_cents: int | None = TWO_PERSON_ABOVE_CENTS) -> int:
    """How many different people must approve an action the gate sent for review: 2 when what
    it charges is over ``above_cents``, else 1. ``None`` turns the two-person rule off."""
    return 2 if above_cents is not None and decision.cost_cents > above_cents else 1
