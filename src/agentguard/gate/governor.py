"""The approval workflow around the policy gate.

``evaluate`` says what *should* happen to one call. A ``Governor`` makes it happen over time:
it holds the actions waiting for a person, checks who is approving, charges the budget only
when something actually runs, and writes every step to a tamper-evident ledger.

The rules carried over from ``agent-governance``, each one a bug that project fixed:

- The agent that proposed an action can't approve it.
- A decision (approve or reject) needs a token only the named reviewer's key can make, bound
  to that action's contents, not just its id (ids restart with every new Governor). The agent
  that asked may withdraw its own request without one.
- An approval only counts for 24 hours after the request (configurable).
- The budget is checked again at the moment an approved action would run, because other
  actions may have spent while it waited.
- A loop of unattended writes from one agent gets slowed down to "ask a person".
- Above a threshold (``two_person_above_cents``, $250 by default), an action needs two
  different reviewers before it runs. Payments to the same target add up within the approval
  window, so splitting one into smaller ones doesn't avoid the second reviewer.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import time
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agentguard.gate.ledger import ChainVerification, Entry, append_entry, verify_chain
from agentguard.gate.policy import (
    _JS_SPACE,
    TWO_PERSON_ABOVE_CENTS,
    UNATTENDED_WINDOW_SECONDS,
    Budget,
    Decision,
    History,
    Verdict,
    approvals_required,
    evaluate,
    money,
)

__all__ = [
    "APPROVAL_TTL_SECONDS",
    "ApprovalAuthError",
    "GatedAction",
    "Governor",
    "SameApproverError",
    "SelfApprovalError",
    "State",
    "mint_approval_token",
    "proposal_digest",
    "verify_approval_token",
]

APPROVAL_TTL_SECONDS = 24 * 60 * 60


class State(str, Enum):
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    AUTO_ALLOWED = "AUTO_ALLOWED"
    BLOCKED = "BLOCKED"
    EXECUTED = "EXECUTED"


class SelfApprovalError(Exception):
    """The agent that proposed an action tried to approve it."""


class ApprovalAuthError(Exception):
    """An approval arrived without a valid token for the named reviewer."""


class SameApproverError(Exception):
    """The reviewer who gave the first of two required approvals tried to give the second."""


def _norm(name: str) -> str:
    return name.strip().lower()


def _js_number(v: float) -> str:
    """A float the way JavaScript's JSON.stringify writes it."""
    if not math.isfinite(v):
        return "null"
    if v == int(v) and abs(v) < 1e21:
        return str(int(v))  # 1.0 -> "1", -0.0 -> "0"
    text = repr(v)
    if "e" in text:
        mantissa, exp = text.split("e")
        sign = "-" if exp.startswith("-") else "+"
        text = f"{mantissa}e{sign}{exp.lstrip('+-').lstrip('0') or '0'}"
    return text


def _canonical(value: Any) -> str:
    """JSON with object keys sorted at every level: the same bytes as the TypeScript ``canonical``."""
    if value is None or isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _js_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical(x) for x in value) + "]"
    if isinstance(value, Mapping):
        # JavaScript sorts keys by UTF-16 code units, so sort the same way.
        keys = sorted((str(k) for k in value), key=lambda k: k.encode("utf-16-be"))
        return "{" + ",".join(f"{json.dumps(k, ensure_ascii=False)}:{_canonical(value[k])}" for k in keys) + "}"
    raise TypeError(f"can't fingerprint a {type(value).__name__}")


def proposal_digest(
    proposal_id: str, agent: str, tool: str, args: Mapping[str, Any], estimated_cost_cents: int, created_at: str
) -> str:
    """A fingerprint of everything an approval is an approval OF. Same hex as agent-governance's
    ``proposalDigest``. ``created_at`` is the ISO-8601 time the action was proposed."""
    body = {
        "id": proposal_id,
        "agent": agent,
        "tool": tool,
        "args": args,
        "estimatedCostCents": estimated_cost_cents,
        "createdAt": created_at,
    }
    return hashlib.sha256(_canonical(body).encode()).hexdigest()


def mint_approval_token(key: str, proposal_id: str, reviewer: str, *, digest: str) -> str:
    """HMAC that binds one reviewer to one proposal's id AND contents. Same bytes as the
    TypeScript version. The digest matters: ids are a counter that restarts, so a token bound
    to the id alone used to approve a different action that got the same id later."""
    message = f"{proposal_id}:{digest}:{_norm(reviewer)}".encode()
    return hmac.new(key.encode(), message, hashlib.sha256).hexdigest()


def verify_approval_token(
    keys: Mapping[str, str], proposal_id: str, reviewer: str, token: str | None, *, digest: str
) -> bool:
    if token is None:
        return False
    key = {_norm(k): v for k, v in keys.items()}.get(_norm(reviewer))
    if key is None:
        return False
    return hmac.compare_digest(mint_approval_token(key, proposal_id, reviewer, digest=digest), token)


@dataclass
class GatedAction:
    id: str
    agent: str
    tool: str
    args: dict[str, Any]
    estimated_cost_cents: int
    created_at: float
    decision: Decision
    state: State
    reviewed_by: str | None = None
    review_note: str | None = None
    #: Reviewers who have said yes so far, on an action that needs two of them.
    approvals: list[str] | None = None


def _digest_of(action: GatedAction) -> str:
    return proposal_digest(
        action.id, action.agent, action.tool, action.args, action.estimated_cost_cents, _iso(action.created_at)
    )


def _iso(ts: float) -> str:
    # Millisecond ISO-8601 in UTC with a Z, matching JavaScript's Date.toISOString().
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(ts * 1000) % 1000:03d}Z"


@dataclass
class Governor:
    """Run agent tool calls through the gate, a human approval step, and a ledger.

    ``reviewer_keys`` maps each person allowed to approve to their secret. ``clock`` returns
    seconds since the epoch; tests pass a fake one.
    """

    budget_limit_cents: int = 50_000
    reviewer_keys: Mapping[str, str] = field(default_factory=dict)
    approval_ttl_seconds: float = APPROVAL_TTL_SECONDS
    #: Above this many cents an action needs two different reviewers; None turns that off.
    two_person_above_cents: int | None = TWO_PERSON_ABOVE_CENTS
    clock: Callable[[], float] = time.time
    spent_cents: int = 0
    _actions: list[GatedAction] = field(default_factory=list, repr=False)
    _ledger: list[Entry] = field(default_factory=list, repr=False)

    # -- reading state ---------------------------------------------------------------------
    @property
    def budget(self) -> Budget:
        return Budget(spent_cents=self.spent_cents, limit_cents=self.budget_limit_cents)

    def actions(self) -> list[GatedAction]:
        return list(self._actions)

    def pending(self) -> list[GatedAction]:
        return [a for a in self._actions if a.state is State.PENDING_APPROVAL]

    def get(self, action_id: str) -> GatedAction | None:
        return next((a for a in self._actions if a.id == action_id), None)

    def ledger(self) -> list[Entry]:
        return list(self._ledger)

    def verify(self) -> ChainVerification:
        return verify_chain(self._ledger)

    def mint_token(self, action_id: str, reviewer: str) -> str:
        """A token for a reviewer whose key this governor holds (what your UI would call).
        It covers the action as it stands now, so it approves nothing else."""
        key = {_norm(k): v for k, v in self.reviewer_keys.items()}.get(_norm(reviewer))
        if key is None:
            raise ApprovalAuthError(f"{reviewer!r} is not a registered reviewer.")
        action = self.get(action_id)
        if action is None:
            raise KeyError(f"no action {action_id!r}")
        return mint_approval_token(key, action_id, reviewer, digest=_digest_of(action))

    # -- the workflow ----------------------------------------------------------------------
    def propose(
        self, agent: str, tool: str, args: dict[str, Any] | None = None, *, estimated_cost_cents: int = 0
    ) -> GatedAction:
        now = self.clock()
        action_id = f"act_{len(self._actions) + 1:04d}"
        args = dict(args or {})
        self._record("PROPOSED", agent, action_id, {"tool": tool, "estimated": _money_or_raw(estimated_cost_cents)})
        decision = evaluate(
            tool,
            estimated_cost_cents=estimated_cost_cents,
            args=args,
            budget=self.budget,
            history=History(
                recent_unattended_writes=self._recent_unattended_writes(agent, now),
                repeat_of=self._live_copy(agent, tool, args, now),
            ),
        )
        self._record(
            "POLICY_EVALUATED",
            "policy-engine",
            action_id,
            {"verdict": decision.verdict.value, "tier": decision.tier, "decidedBy": decision.decided_by},
        )
        action = GatedAction(action_id, agent, tool, args, estimated_cost_cents, now, decision, State.PENDING_APPROVAL)
        if decision.verdict is Verdict.DENY:
            action.state = State.BLOCKED
            self._record("BLOCKED", "policy-engine", action_id, {"why": decision.reason})
        elif decision.verdict is Verdict.ALLOW:
            action.state = State.AUTO_ALLOWED
            self._execute(action)
        self._actions.append(action)
        return action

    def decide(
        self, action_id: str, approve: bool, reviewer: str, *, note: str | None = None, token: str | None = None
    ) -> GatedAction | None:
        """Approve or reject a pending action. Returns None if there's no such pending action."""
        action = self.get(action_id)
        if action is None or action.state is not State.PENDING_APPROVAL:
            return None
        # Separation of duties, checked before anything is written. Withdrawing your own is fine.
        if approve and _norm(reviewer) == _norm(action.agent):
            raise SelfApprovalError(f"{action.agent!r} proposed this action, so it can't also approve it.")
        # A rejection is authenticated too: otherwise anyone could cancel a pending action and the
        # ledger would record a person's name against a decision they never made. Only the agent
        # that asked may withdraw its own request without a token.
        withdrawing = not approve and _norm(reviewer) == _norm(action.agent)
        if not withdrawing and not verify_approval_token(
            self.reviewer_keys, action_id, reviewer, token, digest=_digest_of(action)
        ):
            raise ApprovalAuthError(f"{reviewer!r} did not present a valid token for this decision.")

        # A large action needs two different people. Refusing a repeat yes here, before anything
        # is written, means one reviewer approving twice changes nothing. "Large" counts what
        # already went to the same target within the approval window, so a $300 refund sent as
        # $200 then $100 still needs two people for the part that crosses the line.
        limit = self.two_person_above_cents
        sent = self._sent_to_same_target(action) if approve and limit is not None else 0
        split = limit is not None and sent > 0 and sent + action.decision.cost_cents > limit
        two_person = approve and (approvals_required(action.decision, limit) == 2 or split)
        split_why = (
            f"{money(sent)} already went to the same target in the last {_hours(self.approval_ttl_seconds)}h; "
            f"with this {money(action.decision.cost_cents)} it passes the {money(limit or 0)} two-person line."
            if split
            else None
        )
        if two_person and any(_norm(r) == _norm(reviewer) for r in action.approvals or []):
            raise SameApproverError(f'"{reviewer}" already approved this. It needs a second, different person.')

        # Only a yes expires: rejecting a stale request is how the queue gets cleared. Checked
        # after authentication, so a forged attempt can't change anything.
        waited = self.clock() - action.created_at
        if approve and waited > self.approval_ttl_seconds:
            why = (
                f"This request expired: it was made {int(waited // 3600)}h ago, and an approval only counts "
                f"within {_hours(self.approval_ttl_seconds)}h. Ask the agent to propose it again."
            )
            action.reviewed_by, action.review_note = reviewer, note
            self._block(action, why, "approval-expired", extra={"reviewer": reviewer})
            return action

        action.reviewed_by, action.review_note = reviewer, note
        if not approve:
            action.state = State.REJECTED
            self._record("REJECTED", reviewer, action_id, {"note": note})
            return action

        if two_person:
            action.approvals = [*(action.approvals or []), reviewer]
            detail: dict[str, Any] = {"note": note, "approval": f"{len(action.approvals)} of 2"}
            if split_why:
                detail["why"] = split_why
            if len(action.approvals) < 2:
                # The first yes is on the record, but the action keeps waiting for a second person.
                self._record("APPROVED", reviewer, action_id, detail)
                return action
            action.state = State.APPROVED
            self._record("APPROVED", reviewer, action_id, detail)
        else:
            action.state = State.APPROVED
            self._record("APPROVED", reviewer, action_id, {"note": note})
        # The cap was checked at proposal time, but other actions may have spent since.
        projected = self.spent_cents + action.decision.cost_cents
        if projected > self.budget_limit_cents:
            why = (
                f"Budget changed while this waited: {money(action.decision.cost_cents)} would bring spend to "
                f"{money(projected)}, over the {money(self.budget_limit_cents)} cap."
            )
            self._block(action, why, "budget-exceeded-at-approval")
            return action
        self._execute(action)
        return action

    # -- internals -------------------------------------------------------------------------
    def _live_copy(self, agent: str, tool: str, args: dict[str, Any], now: float) -> tuple[str, str] | None:
        """An earlier request for exactly this action that is still live: waiting, approved or
        already run, within the approval window. Rejected, blocked and expired ones don't count."""
        key = _canonical({"agent": agent, "tool": tool, "args": args})
        live = {State.PENDING_APPROVAL, State.APPROVED, State.EXECUTED, State.AUTO_ALLOWED}
        for a in self._actions:
            if (
                a.state in live
                and now - a.created_at < self.approval_ttl_seconds
                and _canonical({"agent": a.agent, "tool": a.tool, "args": a.args}) == key
            ):
                return a.id, a.state.value
        return None

    def _sent_to_same_target(self, action: GatedAction) -> int:
        """What has already been paid out to the same target as ``action`` within the approval
        window, by any agent: actions that ran and share an order, invoice, customer, account or
        recipient with it (``TARGET_KINDS``), whatever tool or other details they used. With no
        such identifier, only an exact repeat of the tool and the other arguments counts. Waiting,
        rejected and blocked actions moved no money and don't count."""
        mine = _target_ids(action.args)

        def exact(a: GatedAction) -> str:
            rest = {k: v for k, v in a.args.items() if k != "amount_cents"}
            return _canonical({"tool": a.tool, "args": rest})

        key = exact(action)

        def same_target(a: GatedAction) -> bool:
            return exact(a) == key if not mine else bool(mine & _target_ids(a.args))

        return sum(
            a.decision.cost_cents
            for a in self._actions
            if a is not action
            and a.state is State.EXECUTED
            and a.decision.cost_cents > 0
            and action.created_at - a.created_at < self.approval_ttl_seconds
            and same_target(a)
        )

    def _recent_unattended_writes(self, agent: str, now: float) -> int:
        return sum(
            1
            for a in self._actions
            if a.agent == agent
            and a.decision.verdict is Verdict.ALLOW
            and a.decision.tier == 2
            and now - a.created_at < UNATTENDED_WINDOW_SECONDS
        )

    def _execute(self, action: GatedAction) -> None:
        # Spend is counted only when something runs, and it's the gate's cost, not the estimate.
        self.spent_cents += action.decision.cost_cents
        action.state = State.EXECUTED
        self._record("EXECUTED", "runtime", action.id, {"tool": action.tool, "cost": money(action.decision.cost_cents)})

    def _block(self, action: GatedAction, why: str, decided_by: str, extra: dict[str, Any] | None = None) -> None:
        action.state = State.BLOCKED
        action.decision = replace(
            action.decision, verdict=Verdict.DENY, reasons=(*action.decision.reasons, why), decided_by=decided_by
        )
        self._record("BLOCKED", "policy-engine", action.id, {"why": why, **(extra or {})})

    def _record(self, event: str, actor: str, action_id: str, detail: dict[str, Any]) -> None:
        append_entry(self._ledger, at=_iso(self.clock()), event=event, actor=actor, proposal_id=action_id, detail=detail)


def _money_or_raw(cents: object) -> object:
    # A malformed estimate is logged as given; the gate refuses it a moment later.
    if isinstance(cents, int) and not isinstance(cents, bool):
        return money(cents)
    return repr(cents)


#: What a payment can go to. Payments that share one of these (compared without case or
#: surrounding spaces) go to the same target for the two-person rule, so a reason, a note, a
#: request id or a different refund tool doesn't make a split look separate. An argument names one
#: when its name, read as words without a trailing id word, is on this list (see ``target_kind``).
TARGET_KINDS = (
    "order",
    "invoice",
    "customer",
    "account",
    "charge",
    "paymentintent",
    "subscription",
    "user",
    "to",
    "recipient",
    "payee",
    "email",
    # Where a payout goes. Left out on purpose, because they'd add up payments to different people:
    # ``client`` (``client_id`` is usually the app's own credential), ``routing`` (a bank, not an account)
    # and a bare ``address`` (often a postal address; ``wallet_address`` still counts).
    "vendor",
    "merchant",
    "supplier",
    "beneficiary",
    "employee",
    "contractor",
    "counterparty",
    "member",
    "destination",
    "iban",
    "card",
    "wallet",
    "phone",
    "paypal",
)

#: Kinds that say who gets paid. They share one namespace, so the same value under any two of them
#: (``to`` then ``recipient``, ``payee`` then ``vendor``, an IBAN then a bank account) is the same
#: target. The other kinds (order, invoice, charge, payment intent, subscription) say what a payment
#: is for and keep one namespace each, so order 42 and invoice 42 stay apart.
_PAYEE_KINDS = frozenset(
    {
        *("customer", "account", "user", "to", "recipient", "payee", "email", "vendor", "merchant"),
        *("supplier", "beneficiary", "employee", "contractor", "counterparty", "member", "destination"),
        *("iban", "card", "wallet", "phone", "paypal"),
    }
)
#: Kinds whose values are numbers people format with spaces, dashes, dots or brackets.
_NUMBER_KINDS = frozenset({"account", "destination", "iban", "card", "phone"})

#: Words after a target's name that still name it: ``order_id``, ``orderNumber``, ``vendor_name``.
_ID_WORDS = frozenset({"id", "number", "no", "num", "ref", "reference", "code", "address", "name"})
_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_NOT_WORD = re.compile(r"[^a-z0-9]+")


def target_kind(name: str) -> str | None:
    """The kind of target an argument name names, as agent-governance's ``targetKind`` reads it:
    split into words (snake_case, kebab-case, dotted or camelCase), trailing id words dropped, and
    the rest joined (``paymentIntentId`` → ``paymentintent``) or its last word must be a kind.
    ``order``, ``order_id``, ``orderId``, ``ORDER_ID`` and ``orderNumber`` all name the order."""
    words = [w for w in _NOT_WORD.split(_CAMEL.sub(r"\1 \2", name).lower()) if w]
    while len(words) > 1 and words[-1] in _ID_WORDS:
        words.pop()
    joined = "".join(words)
    if joined in TARGET_KINDS:
        return joined
    return words[-1] if words and words[-1] in TARGET_KINDS else None


_EMAIL_TAG = re.compile(r"\+[^@]*@")
# JavaScript's ``\s`` is the same set as its trim, which differs from Python's ``\s``.
_NUMBER_FORMATTING = re.compile("[" + re.escape(_JS_SPACE) + r".()-]")
_EMAIL_SHAPE = re.compile("^[^@" + re.escape(_JS_SPACE) + "]+@[^@" + re.escape(_JS_SPACE) + "]+$")


def _id_value(value: str, kind: str) -> str:
    """An identifier as agent-governance compares it: NFKC-normalised (a full-width "２" is "2"),
    invisible format characters (category Cf) dropped, every dash (Pd) read as "-", trimmed as
    JavaScript trims and lower-cased. Account, card and phone numbers lose their spaces, dashes, dots
    and brackets; a payee value shaped like an email address loses its ``+tag`` (it reaches the same
    mailbox)."""
    text = unicodedata.normalize("NFKC", value)
    text = "".join("-" if unicodedata.category(c) == "Pd" else c for c in text if unicodedata.category(c) != "Cf")
    text = text.strip(_JS_SPACE).lower()
    if kind in _NUMBER_KINDS:
        return _NUMBER_FORMATTING.sub("", text)
    return _EMAIL_TAG.sub("@", text, count=1) if kind in _PAYEE_KINDS and _EMAIL_SHAPE.match(text) else text


def _namespace(kind: str) -> str:
    return "payee" if kind in _PAYEE_KINDS else kind


def _target_ids(args: Mapping[str, Any]) -> set[str]:
    """``kind=value`` for each target identifier in some arguments, normalised the way
    agent-governance does: strings trimmed and lower-cased, finite numbers as JavaScript prints them.
    Nested objects and lists are read too, to the same depth (32), and inside a target's own object
    an ``id`` names that target (``{"customer": {"id": "acme"}}``)."""
    ids: set[str] = set()

    def walk(value: Any, kind: str | None, depth: int) -> None:
        if depth > 32:
            return
        if isinstance(value, (list, tuple)):
            for v in value:
                walk(v, kind, depth + 1)
        elif isinstance(value, Mapping):
            for k, v in value.items():
                name = str(k)
                walk(v, target_kind(name) or (kind if name.lower() == "id" else None), depth + 1)
        elif kind and isinstance(value, str) and _id_value(value, kind):
            ids.add(f"{_namespace(kind)}={_id_value(value, kind)}")
        elif kind and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            ids.add(f"{_namespace(kind)}={int(value) if float(value).is_integer() else value}")

    walk(args, None, 0)
    return ids


def _hours(seconds: float) -> str:
    hours = seconds / 3600
    return str(int(hours)) if hours.is_integer() else str(hours)
