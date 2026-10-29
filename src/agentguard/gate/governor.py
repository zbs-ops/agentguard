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
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agentguard.gate.ledger import ChainVerification, Entry, append_entry, verify_chain
from agentguard.gate.policy import UNATTENDED_WINDOW_SECONDS, Budget, Decision, History, Verdict, evaluate, money

__all__ = [
    "APPROVAL_TTL_SECONDS",
    "ApprovalAuthError",
    "GatedAction",
    "Governor",
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
            history=History(recent_unattended_writes=self._recent_unattended_writes(agent, now)),
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


def _hours(seconds: float) -> str:
    hours = seconds / 3600
    return str(int(hours)) if hours.is_integer() else str(hours)
