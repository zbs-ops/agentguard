"""The policy gate: decide whether an agent's tool call may run, then run the approval workflow."""

from __future__ import annotations

from agentguard.gate.governor import (
    APPROVAL_TTL_SECONDS,
    ApprovalAuthError,
    GatedAction,
    Governor,
    SelfApprovalError,
    State,
    mint_approval_token,
    verify_approval_token,
)
from agentguard.gate.ledger import ChainVerification, Entry, load_entries, verify_chain
from agentguard.gate.policy import (
    POLICY,
    UNATTENDED_WINDOW_SECONDS,
    UNATTENDED_WRITE_LIMIT,
    Budget,
    Decision,
    History,
    Verdict,
    evaluate,
    money,
)

__all__ = [
    "APPROVAL_TTL_SECONDS",
    "POLICY",
    "UNATTENDED_WINDOW_SECONDS",
    "UNATTENDED_WRITE_LIMIT",
    "ApprovalAuthError",
    "Budget",
    "ChainVerification",
    "Decision",
    "Entry",
    "GatedAction",
    "Governor",
    "History",
    "SelfApprovalError",
    "State",
    "Verdict",
    "evaluate",
    "load_entries",
    "mint_approval_token",
    "money",
    "verify_approval_token",
    "verify_chain",
]
