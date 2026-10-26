"""A tamper-evident audit ledger.

Every entry carries the hash of the entry before it, so the ledger is a chain. Change one
field in one old entry and its hash stops matching, and every entry after it is orphaned. You
can append to history, but you can't quietly rewrite it.

Entries serialize with the same field names and the same hashed bytes as the TypeScript
``agent-governance`` project, so a ledger written by either one verifies in the other.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["GENESIS", "ChainVerification", "Entry", "append_entry", "hash_entry", "load_entries", "verify_chain"]

GENESIS = "GENESIS"

EVENTS = ("PROPOSED", "POLICY_EVALUATED", "APPROVED", "REJECTED", "EXECUTED", "BLOCKED")


@dataclass
class Entry:
    """One line in the ledger. ``hash`` is SHA-256 over every other field."""

    seq: int
    at: str
    event: str
    actor: str
    proposal_id: str
    detail: dict[str, Any] = field(default_factory=dict)
    prev_hash: str = GENESIS
    hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "at": self.at,
            "event": self.event,
            "actor": self.actor,
            "proposalId": self.proposal_id,
            "detail": self.detail,
            "prevHash": self.prev_hash,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Entry:
        return cls(
            seq=data["seq"],
            at=data["at"],
            event=data["event"],
            actor=data["actor"],
            proposal_id=data["proposalId"],
            detail=dict(data["detail"]),
            prev_hash=data["prevHash"],
            hash=data["hash"],
        )


@dataclass(frozen=True)
class ChainVerification:
    valid: bool
    length: int
    #: The first sequence number where the chain stops adding up.
    broken_at: int | None
    message: str


def _canonical(entry: Entry) -> str:
    # The same bytes JavaScript's JSON.stringify produces for this array. Field order is fixed:
    # change it and every ledger ever written stops verifying, which is the point.
    return json.dumps(
        [entry.seq, entry.at, entry.event, entry.actor, entry.proposal_id, entry.detail, entry.prev_hash],
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def hash_entry(entry: Entry) -> str:
    return hashlib.sha256(_canonical(entry).encode("utf-8")).hexdigest()


def append_entry(
    chain: list[Entry], *, at: str, event: str, actor: str, proposal_id: str, detail: dict[str, Any]
) -> Entry:
    """Seal a new entry onto the end of ``chain`` and return it."""
    if event not in EVENTS:
        raise ValueError(f"unknown ledger event {event!r}")
    previous = chain[-1] if chain else None
    entry = Entry(
        seq=len(chain) + 1,
        at=at,
        event=event,
        actor=actor,
        proposal_id=proposal_id,
        detail=detail,
        prev_hash=previous.hash if previous else GENESIS,
    )
    entry.hash = hash_entry(entry)
    chain.append(entry)
    return entry


def verify_chain(chain: Iterable[Entry]) -> ChainVerification:
    """Recompute every hash. Reports the first entry that doesn't add up, not just that one doesn't."""
    entries = list(chain)
    expected_prev = GENESIS
    for entry in entries:
        if entry.prev_hash != expected_prev:
            return ChainVerification(
                False,
                len(entries),
                entry.seq,
                f"Entry #{entry.seq} points at the wrong parent. The chain was cut or reordered here.",
            )
        if hash_entry(entry) != entry.hash:
            return ChainVerification(
                False,
                len(entries),
                entry.seq,
                f"Entry #{entry.seq} was edited after it was written. Its contents no longer match its hash.",
            )
        expected_prev = entry.hash
    message = "Empty ledger." if not entries else f"All {len(entries)} entries verify against their hashes."
    return ChainVerification(True, len(entries), None, message)


def load_entries(data: Any) -> list[Entry]:
    """Read a ledger from parsed JSON: a list of entries, or an object with a ``chain`` list."""
    if isinstance(data, Mapping):
        data = data.get("chain")
    if not isinstance(data, list):
        raise ValueError("a ledger is a JSON list of entries (or an object with a 'chain' list)")
    try:
        return [Entry.from_dict(item) for item in data]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"ledger entry is missing a field: {exc}") from exc
