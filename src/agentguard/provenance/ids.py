"""The high-confidence subset of a verdict: flags about an invented identifier."""

from __future__ import annotations

from agentguard.provenance.types import Flag, Verdict
from agentguard.provenance.values import is_id_shaped

__all__ = ["id_flags"]


def id_flags(verdict: Verdict) -> list[Flag]:
    """Keep only flags about an ungrounded order number, item ID, booking reference or user ID,
    each narrowed to the IDs.

    On real agent logs most flags are about amounts, units and dates, where a mismatch is often
    just a formatting difference. An invented ID has no formatting variants and is the
    fabrication that does the most harm: the customer writes it down and the agent may act on
    it. Use this when you want a short, high-yield list, and the full verdict for coverage.
    """
    out: list[Flag] = []
    for flag in verdict.flags:
        ids = tuple(v for v in flag.values if is_id_shaped(v))
        if ids:
            out.append(Flag(kind=flag.kind, message=flag.message, values=ids))
    return out
