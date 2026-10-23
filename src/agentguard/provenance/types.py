"""The small transcript shape the checker reads, plus its verdict and flag types.

The shape is deliberately small so any framework's log can be mapped onto it: what the user
asked, which tools were called, what came back, and what the agent finally said. Use
:func:`load_transcript` to build one from plain dicts (e.g. parsed JSON).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["Flag", "FlagKind", "Step", "Transcript", "TurnVerdict", "Verdict", "load_transcript"]

Role = Literal["system", "user", "tool_call", "tool_result", "assistant"]
FlagKind = Literal["unsupported-value", "fabrication-after-failure", "fabricated-refusal", "wrong-entity"]


@dataclass(frozen=True)
class Step:
    """One step of a transcript. Only the fields a role needs are set:

    - ``system`` / ``user`` / ``assistant``: ``content`` is text.
    - ``tool_call``: ``tool``, ``id``, ``args``.
    - ``tool_result``: ``tool``, ``id``, ``payload``, and optional ``status`` ("ok"/"error").
    """

    role: Role
    content: str = ""
    id: str = ""
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    payload: Any = None
    status: Literal["ok", "error"] | None = None


@dataclass(frozen=True)
class Transcript:
    steps: list[Step]
    id: str | None = None


@dataclass(frozen=True)
class Flag:
    """One thing the checker found, safe to show a human reviewer."""

    kind: FlagKind
    message: str
    values: tuple[str, ...] = ()


@dataclass(frozen=True)
class Verdict:
    """``honest`` is True when nothing was flagged."""

    honest: bool
    flags: tuple[Flag, ...] = ()


@dataclass(frozen=True)
class TurnVerdict(Verdict):
    """A per-turn verdict from :func:`check_turns`, carrying where the turn was."""

    index: int = -1
    content: str = ""


def load_transcript(data: dict[str, Any] | list[dict[str, Any]]) -> Transcript:
    """Build a :class:`Transcript` from plain dicts, e.g. ``json.load(...)``. Accepts either a
    ``{"steps": [...]}`` object or a bare list of step dicts."""
    steps_raw = data["steps"] if isinstance(data, dict) else data
    steps = [
        Step(
            role=s["role"],
            content=s.get("content", ""),
            id=s.get("id", ""),
            tool=s.get("tool", ""),
            args=s.get("args", {}) or {},
            payload=s.get("payload"),
            status=s.get("status"),
        )
        for s in steps_raw
    ]
    tid = data.get("id") if isinstance(data, dict) else None
    return Transcript(steps=steps, id=tid)
