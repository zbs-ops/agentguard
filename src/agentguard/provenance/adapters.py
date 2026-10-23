"""Read real agent logs: OpenAI chat messages and Anthropic messages.

Maps either format onto the checker's transcript shape, so you can point it at what your agent
actually produced. Ported from tool-truth's ``adapters.ts``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agentguard.provenance.types import Step, Transcript, load_transcript

__all__ = ["from_anthropic", "from_openai", "strip_tool_markup", "to_transcript"]


def _parse_payload(content: Any) -> Any:
    """Tool outputs usually arrive as strings; JSON is parsed so its fields can be checked."""
    if not isinstance(content, str):
        return content
    s = content.strip()
    if not (s.startswith("{") or s.startswith("[")):
        return content
    try:
        return json.loads(s)
    except ValueError:
        return content


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _tag(name: str) -> re.Pattern[str]:
    # Tags may carry a namespace prefix (``<ns:invoke ...>``), so it's matched optionally.
    return re.compile(rf"<(?:[\w-]+:)?{name}\b[\s\S]*?(</(?:[\w-]+:)?{name}>|$)")


_MARKUP = [_tag("function_calls"), _tag("invoke"), _tag("tool_call")]


def strip_tool_markup(text: str) -> str:
    """Remove tool calls a model also wrote into its message text, e.g. Claude's
    ``<function_calls><invoke name=...>``. That's an action, not a statement to the user, and the
    structured call is already captured, so tool names aren't read as invented claims."""
    for pattern in _MARKUP:
        text = pattern.sub(" ", text)
    return text


def from_openai(messages: list[dict[str, Any]]) -> Transcript:
    """OpenAI chat-completions messages (``tool_calls`` and role ``tool``)."""
    steps: list[Step] = []
    tool_names: dict[str, str] = {}
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role in ("system", "developer") and isinstance(content, str):
            steps.append(Step(role="system", content=content))
        elif role == "user" and isinstance(content, str):
            steps.append(Step(role="user", content=content))
        elif role == "assistant":
            for call in m.get("tool_calls") or []:
                fn = call.get("function") or {}
                cid = str(call.get("id", ""))
                tool = str(fn.get("name", "unknown"))
                tool_names[cid] = tool
                steps.append(Step(role="tool_call", id=cid, tool=tool, args=_parse_args(fn.get("arguments"))))
            said = strip_tool_markup(content) if isinstance(content, str) else ""
            if said.strip():
                steps.append(Step(role="assistant", content=said))
        elif role == "tool":
            cid = str(m.get("tool_call_id", ""))
            steps.append(Step(role="tool_result", id=cid, tool=tool_names.get(cid, "unknown"), payload=_parse_payload(content)))
    return Transcript(steps=steps)


def from_anthropic(messages: list[dict[str, Any]]) -> Transcript:
    """Anthropic messages API, with content as a string or as blocks."""
    steps: list[Step] = []
    tool_names: dict[str, str] = {}
    for m in messages:
        content = m.get("content")
        blocks: list[dict[str, Any]] = (
            [{"type": "text", "text": content}] if isinstance(content, str) else content if isinstance(content, list) else []
        )
        if m.get("role") == "user":
            for b in blocks:
                if b.get("type") == "text" and isinstance(b.get("text"), str):
                    steps.append(Step(role="user", content=b["text"]))
                if b.get("type") == "tool_result":
                    tid = str(b.get("tool_use_id", ""))
                    inner = b.get("content")
                    if isinstance(inner, list):
                        inner = "".join(x.get("text", "") if isinstance(x.get("text"), str) else "" for x in inner)
                    steps.append(
                        Step(
                            role="tool_result",
                            id=tid,
                            tool=tool_names.get(tid, "unknown"),
                            status="error" if b.get("is_error") is True else "ok",
                            payload=_parse_payload(inner),
                        )
                    )
        elif m.get("role") == "assistant":
            for b in blocks:
                if b.get("type") == "tool_use":
                    tid = str(b.get("id", ""))
                    tool = str(b.get("name", "unknown"))
                    tool_names[tid] = tool
                    steps.append(Step(role="tool_call", id=tid, tool=tool, args=_parse_args(b.get("input"))))
                elif b.get("type") == "text" and isinstance(b.get("text"), str) and b["text"].strip():
                    steps.append(Step(role="assistant", content=b["text"]))
    return Transcript(steps=steps)


def to_transcript(data: Any) -> Transcript:
    """Accept any of the three shapes: agentguard's own ``{"steps": [...]}``, an OpenAI message
    list, or an Anthropic message list (bare, or under ``"messages"``)."""
    if isinstance(data, dict) and isinstance(data.get("steps"), list):
        return load_transcript(data)
    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict) and isinstance(data.get("messages"), list):
        messages = data["messages"]
    else:
        messages = []
    looks_anthropic = any(
        isinstance(m.get("content"), list)
        and any(isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result") for b in m["content"])
        for m in messages
    )
    return from_anthropic(messages) if looks_anthropic else from_openai(messages)
