"""Provenance checking: is an agent's answer backed by what its tools returned?"""

from __future__ import annotations

from agentguard.provenance.check import check_transcript, check_turns
from agentguard.provenance.failure import acknowledges_failure, classify_result
from agentguard.provenance.ids import id_flags
from agentguard.provenance.types import (
    Flag,
    FlagKind,
    Step,
    Transcript,
    TurnVerdict,
    Verdict,
    load_transcript,
)
from agentguard.provenance.values import (
    extract_values,
    is_id_shaped,
    normalize_number,
    payload_values,
)

__all__ = [
    "Flag",
    "FlagKind",
    "Step",
    "Transcript",
    "TurnVerdict",
    "Verdict",
    "acknowledges_failure",
    "check_transcript",
    "check_turns",
    "classify_result",
    "extract_values",
    "id_flags",
    "is_id_shaped",
    "load_transcript",
    "normalize_number",
    "payload_values",
]
