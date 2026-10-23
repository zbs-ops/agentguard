"""Differential checks against the TypeScript tool-truth, on its own example logs.

Expected values are what the current tool-truth CLI (main, f1e62f5) prints for the same files,
in the same order. They also pin determinism: flag values come out in insertion order, not in
Python's per-process set order.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agentguard.provenance import check_transcript, to_transcript

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

EXPECTED = {
    "ok-but-empty.json": ("fabrication-after-failure", ("10-24", "2026-10-24", "2026")),
    "openai-log.json": ("fabrication-after-failure", ("2310.4",)),
    "anthropic-log.json": ("wrong-entity", ("10-18", "2026-10-18", "2026")),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_same_flag_as_typescript(name: str) -> None:
    v = check_transcript(to_transcript(json.loads((EXAMPLES / name).read_text())))
    kind, values = EXPECTED[name]
    assert v.flags[0].kind == kind
    assert v.flags[0].values == values


def test_output_is_identical_across_hash_seeds() -> None:
    code = (
        "import json,sys;from agentguard.provenance import check_transcript,to_transcript;"
        "v=check_transcript(to_transcript(json.load(open(sys.argv[1]))));print(v.flags[0].values)"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", code, str(EXAMPLES / "ok-but-empty.json")],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in range(6)
    }
    assert len(outs) == 1
