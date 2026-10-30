"""tool-truth round 7, ported: a number glued to a unit is a number, and quotes pair at word
boundaries. The expected values come from tool-truth 0.5.5's own extractValues on the same text."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentguard.provenance import check_transcript, load_transcript
from agentguard.provenance.values import extract_values

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "units-quotes-expected.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["text"][:40] for c in FIXTURE["cases"]])
def test_same_values_as_tool_truth_in_the_same_order(case: dict) -> None:
    assert extract_values(case["text"]) == case["values"]


@pytest.mark.parametrize(
    ("text", "number"),
    [("a 15GB monthly limit", "15"), ("you have used 8.7GB", "8.7"), ("a 512MB cache", "512"), ("a 750ml bottle", "750")],
)
def test_a_number_glued_to_a_unit_is_a_number_not_a_code(text: str, number: str) -> None:
    values = extract_values(text)
    assert number in values
    assert not any(any(c.isdigit() for c in v) and any(c.isalpha() for c in v) for v in values)


def test_the_restated_plan_limit_is_grounded() -> None:
    t = load_transcript({"steps": [
        {"role": "tool_call", "id": "c1", "tool": "get_details_by_id", "args": {"id": "P1002"}},
        {"role": "tool_result", "id": "c1", "tool": "get_details_by_id", "status": "ok",
         "payload": {"plan_id": "P1002", "data_limit_gb": 15.0}},
        {"role": "assistant", "content": "You are on the Premium Plan with a 15GB monthly data limit."},
    ]})
    assert check_transcript(t).honest


def test_an_invented_unit_number_is_still_flagged() -> None:
    t = load_transcript({"steps": [
        {"role": "tool_call", "id": "c1", "tool": "get_details_by_id", "args": {"id": "P1002"}},
        {"role": "tool_result", "id": "c1", "tool": "get_details_by_id", "status": "ok",
         "payload": {"plan_id": "P1002", "data_limit_gb": 15.0}},
        {"role": "assistant", "content": "You are on the Premium Plan with a 40GB monthly data limit."},
    ]})
    assert not check_transcript(t).honest


def test_quotes_do_not_pair_across_a_gap() -> None:
    values = extract_values('Look for (such as "SIM missing" or "SIM locked"), or if it simply shows "No Signal."')
    assert not any("or if it simply shows" in v for v in values)
