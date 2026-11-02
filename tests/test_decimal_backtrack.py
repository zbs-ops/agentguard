"""tool-truth 0.5.6, ported: a decimal glued to a unit is never cut short to its whole part.

"8.7GB" used to yield "8.7" and a stray "8": the number pattern matched 8.7, hit the G, and
backtracked to 8, which is followed by "." and passed. Same cases as tool-truth's test.
"""

from __future__ import annotations

import pytest

from agentguard.provenance.values import extract_values


@pytest.mark.parametrize(
    ("text", "expected"),
    [("you have used 8.7GB", ["8.7"]), ("it took 12.5ms", ["12.5"]), ("a 2.5TB disk", ["2.5"])],
)
def test_a_decimal_is_never_cut_short(text: str, expected: list[str]) -> None:
    assert extract_values(text) == expected


@pytest.mark.parametrize(
    ("text", "value"),
    [("the total is 8.", "8"), ("costs $12.50 today", "12.5"), ("ratio 8.7 to 1", "8.7"), ("pi is 3.14.", "3.14")],
)
def test_ordinary_numbers_are_unchanged(text: str, value: str) -> None:
    assert value in extract_values(text)
