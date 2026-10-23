"""Pull the concrete, checkable values out of text, and out of tool payloads.

An agent can be caught inventing a *value* (a price, a date, an order number) but not a vague
phrase, so this extracts numbers, dates, times, ID-like tokens and quoted strings and leaves
the rest alone. That boundary is the honest limit of a deterministic checker. Ported from the
TypeScript ``tool-truth`` project; the normalization rules were each added in response to a
false alarm on real agent logs.
"""

from __future__ import annotations

import math
import re

__all__ = ["extract_values", "is_id_shaped", "normalize_number", "payload_values"]

_MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
)


def _js_number_string(n: float) -> str:
    """Mimic JS ``String(Number(x))``: ``1024.0`` -> ``"1024"``, ``1024.5`` -> ``"1024.5"``."""
    if math.isfinite(n) and n == int(n) and abs(n) < 1e16:
        return str(int(n))
    return repr(n)


def normalize_number(raw: str) -> str | None:
    """``"$1,024.50"`` -> ``"1024.5"``; ``"45.3%"`` -> ``"45.3"``. None if it isn't a number."""
    cleaned = re.sub(r"[$€£%,\s]", "", raw)
    if cleaned in ("", "-"):
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return _js_number_string(value)


def is_id_shaped(value: str) -> bool:
    """Does this look like an identifier (order number, item ID, user ID) rather than an
    amount, date, unit or phrase? An ID has no formatting variants, so an ungrounded one is far
    more likely invented than a badly normalized ``"750 ml"``."""
    if re.fullmatch(r"\d{6,}", value):
        return True
    if not re.fullmatch(r"[A-Za-z0-9_]{5,}", value):
        return False
    if not re.search(r"\d", value) or not re.search(r"[A-Za-z]", value):
        return False
    if re.fullmatch(r"\d+[A-Za-z]{1,3}", value):
        return False
    # A word with one digit on it is a field name (``address1``); codes like ``JG7FMM`` are
    # all capitals.
    return len(re.findall(r"\d", value)) >= 2 or not re.search(r"[a-z]", value)


def _is_trivial(n: str) -> bool:
    """Small counting words ("step 1", "2 options") aren't worth flagging."""
    value = float(n)
    return value.is_integer() and abs(value) <= 3


def _month_number(name: str) -> str:
    return str(_MONTH_NAMES.index(name[:3].lower()) + 1).zfill(2)


def _add_date(found: dict[str, None], year: str | None, month: str, day: str) -> None:
    """Record a date as its ISO form (when the year is known), plus a month-day token and the
    year alone, so "May 19" and "May 19, 2024" both meet a tool's 2024-05-19."""
    md = f"{month}-{day.zfill(2)}"
    found[md] = None
    if year:
        found[f"{year}-{md}"] = None
        found[year] = None


_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_MONTH_DAY = re.compile(
    rf"\b({_MONTHS})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s*[-–]\s*(\d{{1,2}})(?:st|nd|rd|th)?)?(?:,?\s+(\d{{4}}))?\b",
    re.IGNORECASE,
)
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTHS})\.?(?:,?\s+(\d{{4}}))?\b", re.IGNORECASE)
_TIME = re.compile(r"(?<![\d:])(\d{1,2})(?::(\d{2}))?(?::\d{2})?\s?(am|pm|AM|PM|a\.m\.|p\.m\.)?(?![\d])")
_EXAMPLE_CUE = re.compile(
    r"\b((?:(?:like|e\.g\.|eg\.|such as|for example|for instance|format(?: of)?|looks like|similar to|example:?)\s*)+)"
    r"[:,]?\s*['\"‘“`«#]?\s*[A-Za-z0-9_#-]+\s*['\"’”`»]?"
    r"(?:\s*(?:or|,)\s*['\"‘“`«#]?\s*[A-Za-z0-9_#-]+\s*['\"’”`»]?)*",
    re.IGNORECASE,
)
_HEDGED = re.compile(
    r"\b(about|around|approximately|approx\.?|roughly|nearly|almost|over|above|under|below|up to|"
    r"at least|at most|more than|less than|~)\s*\$?\d[\d,]*(?:\.\d+)?(?:\s*(?:to|-|–|and)\s*\$?\d[\d,]*(?:\.\d+)?)?",
    re.IGNORECASE,
)
_LIST_MARKER = re.compile(r"(^|\n|\s)(\d{1,3})[.)]\s")
_SEP_ID = re.compile(r"\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+\b")
_RUN_ID = re.compile(r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{4,}\b")
_QUOTED = re.compile(r'"([^"\n]{2,80})"')
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.-])[$€£]?\s?-?\d[\d,]*(?:\.\d+)?%?(?![A-Za-z0-9_])")


def extract_values(text: str) -> list[str]:
    """Every concrete value stated in ``text``, normalized so the same fact in different forms
    ("May 19, 2024" and ``2024-05-19``; "$1,024" and ``1024``) comes out as the same token."""
    # An insertion-ordered dict, like a JS Set: values come out in the order they were found,
    # so the same text always gives the same list (a Python set's order varies by process).
    found: dict[str, None] = {}

    # Markdown list markers ("4. **Kettle**") are numbering, not facts, unless a month precedes
    # the number ("on May 21. 3. Modify"), where the "21." is a date, not a marker.
    def _strip_marker(m: re.Match[str]) -> str:
        lead = m.group(1)
        window = text[max(0, m.start() - 12) : m.start() + len(lead)]
        if re.search(rf"\b({_MONTHS})\.?\s*$", window, re.IGNORECASE):
            return m.group(0)
        return lead

    prose = _LIST_MARKER.sub(_strip_marker, text)

    def _iso(m: re.Match[str]) -> str:
        _add_date(found, m.group(1), m.group(2), m.group(3))
        return " "

    prose = _ISO_DATE.sub(_iso, prose)

    def _month_day(m: re.Match[str]) -> str:
        _add_date(found, m.group(4), _month_number(m.group(1)), m.group(2))
        if m.group(3):
            _add_date(found, m.group(4), _month_number(m.group(1)), m.group(3))
        return " "

    prose = _MONTH_DAY.sub(_month_day, prose)

    def _day_month(m: re.Match[str]) -> str:
        _add_date(found, m.group(3), _month_number(m.group(2)), m.group(1))
        return " "

    prose = _DAY_MONTH.sub(_day_month, prose)

    def _time(m: re.Match[str]) -> str:
        minute, ap = m.group(2), m.group(3)
        if minute is None and ap is None:
            return m.group(0)  # a plain number, not a time
        hour = int(m.group(1))
        if hour > 23 or (minute is not None and int(minute) > 59):
            return m.group(0)
        pm = ap is not None and ap.lower().startswith("p")
        am = ap is not None and ap.lower().startswith("a")
        if pm and hour < 12:
            hour += 12
        if am and hour == 12:
            hour = 0
        found[f"{hour}:{minute or '00'}"] = None
        return " "

    prose = _TIME.sub(_time, prose)

    # Illustrations are not claims: "a code like 'ABC123'", "e.g. W0000000". Drop the example
    # after the cue; backticks and guillemets count as quotes, and stacked cues don't swallow
    # each other.
    prose = _EXAMPLE_CUE.sub(lambda m: m.group(1) + " ", prose)
    # Hedged numbers are approximations, not exact claims: "about $46 to $54". Left out, since
    # exact matching can't judge them (an invented approximate number passes; stated limit).
    prose = _HEDGED.sub(lambda m: m.group(1) + " ", prose)

    # IDs with separators (ORD-48213, gift_card_7091239), plus their numeric parts ("Gift Card
    # 7091239") and alphabetic parts (omar_rossi_1241 grounds "Omar Rossi").
    for m in _SEP_ID.finditer(prose):
        token = m.group(0)
        if re.fullmatch(r"\d+(?:-[A-Za-z]+)+", token):
            continue
        found[token] = None
        for part in re.split(r"[-_]", token):
            if re.fullmatch(r"\d{4,}", part):
                found[str(int(part))] = None
            elif re.fullmatch(r"[A-Za-z]{3,}", part) and "_" in token:
                found[part.lower()] = None
    # Codes mixing letters and digits with no separator (HAT266, W5272531), taken whole.
    for m in _RUN_ID.finditer(prose):
        found[m.group(0)] = None
    # Quoted strings, minus the trailing punctuation American style puts inside the quotes.
    for m in _QUOTED.finditer(prose):
        q = re.sub(r"[\s.,;:!?]+$", "", m.group(1))
        if len(q) >= 3:  # two-letter quotes ("no", "ok") are instructions, not claims
            found[q] = None
    # Standalone numbers only: never digits glued to letters, which belong to a code.
    for m in _NUMBER.finditer(prose):
        n = normalize_number(m.group(0))
        if n is None or _is_trivial(n):
            continue
        found[n] = None
    return list(found)


def payload_values(payload: object) -> set[str]:
    """Every scalar anywhere inside a tool payload, normalized the same way as extracted values,
    so "1,024" in the answer matches ``1024`` in the JSON. Object keys count as known too."""
    out: set[str] = set()

    def walk(v: object) -> None:
        if v is None:
            return
        if isinstance(v, bool):
            out.add("true" if v else "false")
            return
        if isinstance(v, (int, float)):
            if math.isfinite(v):
                out.add(_js_number_string(float(v)))
            return
        if isinstance(v, str):
            out.add(v)
            out.add(v.lower())
            n = normalize_number(v)
            if n is not None:
                out.add(n)
            out.update(extract_values(v))
            return
        if isinstance(v, (list, tuple)):
            for item in v:
                walk(item)
            return
        if isinstance(v, dict):
            for key, item in v.items():
                out.add(str(key))
                walk(item)

    walk(payload)
    return out
