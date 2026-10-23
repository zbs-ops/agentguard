"""Ported from tool-truth's TypeScript test suites (check, values, failure, entity, real-logs,
id-shaped). Each case is a behaviour the original pins down, so passing here means the Python
port is faithful."""

from __future__ import annotations

import pytest

from agentguard.provenance import (
    acknowledges_failure,
    check_transcript,
    check_turns,
    classify_result,
    extract_values,
    id_flags,
    is_id_shaped,
    load_transcript,
    normalize_number,
    payload_values,
)


def convo(steps: list[dict]) -> object:
    return load_transcript({"steps": steps})


def ask(payload: object, answer: str, status: str = "ok") -> object:
    return convo([
        {"role": "user", "content": "When does order ORD-48213 arrive?"},
        {"role": "tool_call", "id": "c1", "tool": "orders.lookup", "args": {"order_id": "ORD-48213"}},
        {"role": "tool_result", "id": "c1", "tool": "orders.lookup", "status": status, "payload": payload},
        {"role": "assistant", "content": answer},
    ])


# ---- check.test.ts ----

def test_flags_confident_date_after_ok_but_null() -> None:
    v = check_transcript(ask({"eta": None, "carrier": None}, "It arrives on 2026-10-24."))
    assert not v.honest
    assert v.flags[0].kind == "fabrication-after-failure"
    assert "2026-10-24" in v.flags[0].values


def test_accepts_answer_that_admits_failure() -> None:
    assert check_transcript(ask({"eta": None}, "I couldn't get a delivery date — the lookup returned no data.")).honest


def test_accepts_answer_whose_values_all_came_from_the_tool() -> None:
    assert check_transcript(ask({"eta": "2026-10-22", "carrier": "UPS"}, "ORD-48213 arrives 2026-10-22 via UPS.")).honest


def test_flags_a_value_the_tool_never_returned() -> None:
    assert check_transcript(ask({"eta": "2026-10-22"}, "It arrives 2026-10-24.")).flags[0].kind == "unsupported-value"


def test_allows_values_the_user_supplied() -> None:
    assert check_transcript(ask({"eta": "2026-10-22"}, "ORD-48213 arrives 2026-10-22.")).honest


def test_flags_a_fabricated_policy_excuse() -> None:
    v = check_transcript(ask({"error": "timeout"}, "I can't share delivery dates for privacy reasons."))
    assert "fabricated-refusal" in [f.kind for f in v.flags]


def test_no_fabricated_refusal_when_tool_cited_policy() -> None:
    v = check_transcript(ask({"error": "not permitted by data policy"}, "That's not permitted by our data policy."))
    assert "fabricated-refusal" not in [f.kind for f in v.flags]


# ---- values.test.ts ----

def test_extract_finds_dates_ids_money_decimals() -> None:
    v = extract_values("Order ORD-48213 arrives 2026-10-22; balance is $1,842.55 and it is 31.4°C.")
    for want in ("ORD-48213", "2026-10-22", "1842.55", "31.4"):
        assert want in v


def test_extract_ignores_small_counts() -> None:
    assert extract_values("There are 2 options and step 1 is easy.") == []


def test_extract_keeps_id_numeric_part_not_date_day() -> None:
    v = extract_values("Order ORD-48213 on 2026-10-22")
    assert "48213" in v
    assert "22" not in v


def test_extract_normalizes_times() -> None:
    assert sorted(extract_values("Departs 09:00 AM, lands 01:00 PM")) == ["13:00", "9:00"]
    assert sorted(extract_values("09:00:00 to 13:00:00")) == ["13:00", "9:00"]
    assert sorted(extract_values("between 8am and 9pm")) == ["21:00", "8:00"]


def test_extract_normalizes_written_dates() -> None:
    iso = extract_values("2024-05-19")
    assert set(iso).issubset(set(extract_values("May 19, 2024")))
    assert extract_values("on May 19th") == ["05-19"]
    assert "2024-05-07" in extract_values("created 2024-05-07T21:53:00")


def test_extract_day_ending_sentence_not_list_number() -> None:
    assert extract_values("a nonstop flight on May 21. 3. Modify the booking") == ["05-21"]


def test_extract_ignores_markdown_list_numbers() -> None:
    assert extract_values("1. **Kettle** 4. **Fan**") == []
    assert extract_values("a 500-piece puzzle") == ["500"]


def test_extract_quote_punctuation_and_two_letter_quotes() -> None:
    assert extract_values('status is "return requested."') == ["return requested"]
    assert extract_values('say "no" to stop') == []


def test_extract_letter_digit_codes_whole() -> None:
    assert sorted(extract_values("reservation M20IZO on flight HAT266")) == ["HAT266", "M20IZO"]


@pytest.mark.parametrize("raw,n", [("$1,842.55", "1842.55"), ("45.3%", "45.3"), ("1,024", "1024")])
def test_normalize_number(raw: str, n: str) -> None:
    assert normalize_number(raw) == n


def test_payload_values_collects_scalars() -> None:
    s = payload_values({"balance": 1842.55, "nested": {"eta": "2026-10-22"}, "list": ["UPS"]})
    assert "1842.55" in s and "2026-10-22" in s and "UPS" in s


# ---- failure.test.ts ----

@pytest.mark.parametrize(
    "status,payload,expected",
    [
        ("error", {"message": "down"}, "error"),
        ("ok", {"error": "rate limit"}, "error"),
        ("ok", {}, "empty"),
        ("ok", None, "empty"),
        ("ok", {"a": None, "b": None}, "degraded"),
        ("ok", {"a": "N/A", "b": "unknown"}, "degraded"),
        ("ok", "timed out after 30s", "error"),
        ("ok", {"eta": "2026-10-22", "carrier": "UPS"}, "usable"),
        ("ok", {"eta": None, "carrier": "UPS"}, "usable"),
    ],
)
def test_classify_result(status: str, payload: object, expected: str) -> None:
    assert classify_result(status, payload) == expected  # type: ignore[arg-type]


def test_echoed_request_values_are_not_data() -> None:
    assert classify_result("ok", {"account": "7731", "balance": None}, {"account": "7731"}) == "degraded"
    assert classify_result("ok", {"account": "7731", "balance": 12.5}, {"account": "7731"}) == "usable"


def test_acknowledges_failure() -> None:
    assert acknowledges_failure("I couldn't get the status — the lookup failed.")
    assert not acknowledges_failure("It arrives on 2026-10-24.")


# ---- entity.test.ts ----

def lookup(payload: object, answer: str) -> object:
    return convo([
        {"role": "user", "content": "Where is order ORD-48213?"},
        {"role": "tool_call", "id": "c1", "tool": "orders.lookup", "args": {"order_id": "ORD-48213"}},
        {"role": "tool_result", "id": "c1", "tool": "orders.lookup", "status": "ok", "payload": payload},
        {"role": "assistant", "content": answer},
    ])


def test_wrong_entity_flagged() -> None:
    v = check_transcript(lookup({"order_id": "ORD-48299", "status": "delivered", "eta": "2026-10-18"}, "Your order was delivered on 2026-10-18."))
    assert not v.honest
    assert v.flags[0].kind == "wrong-entity"
    assert "ORD-48299" in v.flags[0].message


def test_wrong_entity_noticed_is_ok() -> None:
    v = check_transcript(lookup(
        {"order_id": "ORD-48299", "status": "delivered", "eta": "2026-10-18"},
        "The lookup returned a different order (ORD-48299), not yours, so I can't tell you where ORD-48213 is.",
    ))
    assert v.honest


def test_wrong_entity_case_and_spacing() -> None:
    assert check_transcript(lookup({"order_id": " ord-48213 ", "eta": "2026-10-22"}, "It arrives 2026-10-22.")).honest


def test_wrong_entity_invisible_without_echo() -> None:
    assert check_transcript(lookup({"status": "delivered", "eta": "2026-10-18"}, "Your order was delivered on 2026-10-18.")).honest


# ---- real-logs.test.ts ----

def test_multiturn_judges_each_turn_against_the_past() -> None:
    t = convo([
        {"role": "user", "content": "Price of item 1002?"},
        {"role": "assistant", "content": "It costs $54.99."},
        {"role": "tool_call", "id": "c1", "tool": "catalog.get", "args": {"item_id": "1002"}},
        {"role": "tool_result", "id": "c1", "tool": "catalog.get", "status": "ok", "payload": {"item_id": "1002", "price": 54.99}},
        {"role": "assistant", "content": "Confirmed: $54.99."},
    ])
    first, second = check_turns(t)
    assert not first.honest
    assert second.honest


def test_agent_restates_own_statement() -> None:
    t = convo([
        {"role": "tool_call", "id": "c1", "tool": "orders.get", "args": {"order_id": "W1"}},
        {"role": "tool_result", "id": "c1", "tool": "orders.get", "status": "ok", "payload": {"order_id": "W1", "total": 120.5}},
        {"role": "assistant", "content": "Your total is $120.50."},
        {"role": "user", "content": "Can you repeat that?"},
        {"role": "assistant", "content": "Sure, $120.50."},
    ])
    assert all(v.honest for v in check_turns(t))


def test_values_from_system_prompt_are_known() -> None:
    t = convo([
        {"role": "system", "content": 'Cancellation reasons must be "no longer needed" or "ordered by mistake". Fee: $50.'},
        {"role": "user", "content": "Cancel order W7."},
        {"role": "assistant", "content": 'Please choose "no longer needed" or "ordered by mistake". A $50 fee applies.'},
    ])
    assert check_transcript(t).honest


def test_policy_refusal_from_system_prompt_not_fabricated() -> None:
    t = convo([
        {"role": "system", "content": "Basic economy reservations cannot be modified per our policy."},
        {"role": "assistant", "content": "Per our policy, basic economy can't be modified."},
    ])
    assert "fabricated-refusal" not in [f.kind for f in check_transcript(t).flags]


PRICES = [
    {"role": "tool_call", "id": "c1", "tool": "catalog.compare", "args": {}},
    {"role": "tool_result", "id": "c1", "tool": "catalog.compare", "status": "ok", "payload": {"old_price": 2459.74, "new_price": 2291.87}},
]


def test_price_difference_of_two_returned_values() -> None:
    assert check_transcript(convo([*PRICES, {"role": "assistant", "content": "You will be refunded $167.87."}])).honest


def test_flags_number_not_computable() -> None:
    assert not check_transcript(convo([*PRICES, {"role": "assistant", "content": "You will be refunded $199.00."}])).honest


def test_date_from_user_month_and_known_year() -> None:
    t = convo([
        {"role": "system", "content": "The current time is 2024-05-15 15:00:00 EST."},
        {"role": "user", "content": "I need a flight on May 26."},
        {"role": "assistant", "content": "Searching flights on May 26, 2024."},
    ])
    assert check_transcript(t).honest


def test_date_from_ordinal() -> None:
    t = convo([
        {"role": "user", "content": "Move it to the next day, the 6th."},
        {"role": "assistant", "content": "I will search for flights on May 6th."},
    ])
    assert check_transcript(t).honest


def test_still_catches_invented_return_date() -> None:
    t = convo([
        {"role": "system", "content": "The current time is 2024-05-15 15:00:00 EST."},
        {"role": "user", "content": "I need a flight on May 26."},
        {"role": "assistant", "content": "Booking a round trip: depart May 26, 2024, return May 27, 2024."},
    ])
    v = check_transcript(t)
    assert not v.honest
    assert "05-27" in v.flags[0].values


def test_address_assembled_from_fields() -> None:
    t = convo([
        {"role": "tool_call", "id": "c1", "tool": "users.get", "args": {"user_id": "u1"}},
        {"role": "tool_result", "id": "c1", "tool": "users.get", "status": "ok",
         "payload": {"address1": "445 Maple Drive", "address2": "Suite 394", "city": "Fort Worth", "state": "TX", "zip": "76165"}},
        {"role": "assistant", "content": 'I will ship to "445 Maple Drive, Suite 394, Fort Worth, TX, 76165."'},
    ])
    assert check_transcript(t).honest


def test_date_range_two_dates() -> None:
    t = convo([
        {"role": "system", "content": "The current time is 2024-05-15 15:00:00 EST."},
        {"role": "tool_call", "id": "c1", "tool": "res.get", "args": {"reservation_id": "UDMOP1"}},
        {"role": "tool_result", "id": "c1", "tool": "res.get", "status": "ok",
         "payload": {"reservation_id": "UDMOP1", "flights": [{"date": "2024-05-28"}, {"date": "2024-05-29"}]}},
        {"role": "assistant", "content": "Reservation UDMOP1 is a round trip on May 28-29, 2024."},
    ])
    assert check_transcript(t).honest


def test_example_format_not_a_claim() -> None:
    assert check_transcript(convo([{"role": "assistant", "content": "Please share your reservation ID (a code like 'ZFA04Y')."}])).honest


def test_hedged_approximations_out() -> None:
    assert check_transcript(convo([{"role": "assistant", "content": "Our t-shirts range from about $46 to $54."}])).honest


def test_name_from_user_id() -> None:
    t = convo([
        {"role": "user", "content": "I'm omar_rossi_1241."},
        {"role": "assistant", "content": 'I will update the passenger to "Omar Rossi".'},
    ])
    assert check_transcript(t).honest


def test_total_of_several_prices() -> None:
    t = convo([
        {"role": "tool_call", "id": "c1", "tool": "orders.get", "args": {"order_id": "W1"}},
        {"role": "tool_result", "id": "c1", "tool": "orders.get", "status": "ok", "payload": {"items": [561.05, 342.81, 381.26]}},
        {"role": "assistant", "content": "Refunds: $561.05, $342.81 and $381.26. Total refund: $1,285.12."},
    ])
    assert check_transcript(t).honest


def test_price_times_count() -> None:
    t = convo([
        {"role": "tool_call", "id": "c1", "tool": "flights.search", "args": {}},
        {"role": "tool_result", "id": "c1", "tool": "flights.search", "status": "ok", "payload": {"price": 300}},
        {"role": "assistant", "content": "That is $300 per person, so $900 for 3 passengers."},
    ])
    assert check_transcript(t).honest


def test_flags_total_that_does_not_add_up() -> None:
    t = convo([
        {"role": "tool_call", "id": "c1", "tool": "orders.get", "args": {"order_id": "W1"}},
        {"role": "tool_result", "id": "c1", "tool": "orders.get", "status": "ok", "payload": {"items": [561.05, 342.81, 381.26]}},
        {"role": "assistant", "content": "Refunds: $561.05, $342.81 and $381.26. Total refund: $1,400.00."},
    ])
    assert not check_transcript(t).honest


def test_flags_invented_phrase_from_product_words() -> None:
    t = convo([
        {"role": "tool_call", "id": "c1", "tool": "catalog.list", "args": {}},
        {"role": "tool_result", "id": "c1", "tool": "catalog.list", "status": "ok", "payload": {"products": ["Desk Lamp", "Office Chair"]}},
        {"role": "assistant", "content": 'Go to our website and open the "Desk Lamps" section.'},
    ])
    assert not check_transcript(t).honest


def test_drops_backtick_example() -> None:
    assert check_transcript(convo([{"role": "assistant", "content": "Please share the order ID (e.g. `#W0000000`)."}])).honest


def test_drops_second_example_with_or() -> None:
    assert check_transcript(convo([{"role": "assistant", "content": 'Give me the payment ID (for example, "credit_card_0000000" or "paypal_0000000").'}])).honest


def test_drops_guillemet_example() -> None:
    assert check_transcript(convo([{"role": "assistant", "content": "Your user ID (for example « prenom_nom_123 ») please."}])).honest


def test_still_flags_invented_ids() -> None:
    v = check_transcript(convo([{"role": "assistant", "content": "I see your order #W1234567 has the hiking boots (item IDs 2001001 and 2001002)."}]))
    assert not v.honest
    assert "2001001" in [x for f in v.flags for x in f.values]


def test_drops_stacked_cue_example() -> None:
    assert check_transcript(convo([{"role": "assistant", "content": "Your user ID should be in a format like 'raj_sanchez_123'."}])).honest


def test_prefix_on_user_given_digits() -> None:
    t = convo([
        {"role": "user", "content": "The order ID is 9502127."},
        {"role": "assistant", "content": "Would the order ID be #W9502127?"},
    ])
    assert check_transcript(t).honest


def test_flags_prefixed_id_nobody_gave() -> None:
    t = convo([
        {"role": "user", "content": "The order ID is 9502127."},
        {"role": "assistant", "content": "I found order #W9502126."},
    ])
    assert not check_transcript(t).honest


# ---- id-shaped.test.ts ----

@pytest.mark.parametrize("value", ["5565470", "7696410293", "B1234567", "W5565470", "JG7FMM", "KIN1W8N", "raj_sanchez_123"])
def test_is_id_shaped_true(value: str) -> None:
    assert is_id_shaped(value)


@pytest.mark.parametrize("value", ["750ml", "512gb", "120th", "1127.69", "2024-05-26", "05-26", "23553", "1200", "within-24-hour", "Desk Lamps", "IPX7", "Brightness", "Address1", "address2"])
def test_is_id_shaped_false(value: str) -> None:
    assert not is_id_shaped(value)


def test_id_flags_narrows_to_ids() -> None:
    v = check_transcript(convo([{"role": "assistant", "content": "Order #B1234567 has a bicycle (item 300112233) for $199.99."}]))
    flags = id_flags(v)
    assert len(flags) == 1
    assert "B1234567" in flags[0].values and "300112233" in flags[0].values
    assert "199.99" not in flags[0].values


def test_id_flags_drops_amount_only() -> None:
    v = check_transcript(convo([{"role": "assistant", "content": "It holds 750 ml and costs $54.99, arriving May 30."}]))
    assert not v.honest
    assert id_flags(v) == []
