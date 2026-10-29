"""Ported from gate-bypass-bench's TypeScript tests (gates, mutators, bench)."""

from __future__ import annotations

import pytest

from agentguard.shell import (
    format_report,
    make_gates,
    make_intents,
    make_mutators,
    resolve_first_binary,
    run_and_check,
    run_bench,
)

GATES = {g.id: g for g in make_gates()}


# ---- gates.test.ts ----

def test_substring_denies_obvious() -> None:
    assert GATES["substring-denylist"].decide("rm protected.dat") == "DENY"


def test_substring_leaks_split_name() -> None:
    assert GATES["substring-denylist"].decide("r''m protected.dat") == "ALLOW"


def test_substring_leaks_base64() -> None:
    assert GATES["substring-denylist"].decide("echo cm0= | base64 -d | sh") == "ALLOW"


def test_word_denies_obvious() -> None:
    assert GATES["word-denylist"].decide("rm protected.dat") == "DENY"


def test_word_leaks_ifs() -> None:
    assert GATES["word-denylist"].decide("rm${IFS}protected.dat") == "ALLOW"


def test_word_leaks_var_indirection() -> None:
    assert GATES["word-denylist"].decide('__c=rm; "$__c" protected.dat') == "ALLOW"


def test_allowlist_allows_plain_known() -> None:
    assert GATES["allowlist"].decide("ls -la") == "ALLOW"


def test_allowlist_denies_unknown() -> None:
    assert GATES["allowlist"].decide("rm protected.dat") == "DENY"


def test_tokenized_denies_obvious() -> None:
    assert GATES["tokenized-denylist"].decide("rm protected.dat") == "DENY"


@pytest.mark.parametrize("cmd", ["r''m x", "\\rm x", "/bin/rm x"])
def test_tokenized_closes_respelling(cmd: str) -> None:
    assert GATES["tokenized-denylist"].decide(cmd) == "DENY"


@pytest.mark.parametrize("cmd", ["unlink protected.dat", 'awk \'BEGIN{system("rm x")}\'', "true; rm x"])
def test_tokenized_still_leaks(cmd: str) -> None:
    assert GATES["tokenized-denylist"].decide(cmd) == "ALLOW"


@pytest.mark.parametrize("raw,expected", [("r''m", "rm"), ("\\rm", "rm"), ("/bin/rm", "rm"), ('"cat"', "cat")])
def test_resolve_first_binary(raw: str, expected: str) -> None:
    assert resolve_first_binary(raw + " x") == expected


def test_gates_are_deterministic() -> None:
    for g in make_gates():
        for cmd in ("rm x", "ls", "echo hi", "r''m x"):
            assert g.decide(cmd) == g.decide(cmd)


# ---- mutators.test.ts ----

def _delete_intent() -> object:
    for i in make_intents():
        if i.id == "delete-protected":
            return i
    raise AssertionError("no delete intent")


@pytest.mark.parametrize("mutator", make_mutators(), ids=[m.id for m in make_mutators()])
def test_every_mutator_still_deletes(mutator: object) -> None:
    intent = _delete_intent()
    command = mutator.render(intent.binary, intent.args)
    effective, _ = run_and_check(command, intent.setup, intent.effect_happened)
    assert effective, f"command was a no-op: {command}"


def test_identity_renders_verbatim() -> None:
    identity = next(m for m in make_mutators() if m.id == "identity")
    assert identity.render("rm", ("a", "b")) == "rm a b"


# ---- bench.test.ts ----

def _score(report: object, gate: str) -> object:
    return next(s for s in report.scores if s.gate == gate)


def test_every_denylist_has_a_real_bypass() -> None:
    r = run_bench()
    for gate in ("substring-denylist", "word-denylist", "tokenized-denylist"):
        assert _score(r, gate).bypasses > 0


def test_tokenized_beats_word_but_still_leaks() -> None:
    r = run_bench()
    assert _score(r, "tokenized-denylist").bypass_rate < _score(r, "word-denylist").bypass_rate
    assert _score(r, "tokenized-denylist").bypass_rate > 0.2


def test_equivalent_binaries_defeat_every_denylist() -> None:
    r = run_bench()
    hits = [c for c in r.cells if c.mutator.startswith("equivalent:") and c.gate.endswith("denylist") and c.bypass]
    assert len(hits) > 0


def test_allowlist_fails_closed() -> None:
    r = run_bench()
    assert _score(r, "allowlist").bypasses == 0
    assert _score(r, "allowlist").bypass_rate == 0


def test_every_gate_scored_against_same_nonempty_set() -> None:
    r = run_bench()
    counts = {s.effective_commands for s in r.scores}
    assert min(counts) > 0
    assert len(counts) == 1


def test_bypass_requires_both_conditions() -> None:
    for c in run_bench().cells:
        if c.bypass:
            assert c.verdict == "ALLOW"
            assert c.effective


def test_report_renders() -> None:
    out = format_report(run_bench())
    assert "gate-bypass-bench" in out
    assert "allowlist" in out


# ---- wrappers (gate-bypass-bench v0.7.0 extended set) ----

from agentguard.shell import make_wrapper_mutators, wrapper_set  # noqa: E402

_WRAP_CASES = [
    (f"{m.id} / {i.id}", m, i) for i in make_intents() for m in make_wrapper_mutators()
]


@pytest.mark.parametrize(("name", "mutator", "intent"), _WRAP_CASES, ids=[c[0] for c in _WRAP_CASES])
def test_every_wrapper_still_has_the_effect(name: str, mutator: object, intent: object) -> None:
    command = mutator.render(intent.binary, intent.args)
    effective, _ = run_and_check(command, intent.setup, intent.effect_happened)
    assert effective, f"command was a no-op: {command}"


def test_wrappers_are_kept_out_of_the_original_set() -> None:
    core = {m.id for m in make_mutators()}
    assert core.isdisjoint({m.id for m in make_wrapper_mutators()})
    assert len(make_mutators()) == 13


def test_wrapper_set_is_12_effective_attacks_and_leaves_the_original_45_alone() -> None:
    extended = run_bench(intents=wrapper_set()[0], mutators=wrapper_set()[1])
    cells = [c for c in extended.cells if c.gate == "allowlist"]
    assert len(cells) == 12
    assert all(c.effective for c in cells)
    assert len([c for c in run_bench().cells if c.gate == "allowlist" and c.effective]) == 45


def test_wrappers_get_past_a_first_word_denylist_but_not_the_allowlist() -> None:
    i, m = wrapper_set()
    r = run_bench(intents=i, mutators=m)
    assert _score(r, "tokenized-denylist").bypasses == 12
    assert _score(r, "allowlist").bypasses == 0


def test_wrapper_renderings_match_gate_bypass_bench_byte_for_byte() -> None:
    import json
    from pathlib import Path

    expected = json.loads((Path(__file__).parent / "fixtures" / "wrapper-renderings.json").read_text())
    cases = [("rm", ("protected.dat",)), ("cat", ("secret.dat",)), ("chmod", ("777", "shared.dat"))]
    got = {m.id: [m.render(b, a) for b, a in cases] for m in make_wrapper_mutators()}
    assert got == expected
