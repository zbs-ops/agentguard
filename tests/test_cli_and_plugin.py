"""The CLI subcommands and the pytest fixtures the package ships."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentguard.cli import main


def test_cli_check_flags_invented_value(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log = tmp_path / "t.json"
    log.write_text(json.dumps({"steps": [
        {"role": "tool_call", "id": "c1", "tool": "orders.get", "args": {"order_id": "W1"}},
        {"role": "tool_result", "id": "c1", "tool": "orders.get", "status": "ok", "payload": {"eta": "2026-10-22"}},
        {"role": "assistant", "content": "It arrives 2026-10-24."},
    ]}))
    assert main(["check", str(log)]) == 1
    assert "unsupported-value" in capsys.readouterr().out


def test_cli_check_honest_is_zero(tmp_path: Path) -> None:
    log = tmp_path / "ok.json"
    log.write_text(json.dumps({"steps": [{"role": "assistant", "content": "Hello."}]}))
    assert main(["check", str(log)]) == 0


def test_cli_gate_refuses_unknown_tool(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["gate", "vault.exfiltrate"]) == 1
    assert "DENY" in capsys.readouterr().out


def test_cli_gate_allows_a_read() -> None:
    assert main(["gate", "crm.read"]) == 0


def test_cli_bench_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench"]) == 0
    assert "allowlist" in capsys.readouterr().out


def test_fixture_assert_grounded(assert_grounded) -> None:  # type: ignore[no-untyped-def]
    t = {"steps": [
        {"role": "tool_call", "id": "c1", "tool": "o.get", "args": {}},
        {"role": "tool_result", "id": "c1", "tool": "o.get", "status": "ok", "payload": {"eta": "2026-10-22"}},
        {"role": "assistant", "content": "It arrives 2026-10-22."},
    ]}
    from agentguard.provenance import load_transcript

    assert_grounded(load_transcript(t))
    with pytest.raises(AssertionError):
        bad = {"steps": [{"role": "assistant", "content": "Order #W9999999 is confirmed."}]}
        assert_grounded(load_transcript(bad))


def test_fixture_assert_tool_allowed(assert_tool_allowed) -> None:  # type: ignore[no-untyped-def]
    assert_tool_allowed("crm.read")
    with pytest.raises(AssertionError):
        assert_tool_allowed("refund.create", estimated_cost_cents=9_000)


def test_fixture_assert_gate_holds(assert_gate_holds) -> None:  # type: ignore[no-untyped-def]
    from agentguard.shell import BLOCKED_BINARIES

    with pytest.raises(AssertionError):
        assert_gate_holds(lambda c: "DENY" if any(b in c.split() for b in BLOCKED_BINARIES) else "ALLOW")

    allowed = {"echo", "ls", "true", "pwd"}
    assert_gate_holds(lambda c: "ALLOW" if c.split() and c.split()[0] in allowed and not any(x in c for x in ";&|`$(){}<>\\\n'\"") else "DENY")
