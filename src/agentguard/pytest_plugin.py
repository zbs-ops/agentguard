"""A pytest plugin so you can assert on agent behaviour in your own test suite.

Registered as ``pytest11``, so installing agentguard enables it. It adds assertion helpers for
the three checks and fails with a readable message when an agent's transcript isn't grounded,
a proposed tool call isn't allowed, or a shell gate lets a forbidden command through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentguard.gate import Budget, Decision, History


@pytest.fixture
def assert_grounded() -> Callable[..., None]:
    """Fail the test if an agent transcript states a value its tools never provided."""
    from agentguard.provenance import check_transcript

    def _assert(transcript: object) -> None:
        verdict = check_transcript(transcript)  # type: ignore[arg-type]
        if not verdict.honest:
            lines = "\n".join(f"  - {flag.message}" for flag in verdict.flags)
            raise AssertionError(f"agent stated ungrounded values:\n{lines}")

    return _assert


@pytest.fixture
def assert_tool_allowed() -> Callable[..., Decision]:
    """Fail the test if the policy gate would not auto-allow a proposed tool call."""
    from agentguard.gate import Budget as _Budget
    from agentguard.gate import Verdict, evaluate

    def _assert(
        tool: str,
        *,
        estimated_cost_cents: int = 0,
        args: dict[str, object] | None = None,
        budget: Budget | None = None,
        history: History | None = None,
    ) -> Decision:
        decision = evaluate(
            tool,
            estimated_cost_cents=estimated_cost_cents,
            args=args,
            budget=budget or _Budget(spent_cents=0, limit_cents=50_000),
            history=history,
        )
        if decision.verdict is not Verdict.ALLOW:
            raise AssertionError(f"{tool!r} was not allowed: {decision.verdict.value} ({decision.reason})")
        return decision

    return _assert


@pytest.fixture
def assert_gate_holds() -> Callable[..., None]:
    """Fail the test if a command gate lets any disguised forbidden command through.

    ``gate`` is ``(command: str) -> "ALLOW" | "DENY"`` (or a bool, where True means allow). The
    gate is scored against the benchmark's attacks; the test fails with the bypasses that
    landed, so you can pin a gate at zero bypasses in your own suite.
    """
    from agentguard.shell import Gate, run_bench

    def _assert(gate: Callable[[str], object], *, name: str = "gate-under-test") -> None:
        def decide(command: str) -> str:
            result = gate(command)
            if isinstance(result, bool):
                return "ALLOW" if result else "DENY"
            return str(result)

        report = run_bench(gates=[Gate(name, "user-supplied gate", decide)])
        bypasses = [c for c in report.cells if c.gate == name and c.bypass]
        if bypasses:
            lines = "\n".join(f"  - {c.intent} via {c.mutator}: {c.command}" for c in bypasses)
            raise AssertionError(f"{name} let {len(bypasses)} forbidden command(s) through:\n{lines}")

    return _assert
