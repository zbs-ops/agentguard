"""The ``agentguard`` command line.

- ``agentguard check <log.json>``: check an agent transcript for invented values. Exit code 1
  when it's flagged, so it can gate a CI job over saved runs. ``--ids-only`` keeps only
  high-confidence invented-ID flags.
- ``agentguard gate <tool> [--cost C] [--amount A] [--limit L] [--spent S]``: ask the policy
  gate what it would do with a proposed tool call.
- ``agentguard bench``: run the shell-command gate benchmark and print the scorecard.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from agentguard.gate import Budget, evaluate
from agentguard.provenance import check_transcript, id_flags, load_transcript
from agentguard.shell import format_report, run_bench


def _cmd_check(args: argparse.Namespace) -> int:
    with open(args.log, encoding="utf-8") as fh:
        transcript = load_transcript(json.load(fh))
    verdict = check_transcript(transcript)
    flags = id_flags(verdict) if args.ids_only else list(verdict.flags)
    if not flags:
        print("honest: no ungrounded values found.")
        return 0
    for flag in flags:
        values = f" [{', '.join(flag.values)}]" if flag.values else ""
        print(f"{flag.kind}: {flag.message}{values}")
    return 1


def _cmd_gate(args: argparse.Namespace) -> int:
    call_args: dict[str, object] = {}
    if args.amount is not None:
        call_args["amount_cents"] = args.amount
    decision = evaluate(
        args.tool,
        estimated_cost_cents=args.cost,
        args=call_args,
        budget=Budget(spent_cents=args.spent, limit_cents=args.limit),
    )
    print(f"{decision.verdict.value}  ({decision.decided_by})")
    for reason in decision.reasons:
        print(f"  - {reason}")
    return 0 if decision.verdict.value == "ALLOW" else 1


def _cmd_bench(_args: argparse.Namespace) -> int:
    print(format_report(run_bench()))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentguard", description="Guardrails and evals for AI agents.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="check an agent transcript (JSON) for invented values")
    p_check.add_argument("log", help="path to a transcript JSON file")
    p_check.add_argument("--ids-only", action="store_true", help="keep only high-confidence invented-ID flags")
    p_check.set_defaults(func=_cmd_check)

    p_gate = sub.add_parser("gate", help="ask the policy gate about a proposed tool call")
    p_gate.add_argument("tool")
    p_gate.add_argument("--cost", type=int, default=0, help="the agent's estimated cost, in cents")
    p_gate.add_argument("--amount", type=int, default=None, help="amount_cents the call would move")
    p_gate.add_argument("--limit", type=int, default=50_000, help="budget cap, in cents")
    p_gate.add_argument("--spent", type=int, default=0, help="cents already spent")
    p_gate.set_defaults(func=_cmd_gate)

    p_bench = sub.add_parser("bench", help="run the shell-command gate benchmark")
    p_bench.set_defaults(func=_cmd_bench)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
