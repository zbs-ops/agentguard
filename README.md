# agentguard

> **Problem:** AI agents with tools invent facts, overspend, and get their shell commands talked around. Most stacks catch none of it.
>
> **Approach:** three deterministic guardrails, no API keys and no model calls: a policy gate for tool calls, a checker for invented values, and a benchmark for shell-command gates.
>
> **Proof:** 167 tests. The checker and the benchmark are ports of work measured on real agent logs and on three shipping AI coding tools.

```bash
pip install agentguard
```

Python 3.10+. Zero runtime dependencies. Everything here runs offline and gives the same answer every time, which is what makes it testable and what makes it safe to put in a CI job.

---

## What's in it

Three things that go wrong when you give a language model real tools, and a guard for each.

### 1. A policy gate: decide whether a tool call may run

Every tool call is a *proposal*. The gate returns one of three answers: run it, hold it for a person, or refuse it. Reads run unattended; money, customer messages, and deletes wait for a human. It fails closed, checks the budget before anything else, and catches an agent that estimates one cost while its arguments move a bigger one.

```python
from agentguard.gate import evaluate, Budget

d = evaluate("refund.create", estimated_cost_cents=100, args={"amount_cents": 9000},
             budget=Budget(spent_cents=0, limit_cents=50_000))
print(d.verdict.value, d.decided_by)   # REQUIRE_APPROVAL cost-understated
print(d.reason)   # "The agent estimated $1.00, but the arguments move $90.00 — escalated for review."
```

Seven ordered checks, each one asserted by a test: a malformed tool name is refused before it can borrow another rule, an unknown tool fails closed, a negative or non-finite cost is refused before the budget math, the budget is checked before the tier, an understated or unusually large spend is escalated, and a runaway loop of writes is paused.

### 2. A provenance checker: catch invented facts

Checks whether an agent's answer is backed by what its tools actually returned. Four checks: a confident answer after a tool failed, a value no tool returned and nobody stated, a record for the wrong entity passed off as the right one, and a refusal that cites a limit appearing nowhere in the transcript.

```python
from agentguard.provenance import check_transcript, load_transcript

t = load_transcript({"steps": [
    {"role": "tool_call", "id": "c1", "tool": "orders.get", "args": {"order_id": "W1"}},
    {"role": "tool_result", "id": "c1", "tool": "orders.get", "status": "ok", "payload": {"eta": "2026-10-22"}},
    {"role": "assistant", "content": "It arrives 2026-10-24."},
]})
v = check_transcript(t)
print(v.honest, v.flags[0].kind, v.flags[0].values)   # False unsupported-value ('2026-10-24',)
```

It reads the user, the system prompt, the tool calls and their results, and normalizes values so the same fact in different forms ("May 19, 2024" and `2024-05-19`, "$1,024" and `1024`) matches. It knows an agent computes (a refund that's the difference of two returned prices is grounded), and it leaves vague language alone, because exact-match is the honest limit of a checker with no model behind it.

For the short, high-yield list, `id_flags(verdict)` keeps only flags about an invented order number or product code, the fabrication that does the most harm.

It reads your agent's real logs directly. `to_transcript` takes an **OpenAI** chat log (`tool_calls` / role `tool`), an **Anthropic** messages log (`tool_use` / `tool_result` blocks), or agentguard's own `{"steps": [...]}`, and works out which one it got:

```python
import json
from agentguard.provenance import check_transcript, to_transcript

v = check_transcript(to_transcript(json.load(open("run.json"))))
```

Tool calls a model also writes into its message text (Claude's `<function_calls>` markup) are stripped first, so tool names aren't read as invented claims. On tool-truth's example logs, agentguard prints exactly the flags and values the TypeScript original prints, in the same order (`tests/test_parity.py`).

### 3. A benchmark: how often can a command gate be talked around?

Give an agent a shell and you put a gate in front of it: a list of commands it may not run. This rewrites a forbidden command in 13 known bypass styles (chaining, quoting, `$IFS`, variable indirection, `awk`, `base64`, equivalent binaries) and counts a bypass **only when the forbidden effect actually happens in a sandbox**.

```python
from agentguard.shell import run_bench, format_report

print(format_report(run_bench()))
```

```
gate                bypass rate   (bypasses / effective)
substring-denylist          31%   (14/45)
word-denylist               78%   (35/45)
tokenized-denylist          71%   (32/45)
allowlist                    0%   (0/45)
```

Two findings worth the whole module: a more precise denylist is not a safer one (whole-word matching misses `rm${IFS}file` that a dumb substring search catches), and only the allowlist, which names what may run and refuses everything else, holds at 0%.

---

## A pytest plugin

Installing agentguard adds fixtures so you can assert on agent behaviour in your own suite:

```python
def test_my_agent_is_grounded(assert_grounded):
    assert_grounded(my_agent_transcript)      # fails, listing the invented values

def test_my_policy_allows_reads(assert_tool_allowed):
    assert_tool_allowed("crm.read")           # fails if the gate wouldn't auto-allow it

def test_my_shell_gate_holds(assert_gate_holds):
    assert_gate_holds(my_gate)                # fails, listing the commands that got through
```

## A command line

```bash
agentguard check run.json            # OpenAI, Anthropic or agentguard format; exit 1 if flagged
agentguard check run.json --ids-only # only high-confidence invented IDs
agentguard gate refund.create --cost 9000 --amount 9000
agentguard bench                     # the shell-gate scorecard
```

`agentguard check` exits non-zero when a transcript is flagged, so it can gate a CI job over saved agent runs.

---

## Where this comes from

The checker and the benchmark are Python ports of two projects I tested on real data, and the write-ups there carry the honest numbers:

- **[tool-truth](https://github.com/zbs-ops/tool-truth)**: the checker, run on 660+ real agent conversations from τ-bench and τ²-bench. The first version was useless on real logs; after fixes it caught agents inventing an order number a customer then relied on. Held-out results, good and bad, are written up in full.
- **[gate-bypass-bench](https://github.com/zbs-ops/gate-bypass-bench)**: the benchmark, which I ran against the real command rules of Google's Gemini CLI, OpenAI's Codex CLI, and opencode. Their deny rules were bypassed 78–84% of the time; the same engines as allowlists held at 0%.
- **[agent-governance](https://github.com/zbs-ops/agent-governance)**: the gate, with a live demo and a tamper-evident audit log.

Background: Chen & Lin, *denylist fragility in AI agents* ([arXiv:2606.15549](https://arxiv.org/abs/2606.15549)); Sethi et al., *fabrication after tool failure* ([arXiv:2609.14758](https://arxiv.org/abs/2609.14758)); Saltzer & Schroeder (1975) on fail-safe defaults.

## Scope and limits

- **No model, by design.** The checker uses exact-match provenance. It catches invented concrete values; it cannot judge paraphrase or multi-step arithmetic, and the TypeScript write-ups measure that boundary on real data.
- **The gate's policy is a sensible default, not your policy.** The tiers and the $500 cap are the demo's; the point is the shape.
- **The benchmark runs POSIX `sh`.** The rewrites re-spell dangerous-shaped commands; they only touch files inside a per-case sandbox.

## How I build

I work with an AI coding assistant as a pair programmer. I decide what gets built, review every change, and own what ships.

## License

MIT
