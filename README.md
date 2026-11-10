# agentguard

> **Problem:** AI agents with tools invent facts, overspend, and get their shell commands talked around. Most stacks catch none of it.
>
> **Approach:** three deterministic guardrails, no API keys and no model calls: a policy gate for tool calls, a checker for invented values, and a benchmark for shell-command gates.
>
> **Proof:** 345 tests. The checker and the benchmark are ports of work measured on real agent logs and on three shipping AI coding tools.

```bash
pip install agentguard-eval
```

The package installs as `agentguard-eval` (PyPI found `agentguard` too close to an existing name) and imports as `agentguard`.

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

Eight ordered checks, each one asserted by a test: a malformed tool name is refused before it can borrow another rule, an unknown tool fails closed, a negative or non-finite cost is refused before the budget math, the budget is checked before the tier, a repeat of a money-moving or tier-3 request that is still waiting or already ran is refused, an understated or unusually large spend is escalated, and a runaway loop of writes is paused.

The repeat check is there because an agent that times out waiting for an answer often just asks again; without it, a person approving both copies would pay the same refund twice. `Governor` supplies it as `History(repeat_of=...)`, so `evaluate` stays pure. Asking again after a rejection or expiry is fine, and a different order id is a different request. A ledger written by agent-governance for this case (`tests/fixtures/governance-repeat-ledger.json`) is reproduced byte for byte.

`evaluate` judges one call. `Governor` runs the whole approval workflow around it: it holds what's waiting for a person, checks who is approving, charges the budget only when something actually runs, and writes every step to a hash-chained ledger.

```python
from agentguard.gate import Governor

g = Governor(reviewer_keys={"you@company.com": "your-secret"})
a = g.propose("billing-agent", "refund.create", {"amount_cents": 9000}, estimated_cost_cents=9000)
print(a.state.value)                     # PENDING_APPROVAL

token = g.mint_token(a.id, "you@company.com")   # only the key holder can make this
g.decide(a.id, True, "you@company.com", token=token)
print(a.state.value, g.spent_cents)     # EXECUTED 9000
print(g.verify().message)               # All 4 entries verify against their hashes.
```

Each rule here is a bug the TypeScript project found and fixed: the agent that asked can't approve its own request; typing a reviewer's name isn't enough, for a yes or a no (only the agent that asked can withdraw without one), it takes a token bound to that one action's contents (ids restart with every new Governor, so a token bound to `act_0001` alone once approved a different, bigger `act_0001`); an approval stops counting 24 hours after the request; and the budget is checked again at the moment an approved action runs, because two refunds that each fit alone can both get a yes. Ledgers are interchangeable with the TypeScript version: the same calls produce the same ledger byte for byte, hashes included (`tests/test_governor.py` checks this against a ledger the TypeScript store wrote).

Above $250 (`two_person_above_cents`, by the amount the gate charges, so an understated estimate doesn't avoid it), one yes isn't enough: two different reviewers have to approve before the action runs. The first is logged as `APPROVED` with `"approval": "1 of 2"` and the action keeps waiting (`action.approvals` lists who said yes). The same reviewer again raises `SameApproverError` and writes nothing, either reviewer can still reject, and the budget and the 24-hour window are checked again at the second yes. Pass `two_person_above_cents=None` to turn it off. A 28-entry ledger agent-governance wrote for this (`tests/fixtures/governance-two-person-ledger.json`) is reproduced byte for byte.

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

Tool calls a model also writes into its message text (Claude's `<function_calls>` markup) are stripped first, so tool names aren't read as invented claims. On tool-truth's example logs, agentguard prints exactly the flags and values the TypeScript original prints, in the same order (`tests/test_parity.py`). A second fixture pins value extraction itself: 12 sentences (numbers with units like `15GB`, quoted phrases) give the same values, in the same order, as tool-truth 0.5.6 (`tests/test_units_quotes.py`).

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

There's also a wrapper set (`agentguard bench --wrappers`, or `i, m = wrapper_set(); run_bench(intents=i, mutators=m)`): one harmless word in front, like `timeout 5 rm` or `env rm`. The [gate-bypass-bench](https://github.com/zbs-ops/gate-bypass-bench) study found this gets past the real deny rules of Gemini CLI, Codex and opencode 12 times out of 12. A first-word denylist here does the same; the allowlist still holds at 0/12. It's kept separate, so the scorecard above is unchanged.

And a script-file set (`agentguard bench --scripts`): write the blocked command into a file and run it (`sh s.sh`, dot-source, a here-document, or base64 into a file). The gate judges `sh s.sh`, not the command in the file, so even the tokenized denylist lets all 12 through; the same gate-bypass-bench study found all three products' deny rules do too. The allowlist holds at 0/12. The renderings match gate-bypass-bench byte for byte (`tests/fixtures/script-renderings.json`).

And a launcher set (`agentguard bench --launchers`): the command run from `find -exec`, `find -execdir`, `find -exec sh -c`, or a `git -c alias.x='!cmd' x` alias. This is the one case the allowlist doesn't cover for free: the built-in allowlist here still holds at 0/12, because it names only four harmless programs, but it names neither `find` nor `git`. The gate-bypass-bench study added those two (the programs agents use all day) to each product's allow rule and found Gemini CLI and opencode then let all 12 through, Codex 3 of 12. So an allowlist is only as safe as the programs on it. Renderings pinned to gate-bypass-bench (`tests/fixtures/launcher-renderings.json`).

And an interpreter set (`agentguard bench --interpreters`): the command run from a `python3 -c`, `node -e` or `perl -e` one-liner, like `python3 -c 'import os; os.system("...")'`. These are the same problem as the launchers, with programs an agent is even more likely to be allowed. The tokenized denylist lets every one through, and an allowlist that names `python3`, `node` or `perl` does too (a test builds that allowlist and checks). That's 9 attacks where all three interpreters are installed; an attack whose interpreter is missing doesn't run, so it drops out of the denominator instead of counting as blocked. The sandbox PATH gained `/usr/local/bin` at the end, where node usually lives; every other scorecard here is byte-for-byte unchanged. Renderings, and the whole scorecard, match gate-bypass-bench (`tests/fixtures/interpreter-renderings.json`).

And a utility set (`agentguard bench --utilities`): the command run from GNU `sed`'s `e` command (`sed -n '1e cmd' /etc/passwd`), a GNU `tar` checkpoint action (`--checkpoint-action=exec='cmd'`) or `make --eval='x: ; @cmd' x`. These look read-only or build-only, so they end up on allowlists. The tokenized denylist lets every effective one through, the built-in allowlist none, and an allowlist naming `sed`, `tar` and `make` all of them (a test builds it). That's 9 attacks where all three are installed (CI prints the scorecard) and 6 on a box without `make`; a missing program drops out of the denominator instead of counting as blocked. Renderings, and the whole scorecard, match gate-bypass-bench (`tests/fixtures/utility-renderings.json`).

And an option set (`agentguard bench --options`): the command run from `git ls-remote --upload-pack='cmd; :' .`, from a module `node --import` loads out of a `data:` URL, or from the statement `python3 -m timeit` times. None of them uses `-c`, `-e` or `-p`. That matters because a careful gate allows git, node and python3 but refuses their inline-code flags. A test builds that gate: it stops every interpreter one-liner above and lets every effective option attack through. git's upload-pack sends its output to git rather than the terminal, so the secret read can't show and is left out of the denominator rather than forced in. That leaves 8 attacks where python3 is installed (CI prints the scorecard) and 5 without it. The built-in allowlist holds at 0, since it names none of the three. Renderings, and the whole scorecard, match gate-bypass-bench (`tests/fixtures/option-renderings.json`).

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
agentguard bench --scripts           # the script-file attack set
agentguard bench --launchers         # the launcher set (find -exec, git aliases)
agentguard bench --interpreters      # the interpreter set (python3 -c, node -e, perl -e)
agentguard bench --utilities         # the utility set (sed e, tar checkpoint, make --eval)
agentguard bench --options           # the option set (git --upload-pack, node --import, python3 -m timeit)
agentguard verify-ledger ledger.json # exit 1 and name the first edited entry
```

`agentguard check` exits non-zero when a transcript is flagged, so it can gate a CI job over saved agent runs.

`verify-ledger` reads the [agent-governance live demo](https://agent-governance-theta.vercel.app)'s audit log as-is, so you can check a server's log without trusting the server:

```bash
curl -s https://agent-governance-theta.vercel.app/api/audit > audit.json
agentguard verify-ledger audit.json   # valid: All 15 entries verify against their hashes.
```

Hit "Tamper with it" on the demo first and it names the same broken entry the demo's own verifier does.

---

## Where this comes from

The checker and the benchmark are Python ports of two projects I tested on real data, and the write-ups there carry the honest numbers:

- **[tool-truth](https://github.com/zbs-ops/tool-truth)**: the checker, run on 660+ real agent conversations from τ-bench and τ²-bench. The first version was useless on real logs; after fixes it caught agents inventing an order number a customer then relied on. Held-out results, good and bad, are written up in full.
- **[gate-bypass-bench](https://github.com/zbs-ops/gate-bypass-bench)**: the benchmark, which I ran against the real command rules of Google's Gemini CLI, OpenAI's Codex CLI, and opencode. Their deny rules were bypassed 78–84% of the time; the same engines as allowlists held at 0%.
- **[agent-governance](https://github.com/zbs-ops/agent-governance)**: the gate and the approval workflow, with a live demo and a tamper-evident audit log.

Background: Chen & Lin, *denylist fragility in AI agents* ([arXiv:2606.15549](https://arxiv.org/abs/2606.15549)); Sethi et al., *fabrication after tool failure* ([arXiv:2609.14758](https://arxiv.org/abs/2609.14758)); Saltzer & Schroeder (1975) on fail-safe defaults.

## Scope and limits

- **No model, by design.** The checker uses exact-match provenance. It catches invented concrete values; it cannot judge paraphrase or multi-step arithmetic, and the TypeScript write-ups measure that boundary on real data.
- **The gate's policy is a sensible default, not your policy.** The tiers and the $500 cap are the demo's; the point is the shape.
- **The Governor keeps state in memory.** Save `g.ledger()` (each entry has `to_dict()`) wherever you keep records; `verify-ledger` checks it later. Reviewer keys are plain secrets you hand it; in production they belong behind your identity provider.
- **The benchmark runs POSIX `sh`.** The rewrites re-spell dangerous-shaped commands; they only touch files inside a per-case sandbox.

## How I build

I pick what gets built and why: a pip-installable version of guardrails I'd already tested in TypeScript, so they can run in anyone's Python CI. The building is done by Friday, an AI agent I run. Friday writes the code and the tests on its own cloud machine, opens each change as a pull request, and merges it when CI passes. For each release I get a plain-language write-up of what changed and how it was checked.

**How a port is checked.** A port isn't trusted because it looks right. Both versions run on the same inputs and the outputs are compared: the ledger has to match the TypeScript one byte for byte (hashes included), the checker has to print the same flags in the same order on tool-truth's logs, and the script attacks have to render exactly as gate-bypass-bench renders them. Each of those is a committed fixture with a test.

**Problems hit, and the fixes:**

- **The name was taken.** PyPI refused `agentguard` as too close to an existing project, so it ships as `agentguard-eval` and still imports as `agentguard`.
- **An approval token could approve the wrong thing.** Found in agent-governance and carried into the port: action ids restart with every new Governor, so a token bound only to `act_0001` approved a different, bigger `act_0001`. Tokens here are bound to the action's contents, with a test for that case.
- **A number cut short.** The value extractor could read `1.5GB` as `1`, because the number pattern backed off to the whole part. Found in tool-truth first; both are fixed and a shared fixture keeps them matching.
- **Branches that never got tested.** CI here runs on pull requests, so a pushed branch on its own showed nothing. The pull request now gets opened before waiting on CI.

## License

MIT
