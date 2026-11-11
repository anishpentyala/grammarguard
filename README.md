# grammarguard

Offline preflight checks for **regex constraints** used in guided / structured
generation — before a pattern reaches a constrained-decoding engine or a
validator.

When you pass a regex to vLLM/SGLang structured outputs (`regex=`), outlines,
or a JSON Schema `pattern` field, problems surface late and expensively:
unsatisfiable patterns that hang or produce nothing, automata that explode in
size, and patterns carrying known catastrophic-backtracking shapes (vLLM
patched a ReDoS advisory on exactly this input path:
[GHSA-rwxx-mrjm-wc2m](https://github.com/vllm-project/vllm/security/advisories/GHSA-rwxx-mrjm-wc2m);
engine compile time on pathological input is also a live issue:
[vllm#54003](https://github.com/vllm-project/vllm/issues/54003)).
grammarguard checks your pattern **offline, deterministically, with zero
dependencies**: parse it, build the actual automaton, and report what the
engine will see.

## Install from source

Python 3.10 or newer and Git are required. This project is not currently published
to PyPI; install the code from this repository instead of `pip install grammarguard`.
In a POSIX shell:

```bash
git clone https://github.com/anishpentyala/grammarguard.git
cd grammarguard
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
grammarguard '\d{4}-\d{2}-\d{2}'
```

```
pattern: \d{4}-\d{2}-\d{2}   mode: fullmatch
satisfiable: True
shortest witnesses: ["0000-00-00"]
sample matches (<= 16, truncated=True): ["0000-00-00", "0000-00-01", ...]
near-miss controls (all rejected): ["000-00-00", "00000-00-00", ...]
common literal prefix: ""
length bounds: min=10 max=10  finite=True
automaton: 11 DFA states (11 minimized), 10 transitions
ReDoS hazard lint: no known hazard shapes found (heuristic; not a safety proof)
```

```bash
grammarguard '(a+)+$'
#   [high] nested-unbounded: unbounded quantifier nested inside another
#   unbounded quantifier; classic catastrophic-backtracking shape (e.g. (a+)+)

grammarguard 'true|false' --json
grammarguard 'https://.*' --intersect 'https://a\.example\.com'
grammarguard '\d{4}' --vocab tokens.txt        # which tokens can never appear
grammarguard '\d{4}' --mode search             # JSON Schema `pattern` semantics
```

## What it checks

| Check | Question answered |
| --- | --- |
| satisfiability + witnesses | Can this pattern match anything at all? Show me the shortest matches. |
| bounded enumeration + near-miss controls | What do accepted strings look like — and what is correctly rejected? |
| automaton size (DFA + minimized DFA) | How big is the automaton the engine compiles? Explosion warnings. |
| length bounds + finiteness | Min/max output length (useful for `max_tokens` budgeting); is the language finite? |
| common literal prefix | Do all outputs share a stable prefix? (prompt-cache friendliness) |
| ReDoS hazard lint | Does the pattern contain known catastrophic-backtracking shapes? |
| pattern intersection | Do two patterns accept a common string? (constraint vs. wrapper format) |
| dead-token analysis | Which vocabulary tokens can never appear in any accepted string? |

## Honest scope

**Supported regex subset:** literals, `.` (any char except newline), character
classes with ranges/negation/escapes and `\d \D \w \W \s \S`, groups `(...)` /
`(?:...)`, alternation `|`, quantifiers `* + ? {n} {n,} {n,m}` (lazy and
possessive suffixes accepted; language-identical), and `^`/`$` at the pattern
edges (no-ops under full-match semantics, honored in search mode).

**Deliberate scope: predefined classes are ASCII-only.** `\d` is `[0-9]`,
`\w` is `[0-9A-Za-z_]`, `\s` is `[ \t\n\r\f\v]`. This is a standalone
modeling choice, not an engine-equivalence claim: it **diverges from
Python `re` on non-ASCII text** (Python's `\d` also matches e.g.
Arabic-Indic digits and `\w` matches Unicode word characters), and
ECMA-262's `\s` covers Unicode whitespace while grammarguard's does not.
grammarguard's test suite pins the ASCII semantics on purpose; if your engine is Unicode-aware (xgrammar and
outlines operate on Unicode), treat analysis of patterns containing
`\d \w \s` on non-ASCII text as an under-approximation.

**Loudly rejected** (`UnsupportedFeature`): lookarounds, backreferences, named
groups, atomic groups, conditionals, inline flags, `\b \B \A \Z \z`, and
anchors anywhere other than the pattern edges. That last one excludes valid
engine syntax such as `^a$|^b$` (per-branch anchors): it raises
`UnsupportedFeature` instead of being mis-modeled. Nothing is silently
mis-analyzed.

**Semantics:** full-match by default (the usual guided-decoding behavior);
`--mode search` models JSON Schema `pattern` substring semantics as
`.*pattern.*`. This is a modeling choice, not an engine guarantee: engine
regex dialects (xgrammar, outlines) and JSON Schema validators are not
identical to Python `re` or to each other, and grammarguard does not test
against any engine — it analyzes the documented subset above, with its test
suite validating language equivalence against Python `re` on randomized
patterns with positive *and* negative control strings.

**Limits of the checks:**

- The ReDoS lint is **heuristic hazard detection, not a safety proof**. A
  clean report means none of the known hazard shapes were found — it does not
  prove the pattern is safe with a backtracking engine.
- Dead-token analysis is **conservative, character-level and
  substring-based**: it compares raw token *text* against accepted *strings*.
  A "dead" token is provably unusable (its text occurs in no accepted
  string). A "live" token is only *possibly* usable — actual encoder
  behavior (BPE merges, byte pre-tokenization) is out of reach of a
  character-level analysis.
- Size-warning thresholds are heuristics, not engine limits.

**Bounded execution:** counted-repeat expansion (512 per repeat, 20,000 total
AST nodes), DFA construction (20,000 states), enumeration (length 256, per-level
budgets), and vocabulary checks (5M token-state pairs) are all capped; hitting a
cap raises `PatternTooLarge` or reports `truncated` instead of hanging.

## Library use

```python
import grammarguard

report = grammarguard.analyze(r"\d{4}-\d{2}-\d{2}", vocab_path="tokens.txt")
report["satisfiable"], report["witnesses"], report["min_dfa_states"]

grammarguard.analyze_intersection(r"a+", r"aa+")
# {'intersection_empty': False, 'witness': 'aa', ...}

grammarguard.redos_lint(grammarguard.parse("(a+)+"))
# [{'rule': 'nested-unbounded', 'risk': 'high', ...}]
```

## Development

```bash
python -m pytest tests/    # 124 tests, stdlib + pytest only
```

## Differentiation

Several tools count or shrink the *token cost* of tool schemas; grammarguard
checks the *behavioral correctness and cost hazards of the constraint itself*.
No dedicated user-facing tool for this was found in a targeted search
(~30 queries across PyPI/GitHub/web, November 2026); this claim is tentative
and corrections are welcome.

## License

MIT
