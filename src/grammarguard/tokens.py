"""Token-vocabulary analysis: which vocabulary tokens can never appear in
any string the pattern accepts.

Semantics and honesty: a token is reported "dead" when its text does not
occur as a *contiguous substring* of any accepted string (checked via the
DFA: no state can consume the whole token and remain on a productive path).
Not occurring as a substring proves the token can never be emitted by a
constrained decoder under this pattern. The converse is NOT guaranteed: a
token reported "live" merely occurs as a substring of some accepted string;
whether a decoder's actual tokenization would emit it depends on the
tokenizer's merge rules, which this analysis does not model. Character-level
regexes also cannot see byte-level BPE behavior. Treat results as a
conservative audit, not a correctness proof.

Execution is bounded: at most ``max_checks`` (token, state) pairs are
examined; when the budget is exhausted the result reports ``truncated``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .automata import DFA, productive_states

DEFAULT_CHECK_BUDGET = 5_000_000


def load_vocab(path: str) -> List[str]:
    """Load a vocabulary file: one token per line, duplicates removed.

    Blank lines are skipped. This is intentionally a plain-text format;
    export tokens from a tokenizer (e.g. ``tokenizer.get_vocab()``) first.
    """
    tokens: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            tok = line.rstrip("\n").rstrip("\r")
            if not tok or tok in seen:
                continue
            seen.add(tok)
            tokens.append(tok)
    return tokens


def dead_tokens(
    dfa: DFA,
    vocab,
    max_checks: int = DEFAULT_CHECK_BUDGET,
    max_reported: int = 200,
) -> dict:
    """Classify vocabulary tokens as provably-unusable ("dead") or not.

    Returns a dict with keys: dead (list, capped at max_reported),
    dead_count, live_count, checked, truncated.
    """
    prod = productive_states(dfa)
    # Precompute per-state sorted transitions as list for scanning.
    dead: List[str] = []
    dead_count = 0
    live_count = 0
    checks = 0
    truncated = False
    for tok in vocab:
        if checks >= max_checks:
            truncated = True
            break
        live = False
        for start_state in range(dfa.n_states):
            if checks >= max_checks:
                truncated = True
                break
            checks += 1
            state = start_state
            ok = True
            for ch in tok:
                nxt = dfa.step(state, ord(ch))
                if nxt is None:
                    ok = False
                    break
                state = nxt
            if ok and state in prod:
                live = True
                break
        if truncated:
            break
        if live:
            live_count += 1
        else:
            dead_count += 1
            if len(dead) < max_reported:
                dead.append(tok)
    return {
        "dead": dead,
        "dead_count": dead_count,
        "live_count": live_count,
        "checked": dead_count + live_count,
        "truncated": truncated,
    }
