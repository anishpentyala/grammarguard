"""High-level analysis API for grammarguard."""

from __future__ import annotations

from typing import Optional

from . import automata, redos
from .regex_ast import (
    DOTALL_RANGES,
    Alt,
    AnchorEnd,
    AnchorStart,
    Class,
    EmptyStr,
    PatternTooLarge,
    Repeat,
    Seq,
    parse,
)
from .tokens import dead_tokens, load_vocab

# Heuristic size-warning thresholds (documented, not safety boundaries).
WARN_DFA_STATES = 4096
WARN_STATE_TO_PATTERN_RATIO = 8
WARN_RATIO_MIN_STATES = 64


def _dotall_star():
    return Repeat(Class(tuple(DOTALL_RANGES), False), 0, None)


def _wrap_search(ast):
    """Wrap a pattern for substring (JSON Schema `pattern`-style) semantics.

    Anchors are branch-local: ^a|b is (^a)|(b), so each alternation branch
    is wrapped independently and an anchored side is not wrapped. ^foo$
    in search mode only matches the whole string "foo".
    """
    if isinstance(ast, Alt):
        return Alt(tuple(_wrap_search_branch(b) for b in ast.branches))
    return _wrap_search_branch(ast)


def _wrap_search_branch(ast):
    parts = list(ast.parts) if isinstance(ast, Seq) else [ast]
    anchored_start = bool(parts) and isinstance(parts[0], AnchorStart)
    anchored_end = bool(parts) and isinstance(parts[-1], AnchorEnd)
    if anchored_start:
        parts = parts[1:]
    if anchored_end:
        parts = parts[:-1]
    if not parts:
        inner = EmptyStr()
    elif len(parts) == 1:
        inner = parts[0]
    else:
        inner = Seq(tuple(parts))
    out = []
    if not anchored_start:
        out.append(_dotall_star())
    out.append(inner)
    if not anchored_end:
        out.append(_dotall_star())
    if len(out) == 1:
        return out[0]
    return Seq(tuple(out))


def analyze(
    pattern: str,
    *,
    mode: str = "fullmatch",
    witness_count: int = 3,
    enum_limit: int = 16,
    enum_max_len: int = 24,
    near_miss_limit: int = 8,
    vocab_path: Optional[str] = None,
    max_dfa_states: int = automata.MAX_DFA_STATES,
) -> dict:
    """Analyze a regex constraint for guided/structured generation.

    mode: "fullmatch" (guided-decoding semantics: the whole output must
    match) or "search" (JSON Schema `pattern` semantics: the pattern may
    match a substring; analyzed as ``.*pattern.*`` over any characters).
    """
    if mode not in ("fullmatch", "search"):
        raise ValueError("mode must be 'fullmatch' or 'search'")
    ast = parse(pattern)
    try:
        redos_findings = redos.lint(ast)
    except RecursionError:
        raise PatternTooLarge(
            "pattern nesting too deep to lint (recursion limit)"
        ) from None
    if mode == "search":
        ast = _wrap_search(ast)
    dfa = automata.compile_ast(ast, max_states=max_dfa_states)
    min_dfa = automata.minimize(dfa)

    satisfiable = automata.is_satisfiable(dfa)
    witnesses = automata.shortest_witnesses(dfa, k=witness_count)
    matches, enum_truncated = automata.enumerate_accepted(
        dfa, max_len=enum_max_len, limit=enum_limit
    )
    misses = (
        automata.near_misses(dfa, witnesses, limit=near_miss_limit)
        if satisfiable
        else []
    )
    min_len, max_len, finite = automata.length_bounds(dfa)
    prefix = automata.common_prefix(dfa) if satisfiable else ""

    size_warnings = []
    if dfa.n_states > WARN_DFA_STATES:
        size_warnings.append(
            f"DFA has {dfa.n_states} states (> {WARN_DFA_STATES}); "
            "constrained-decoding engines build per-step token masks over "
            "this automaton, so compile/mask time grows with state count"
        )
    if (
        min_dfa.n_states > WARN_RATIO_MIN_STATES
        and min_dfa.n_states > WARN_STATE_TO_PATTERN_RATIO * len(pattern)
    ):
        size_warnings.append(
            f"minimized DFA has {min_dfa.n_states} states for a "
            f"{len(pattern)}-character pattern; bounded repeats and wide "
            "alternations expand combinatorially (heuristic threshold "
            f"{WARN_STATE_TO_PATTERN_RATIO}x)"
        )

    report = {
        "pattern": pattern,
        "mode": mode,
        "satisfiable": satisfiable,
        "witnesses": witnesses,
        "sample_matches": matches,
        "enumeration_truncated": enum_truncated,
        "near_misses": misses,
        "common_prefix": prefix,
        "min_length": min_len,
        "max_length": max_len,
        "finite_language": finite,
        "dfa_states": dfa.n_states,
        "min_dfa_states": min_dfa.n_states,
        "dfa_transitions": dfa.n_transitions(),
        "size_warnings": size_warnings,
        "redos_findings": redos_findings,
        "notes": [
            (
                "full-match semantics; JSON Schema `pattern` is "
                "substring-based (use mode='search')"
                if mode == "fullmatch"
                else "search/substring semantics; edge anchors ^ and $ are "
                "honored (an anchored side is not wrapped)"
            ),
            "ReDoS findings are heuristic hazard detection, not a safety "
            "proof; a clean report does not prove the pattern is safe",
        ],
        "limits": {
            "max_repeat_expansion": automata.MAX_REPEAT_EXPANSION,
            "max_expanded_nodes": automata.MAX_EXPANDED_NODES,
            "max_dfa_states": max_dfa_states,
        },
    }
    if vocab_path:
        vocab = load_vocab(vocab_path)
        report["dead_tokens"] = dead_tokens(dfa, vocab)
        report["dead_tokens"]["note"] = (
            "dead = token text occurs in no accepted string (provably "
            "unusable); live = substring-level possibility only, not a "
            "tokenization guarantee"
        )
    return report


def analyze_intersection(
    pattern_a: str,
    pattern_b: str,
    *,
    mode: str = "fullmatch",
    max_dfa_states: int = automata.MAX_DFA_STATES,
) -> dict:
    """Check whether two patterns accept at least one common string."""
    if mode not in ("fullmatch", "search"):
        raise ValueError("mode must be 'fullmatch' or 'search'")
    ast_a = parse(pattern_a)
    ast_b = parse(pattern_b)
    if mode == "search":
        ast_a = _wrap_search(ast_a)
        ast_b = _wrap_search(ast_b)
    dfa_a = automata.compile_ast(ast_a, max_states=max_dfa_states)
    dfa_b = automata.compile_ast(ast_b, max_states=max_dfa_states)
    witness = automata.intersect_witness(dfa_a, dfa_b)
    return {
        "pattern_a": pattern_a,
        "pattern_b": pattern_b,
        "mode": mode,
        "intersection_empty": witness is None,
        "witness": witness,
    }
