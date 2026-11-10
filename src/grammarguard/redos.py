"""Heuristic ReDoS (catastrophic backtracking) hazard lint.

IMPORTANT - scope and honesty. This is a *heuristic hazard detector*, not a
safety proof. A clean report does NOT mean a pattern is safe against
backtracking blowups; it means none of the known structural hazard shapes
below were found. Guided-decoding engines (xgrammar, outlines) generally do
not backtrack, but the same patterns are often reused with backtracking
engines (Python ``re``, validator libraries), and vLLM added regex compile
timeouts precisely because pathological patterns reach engines. See:
https://github.com/vllm-project/vllm/security/advisories/GHSA-rwxx-mrjm-wc2m

Rules (each returns a finding with risk "high" or "medium"):

  nested-unbounded        an unbounded quantifier inside another unbounded
                          quantifier, e.g. (a+)+  ([a-zA-Z]+)*
  nullable-loop           an unbounded quantifier whose body can match the
                          empty string, e.g. (a?)*  (|ab)*
  overlapping-alternatives an unbounded quantifier directly over an
                          alternation whose branches share a first
                          character range, e.g. (a|a)*  (a|aa)+
  adjacent-overlap        an unbounded quantified atom followed by another
                          atom whose first character range overlaps,
                          e.g. .*.*  a*a  \\d+\\w
"""

from __future__ import annotations

from typing import List

from .regex_ast import (
    Alt,
    AnchorEnd,
    AnchorStart,
    Class,
    Dot,
    DOT_RANGES,
    EmptyStr,
    Literal,
    Repeat,
    Seq,
    ranges_overlap,
)


def nullable(node) -> bool:
    if isinstance(node, (EmptyStr, AnchorStart, AnchorEnd)):
        return True
    if isinstance(node, (Literal, Dot, Class)):
        return False
    if isinstance(node, Seq):
        return all(nullable(p) for p in node.parts)
    if isinstance(node, Alt):
        return any(nullable(b) for b in node.branches)
    if isinstance(node, Repeat):
        return node.min_count == 0 or nullable(node.node)
    raise TypeError(f"unknown AST node {node!r}")


def first_ranges(node) -> tuple:
    """Union of character ranges that can start a match of *node*."""
    if isinstance(node, (EmptyStr, AnchorStart, AnchorEnd)):
        return ()
    if isinstance(node, Literal):
        c = ord(node.char)
        return ((c, c),)
    if isinstance(node, Dot):
        return DOT_RANGES
    if isinstance(node, Class):
        return node.resolved()
    if isinstance(node, Seq):
        out: list = []
        for p in node.parts:
            out.extend(first_ranges(p))
            if not nullable(p):
                break
        return tuple(out)
    if isinstance(node, Alt):
        out: list = []
        for b in node.branches:
            out.extend(first_ranges(b))
        return tuple(out)
    if isinstance(node, Repeat):
        return first_ranges(node.node)
    raise TypeError(f"unknown AST node {node!r}")


def _describe(node) -> str:
    if isinstance(node, Literal):
        return repr(node.char)
    if isinstance(node, Dot):
        return "."
    if isinstance(node, Class):
        return "[...]" + (" (negated)" if node.negated else "")
    if isinstance(node, Repeat):
        hi = "" if node.max_count is None else str(node.max_count)
        return f"(...){{{node.min_count},{hi}}}"
    return type(node).__name__


def lint(node) -> List[dict]:
    findings: List[dict] = []

    def add(rule, risk, detail):
        findings.append({"rule": rule, "risk": risk, "detail": detail})

    def walk(n, inside_unbounded: bool):
        if isinstance(n, Repeat):
            unbounded = n.max_count is None
            if unbounded and inside_unbounded:
                add(
                    "nested-unbounded",
                    "high",
                    "unbounded quantifier nested inside another unbounded "
                    "quantifier; classic catastrophic-backtracking shape "
                    "(e.g. (a+)+)",
                )
            if unbounded and nullable(n.node):
                add(
                    "nullable-loop",
                    "high",
                    "unbounded quantifier whose body can match the empty "
                    "string; engines may loop or blow up",
                )
            if unbounded and isinstance(n.node, Alt):
                branches = n.node.branches
                for i in range(len(branches)):
                    for j in range(i + 1, len(branches)):
                        if ranges_overlap(
                            first_ranges(branches[i]), first_ranges(branches[j])
                        ):
                            add(
                                "overlapping-alternatives",
                                "high",
                                "alternation branches with overlapping first "
                                "characters under an unbounded quantifier "
                                "(e.g. (a|aa)+)",
                            )
                            break
                    else:
                        continue
                    break
            walk(n.node, inside_unbounded or unbounded)
            return
        if isinstance(n, Seq):
            parts = n.parts
            for i, p in enumerate(parts):
                if (
                    isinstance(p, Repeat)
                    and p.max_count is None
                    and i + 1 < len(parts)
                ):
                    follower = parts[i + 1]
                    if ranges_overlap(first_ranges(p.node), first_ranges(follower)):
                        add(
                            "adjacent-overlap",
                            "medium",
                            f"unbounded quantified {_describe(p.node)} followed "
                            f"by {_describe(follower)} with overlapping first "
                            "characters; ambiguous split points can cause "
                            "quadratic backtracking",
                        )
            for p in parts:
                walk(p, inside_unbounded)
            return
        if isinstance(n, Alt):
            for b in n.branches:
                walk(b, inside_unbounded)
            return

    walk(node, False)
    # dedupe identical (rule, risk) pairs, keep first detail
    seen = set()
    uniq = []
    for f in findings:
        key = (f["rule"], f["risk"])
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq
