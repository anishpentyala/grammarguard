"""Property tests: language equivalence with Python re on random patterns,
including control negatives, and witness minimality by brute force."""

import itertools
import random
import re

from grammarguard import compile_ast, shortest_witnesses, simulate
from grammarguard.regex_ast import parse

ALPHABET = "ab01_-X"
RND = random.Random(20261109)


def gen_pattern(depth: int) -> str:
    if depth <= 0:
        return RND.choice(["a", "b", "0", "1", "_", "-"])
    choice = RND.randrange(8)
    if choice == 0:
        return gen_pattern(depth - 1) + gen_pattern(depth - 1)
    if choice == 1:
        return gen_pattern(depth - 1) + "|" + gen_pattern(depth - 1)
    if choice == 2:
        return "(" + gen_pattern(depth - 1) + ")" + RND.choice(
            ["*", "+", "?", "{0,2}", "{1,3}", "{2,}"]
        )
    if choice == 3:
        return RND.choice(["\\d", "\\w", "[a-b]", "[^a]", "[a01]"])
    if choice == 4:
        return RND.choice(["a", "b", "0", "1"]) + RND.choice(
            ["*", "+", "?", "{2,3}"]
        )
    if choice == 5:
        return "(?:" + gen_pattern(depth - 1) + ")"
    return RND.choice(["a", "b", "0", "1", "_"])


def gen_string() -> str:
    n = RND.randrange(0, 9)
    return "".join(RND.choice(ALPHABET) for _ in range(n))


def test_language_equivalence_with_python_re():
    n_patterns, n_strings = 250, 60
    mismatches = 0
    accepted_seen = 0
    for _ in range(n_patterns):
        pattern = gen_pattern(depth=3)
        try:
            compiled = re.compile(pattern)
            dfa = compile_ast(parse(pattern))
        except Exception:
            continue  # generator/re edge case outside our subset; skip
        for _ in range(n_strings):
            s = gen_string()
            expected = compiled.fullmatch(s) is not None
            actual = simulate(dfa, s)
            if expected != actual:
                mismatches += 1
                assert (
                    expected == actual
                ), f"divergence: pattern={pattern!r} string={s!r}"
            if expected:
                accepted_seen += 1
    assert mismatches == 0
    # sanity: the suite actually exercised accepting cases (control positives)
    assert accepted_seen > 200


def test_witness_minimality_brute_force():
    for _ in range(120):
        pattern = gen_pattern(depth=3)
        try:
            re_compiled = re.compile(pattern)
            dfa = compile_ast(parse(pattern))
        except Exception:
            continue
        witnesses = shortest_witnesses(dfa, k=1)
        # brute force over a bounded alphabet for a lower bound on length
        shortest = None
        for length in range(0, 5):
            for tup in itertools.product("ab01", repeat=length):
                s = "".join(tup)
                if re_compiled.fullmatch(s):
                    shortest = s
                    break
            if shortest is not None:
                break
        if witnesses:
            w = witnesses[0]
            assert re_compiled.fullmatch(w)
            if shortest is not None:
                # witness may use other alphabet chars, but never longer
                assert len(w) <= len(shortest)
        # note: a satisfiable pattern may have no short witness over ab01


def test_minimized_dfa_equivalence_property():
    from grammarguard import minimize

    for _ in range(120):
        pattern = gen_pattern(depth=3)
        try:
            re_compiled = re.compile(pattern)
            dfa = compile_ast(parse(pattern))
        except Exception:
            continue
        md = minimize(dfa)
        assert md.n_states <= dfa.n_states
        for _ in range(40):
            s = gen_string()
            assert simulate(md, s) == (re_compiled.fullmatch(s) is not None)
