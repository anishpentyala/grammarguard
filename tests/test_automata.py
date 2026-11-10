"""DFA construction, simulation and minimization against Python re."""

import re

from grammarguard import compile_ast, minimize, simulate
from grammarguard.regex_ast import parse


def check(pattern, accept, reject):
    dfa = compile_ast(parse(pattern))
    for s in accept:
        assert simulate(dfa, s), f"{pattern!r} should accept {s!r}"
        assert re.fullmatch(pattern, s) is not None
    for s in reject:
        assert not simulate(dfa, s), f"{pattern!r} should reject {s!r}"
        assert re.fullmatch(pattern, s) is None


def test_literal():
    check("abc", ["abc"], ["ab", "abcd", "xbc", "ABC", ""])


def test_star_plus():
    check("ab*c", ["ac", "abc", "abbbc"], ["a", "bc", "abbd"])
    check("a+", ["a", "aa", "aaaaaaa"], ["", "b", "ab"])


def test_alternation():
    check("true|false", ["true", "false"], ["True", "tru", "falsee", ""])


def test_optional_suffix():
    check("(?:cat|dog)s?", ["cat", "cats", "dog", "dogs"], ["ca", "catss", ""])


def test_bounded_repeat():
    check("a{2,4}", ["aa", "aaa", "aaaa"], ["a", "aaaaa", ""])
    check("a{3}", ["aaa"], ["aa", "aaaa"])
    check("a{2,}", ["aa", "aaa", "aaaaaaaaaa"], ["a", ""])


def test_date_pattern():
    check(
        "\\d{4}-\\d{2}-\\d{2}",
        ["2026-11-09", "0000-00-00"],
        ["2026-1-9", "202a-11-09", "2026-11-09 ", " 2026-11-09"],
    )


def test_char_classes():
    check("[a-c]+", ["a", "abc", "cba"], ["d", "A", "abcd"])
    check("[^a-c]+", ["d", "XYZ", "9"], ["a", "dad"])
    check("[\\dA-F]+", ["9AF", "F0"], ["G0", "af"])


def test_dot():
    check("a.c", ["abc", "a-c", "a c"], ["ac", "a\nc", "abbc"])


def test_empty_language():
    # complement of the full codepoint range
    dfa = compile_ast(parse("[^\\x00-\\U0010FFFF]"))
    assert not simulate(dfa, "")
    assert not simulate(dfa, "a")


def test_empty_string_match():
    check("a*", ["", "a", "aa"], ["b"])
    check("(?:)", [""], ["a"])


def test_unicode_literals():
    check("héllo", ["héllo"], ["hello"])


def test_minimize_preserves_language_and_shrinks():
    pattern = "(a|b)*abb"
    dfa = compile_ast(parse(pattern))
    md = minimize(dfa)
    assert md.n_states <= dfa.n_states
    cases = ["abb", "aabb", "babb", "ab", "abbb", "abba", "", "bab"]
    for s in cases:
        assert simulate(md, s) == simulate(dfa, s) == bool(
            re.fullmatch(pattern, s)
        )


def test_minimize_actually_minimal_for_redundant_alternation():
    # (a|a) is redundant; minimization should collapse equivalent states
    dfa = compile_ast(parse("(?:a|a)(?:a|a)"))
    md = minimize(dfa)
    plain = compile_ast(parse("aa"))
    assert md.n_states == minimize(plain).n_states


def test_transitions_binary_search_edges():
    # ranges around the dot hole at \n
    check(".", ["a", "\t", "€"], ["\n", ""])
