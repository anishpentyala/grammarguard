"""Witnesses, enumeration, near misses, prefix, bounds, intersection."""

import itertools
import re

from grammarguard import (
    analyze,
    analyze_intersection,
    common_prefix,
    compile_ast,
    enumerate_accepted,
    intersect_witness,
    is_satisfiable,
    length_bounds,
    near_misses,
    shortest_witnesses,
    simulate,
)
from grammarguard.regex_ast import parse


def dfa_of(pattern):
    return compile_ast(parse(pattern))


def brute_force_shortest(pattern, alphabet, max_len):
    for length in range(0, max_len + 1):
        for tup in itertools.product(alphabet, repeat=length):
            s = "".join(tup)
            if re.fullmatch(pattern, s):
                return s
    return None


def test_witness_is_shortest_brute_force():
    cases = [
        ("ab+c", "ab"),
        ("(?:cat|dog)s?", "abcd"),
        ("\\d{2,4}", "ab"),
        ("a|b|cd", "ab"),
    ]
    for pattern, alphabet in cases:
        dfa = dfa_of(pattern)
        witnesses = shortest_witnesses(dfa, k=1)
        expected = brute_force_shortest(pattern, alphabet, 6)
        if expected is None:
            # brute force over a limited alphabet found nothing; any witness
            # must still be a genuine match (may use chars outside alphabet)
            for w in witnesses:
                assert re.fullmatch(pattern, w)
        else:
            assert witnesses and len(witnesses[0]) == len(expected)
            assert re.fullmatch(pattern, witnesses[0])


def test_witnesses_multiple_lengths():
    dfa = dfa_of("a|bb|ccc")
    assert shortest_witnesses(dfa, k=3) == ["a", "bb", "ccc"]


def test_witnesses_unsatisfiable():
    assert shortest_witnesses(dfa_of("[^\\x00-\\U0010FFFF]"), k=3) == []


def test_enumeration_all_match_and_complete_for_finite():
    pattern = "(?:cat|dog)s?"
    dfa = dfa_of(pattern)
    matches, truncated = enumerate_accepted(dfa, max_len=8, limit=64)
    assert not truncated
    assert sorted(matches) == ["cat", "cats", "dog", "dogs"]
    for m in matches:
        assert re.fullmatch(pattern, m)


def test_enumeration_truncation_flag():
    dfa = dfa_of("a+")
    matches, truncated = enumerate_accepted(dfa, max_len=10, limit=3)
    assert truncated and len(matches) == 3
    assert all(set(m) == {"a"} for m in matches)


def test_enumeration_respects_max_len():
    dfa = dfa_of("a+")
    matches, _ = enumerate_accepted(dfa, max_len=4, limit=100)
    assert matches == ["a", "aa", "aaa", "aaaa"]


def test_enumeration_empty_string_included():
    dfa = dfa_of("a?")
    matches, _ = enumerate_accepted(dfa, max_len=2, limit=10)
    assert "" in matches and "a" in matches


def test_near_misses_all_rejected():
    pattern = "\\d{4}-\\d{2}-\\d{2}"
    dfa = dfa_of(pattern)
    witnesses = shortest_witnesses(dfa, k=1)
    misses = near_misses(dfa, witnesses, limit=8)
    assert len(misses) >= 4
    for m in misses:
        assert not simulate(dfa, m)
        assert re.fullmatch(pattern, m) is None


def test_common_prefix():
    assert common_prefix(dfa_of("abc|abd")) == "ab"
    assert common_prefix(dfa_of("https://(a|b)\\.example\\.com")) == "https://"
    assert common_prefix(dfa_of("a*z")) == ""
    assert common_prefix(dfa_of("a+")) == "a"
    assert common_prefix(dfa_of("[ab]x")) == ""


def test_length_bounds():
    assert length_bounds(dfa_of("a*z")) == (1, None, False)
    assert length_bounds(dfa_of("a{2,4}")) == (2, 4, True)
    assert length_bounds(dfa_of("(?:cat|dog)s?")) == (3, 4, True)
    assert length_bounds(dfa_of("[^\\x00-\\U0010FFFF]")) == (None, None, True)
    assert length_bounds(dfa_of("\\d{4}-\\d{2}-\\d{2}")) == (10, 10, True)


def test_finite_vs_infinite():
    assert length_bounds(dfa_of("true|false"))[2] is True
    assert length_bounds(dfa_of("(ab)+"))[2] is False


def test_intersection():
    d1, d2 = dfa_of("a+"), dfa_of("aa+")
    assert intersect_witness(d1, d2) == "aa"
    d3, d4 = dfa_of("a+"), dfa_of("b+")
    assert intersect_witness(d3, d4) is None
    d5, d6 = dfa_of("\\d{4}"), dfa_of("2026.*")
    w = intersect_witness(d5, d6)
    assert w is not None and w.startswith("2026") and len(w) == 4


def test_intersection_with_literal_prefix_constraint():
    # common prefix + another pattern
    res = analyze_intersection("https://.*", "https://a\\.example\\.com")
    assert res["intersection_empty"] is False
    assert res["witness"] == "https://a.example.com"


def test_analyze_report_shape():
    r = analyze("\\d{4}-\\d{2}-\\d{2}")
    for key in (
        "pattern", "mode", "satisfiable", "witnesses", "sample_matches",
        "near_misses", "common_prefix", "min_length", "max_length",
        "finite_language", "dfa_states", "min_dfa_states",
        "dfa_transitions", "size_warnings", "redos_findings", "notes",
        "limits",
    ):
        assert key in r
    assert r["satisfiable"] and r["min_length"] == 10 and r["max_length"] == 10
    assert r["redos_findings"] == []
    assert all(re.fullmatch(r["pattern"], m) for m in r["sample_matches"])


def test_analyze_search_mode():
    # JSON Schema `pattern` semantics: unanchored
    r_full = analyze("\\d{4}", mode="fullmatch")
    assert r_full["min_length"] == 4 and r_full["max_length"] == 4
    r_search = analyze("\\d{4}", mode="search")
    assert r_search["max_length"] is None  # any surrounding text allowed
    assert r_search["satisfiable"]


def test_is_satisfiable():
    assert is_satisfiable(dfa_of("a+"))
    assert not is_satisfiable(dfa_of("[^\\x00-\\U0010FFFF]"))
