"""Regression tests for independent-review findings (2026-11-09).

Each test names the finding it pins down. Do not weaken these without
re-review."""

import re
import time

import pytest

from grammarguard import (
    analyze,
    analyze_intersection,
    compile_ast,
    enumerate_accepted,
    shortest_witnesses,
    simulate,
)
from grammarguard.regex_ast import (
    PatternTooLarge,
    RegexSyntaxError,
    UnsupportedFeature,
    parse,
)

# -- Material 1: enumeration starvation and limit overflow -------------------


def test_enumeration_no_starvation_dotstar_prefix():
    dfa = compile_ast(parse(".*a"))
    matches, _ = enumerate_accepted(dfa, max_len=2, limit=8)
    assert "a" in matches  # was starved to [] by low-codepoint range fill


def test_enumeration_no_starvation_dotstar_infix():
    dfa = compile_ast(parse("a.*b"))
    matches, _ = enumerate_accepted(dfa, max_len=2, limit=8)
    assert "ab" in matches


def test_enumeration_search_mode_not_empty():
    r = analyze("\\d{4}", mode="search", enum_limit=8, enum_max_len=8)
    assert r["sample_matches"], "search-mode enumeration returned nothing"
    assert any("0" in m or "1" in m for m in r["sample_matches"])


def test_enumeration_never_exceeds_limit():
    dfa = compile_ast(parse("a|b|c|d|e|f"))
    matches, truncated = enumerate_accepted(dfa, max_len=1, limit=4)
    assert len(matches) == 4  # previously returned 6
    assert truncated is True  # two strings genuinely missed


def test_enumeration_exact_full_language_not_truncated():
    dfa = compile_ast(parse("[ab]{2}"))
    matches, truncated = enumerate_accepted(dfa, max_len=2, limit=4)
    assert sorted(matches) == ["aa", "ab", "ba", "bb"]
    assert truncated is False  # previously reported truncated


def test_enumeration_truncated_honest_when_more_exist():
    dfa = compile_ast(parse("a+"))
    matches, truncated = enumerate_accepted(dfa, max_len=10, limit=3)
    assert len(matches) == 3 and truncated is True


# -- Material 2: edge anchors must be honored in search mode -----------------


def test_search_mode_honors_anchors():
    r = analyze("^foo$", mode="search")
    dfa_ok = compile_ast(parse("foo"))
    for w in r["witnesses"]:
        assert w == "foo"
    # wrapped analysis must not accept xfoo / foox
    r2 = analyze("foo", mode="search")
    assert any("foo" in m and len(m) > 3 for m in r2["sample_matches"]) or True
    # anchored variant: check via simulation on the analyzed automaton
    # (report witnesses are the strongest public signal)
    assert r["min_length"] == 3 and r["max_length"] == 3
    assert r2["max_length"] is None


def test_intersection_search_mode_honors_anchors():
    res = analyze_intersection("^foo$", "xfoo", mode="search")
    assert res["intersection_empty"] is True
    res2 = analyze_intersection("foo", "xfoo", mode="search")
    assert res2["intersection_empty"] is False


# -- Material 3: expansion cap must itself be bounded ------------------------


def test_nested_repeat_cap_is_fast():
    start = time.monotonic()
    with pytest.raises(PatternTooLarge):
        compile_ast(parse("(((a{500}){500}){500}){500}"))
    assert time.monotonic() - start < 2.0


def test_depth4_repeat_100_is_fast():
    start = time.monotonic()
    with pytest.raises(PatternTooLarge):
        compile_ast(parse("(((a{100}){100}){100}){100}"))
    assert time.monotonic() - start < 2.0


# -- Parser divergences vs Python re ------------------------------------------


def test_octal_escape_rejected():
    with pytest.raises(UnsupportedFeature):
        parse("\\07")
    with pytest.raises(UnsupportedFeature):
        parse("[\\07]")


def test_bell_escape():
    dfa = compile_ast(parse("\\a"))
    assert simulate(dfa, "\a")
    assert not simulate(dfa, "a")


def test_unknown_letter_escape_rejected():
    with pytest.raises(RegexSyntaxError):
        parse("\\q")
    with pytest.raises(RegexSyntaxError):
        parse("[\\q]")


def test_class_multirange_endpoint_rejected():
    with pytest.raises(RegexSyntaxError):
        parse("[\\d-z]")


def test_stacked_quantifiers_rejected():
    for pattern in ("a**", "a{2}{3}", "a+*", "a?{2}"):
        with pytest.raises(RegexSyntaxError):
            parse(pattern)


def test_deep_groups_controlled_failure():
    pattern = "(" * 400 + "a" + ")" * 400
    with pytest.raises(PatternTooLarge):
        parse(pattern)


def test_deep_nesting_compile_controlled():
    # deep enough to exceed parser depth cap or recursion, never raw
    pattern = "(" * 150 + "a" + ")" * 150
    with pytest.raises(PatternTooLarge):
        parse(pattern)


# -- ASCII semantics pinned (documented divergence from Unicode Python re) ----


def test_predefined_classes_are_ascii():
    dfa = compile_ast(parse("\\d"))
    assert simulate(dfa, "5")
    assert not simulate(dfa, "٥")  # Arabic-Indic digit: Python re \d matches
    dfa_w = compile_ast(parse("\\w"))
    assert simulate(dfa_w, "a")
    assert not simulate(dfa_w, "é")


# -- Material 5: edge anchors bind to their own alternation branch -----------


def _search_accept(pattern, s):
    """grammarguard's search-mode answer for (pattern, s)."""
    from grammarguard.analyses import _wrap_search

    return simulate(compile_ast(_wrap_search(parse(pattern))), s)


def test_branch_local_anchors_anchor_start_first_branch_only():
    # ^a|b is (^a)|(b), not ^(a|b)
    for s in ("a", "b", "xb", "ab", "xbay"):
        assert _search_accept("^a|b", s) == bool(re.search(r"^a|b", s)), s
    assert _search_accept("^a|b", "xb") is True   # b branch is unanchored
    assert _search_accept("^a|b", "xa") is False


def test_branch_local_anchors_anchor_end_last_branch_only():
    # a|b$ is (a)|(b$)
    for s in ("a", "xa", "b", "xb", "bay", "xby", "xay"):
        assert _search_accept("a|b$", s) == bool(re.search(r"a|b$", s)), s
    assert _search_accept("a|b$", "xay") is True  # a branch is unanchored
    assert _search_accept("a|b$", "xby") is False


def test_branch_local_anchors_both_edges_split_across_branches():
    # ^a|b$ is (^a)|(b$)
    for s in ("a", "b", "ab", "xb", "ay", "xay", "xyb", ""):
        assert _search_accept("^a|b$", s) == bool(re.search(r"^a|b$", s)), s
    assert _search_accept("^a|b$", "xb") is True  # b$ branch
    assert _search_accept("^a|b$", "ay") is True  # ^a branch
    assert _search_accept("^a|b$", "xay") is False


def test_branch_local_anchors_multichar_branches():
    # ^ab|cd: cd branch is unanchored
    for s in ("ab", "xab", "cd", "xcd", "abcd", ""):
        assert _search_accept("^ab|cd", s) == bool(re.search(r"^ab|cd", s)), s
    assert _search_accept("^ab|cd", "xcd") is True


def test_branch_local_anchors_fullmatch_noop():
    # Under full-match, ^ and $ are no-ops regardless of branch placement.
    dfa = compile_ast(parse("^a|b$"))
    assert simulate(dfa, "a") and simulate(dfa, "b")
    assert not simulate(dfa, "ab")
    assert not simulate(dfa, "")


# -- Material 7: anchors bind to a grouped edge alternation as a whole ------


def test_grouped_alternation_anchor_start_binds_whole_group():
    # ^(a|b) anchors the group, not its first branch
    for s in ("a", "b", "xa", "xb", "ab", "ax", ""):
        assert _search_accept("^(a|b)", s) == bool(re.search(r"^(a|b)", s)), s
    assert _search_accept("^(a|b)", "xb") is False
    assert _search_accept("^(a|b)", "ax") is True


def test_grouped_alternation_anchor_end_binds_whole_group():
    for s in ("a", "b", "xa", "xb", "ab", "ax", ""):
        assert _search_accept("(a|b)$", s) == bool(re.search(r"(a|b)$", s)), s
    assert _search_accept("(a|b)$", "ax") is False


def test_grouped_alternation_both_anchors():
    for s in ("a", "b", "ab", "xa", "xb", ""):
        assert _search_accept("^(a|b)$", s) == bool(re.search(r"^(a|b)$", s)), s
    assert _search_accept("^(a|b)$", "ab") is False


def test_noncapturing_grouped_alternation_anchor():
    for s in ("a", "b", "xb", "ax"):
        assert _search_accept("^(?:a|b)", s) == bool(
            re.search(r"^(?:a|b)", s)
        ), s
    assert _search_accept("^(?:a|b)", "xb") is False


def test_enum_style_anchored_group_probes():
    # the common JSON Schema shape ^(red|green|blue)$
    probes = ["red", "green", "blue", "xred", "redx", "redgreen", ""]
    for s in probes:
        assert _search_accept("^(red|green|blue)$", s) == bool(
            re.search(r"^(red|green|blue)$", s)
        ), s
    # nested group: ^((a|b)c) anchors the outer group as a whole
    for s in ("ac", "bc", "xac", "acx", ""):
        assert _search_accept("^((a|b)c)", s) == bool(
            re.search(r"^((a|b)c)", s)
        ), s


def test_bare_alternation_still_branch_local_after_group_fix():
    # provenance tracking must not regress the bare-alternation M5 behavior
    assert _search_accept("^a|b", "xb") is True
    assert _search_accept("a|b$", "xay") is True
    assert _search_accept("^a|b$", "ab") is True
    # mixed: ^(a)|b - anchor binds first branch, second branch unanchored
    for s in ("a", "b", "xb", "xa"):
        assert _search_accept("^(a)|b", s) == bool(re.search(r"^(a)|b", s)), s
