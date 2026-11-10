"""Error handling and bounded-execution limits."""

import pytest

from grammarguard import analyze, compile_ast, enumerate_accepted
from grammarguard.regex_ast import (
    PatternTooLarge,
    RegexError,
    UnsupportedFeature,
    parse,
)


def test_repeat_expansion_limit():
    with pytest.raises(PatternTooLarge):
        compile_ast(parse("a{1000}"))


def test_nested_expansion_limit():
    with pytest.raises(PatternTooLarge):
        compile_ast(parse("(?:a{500}){500}"))


def test_dfa_state_cap():
    # many wide alternation branches -> state growth; tiny cap forces refusal
    pattern = "(?:[a-z]{2}){6}"
    with pytest.raises(PatternTooLarge):
        compile_ast(parse(pattern), max_states=10)


def test_enum_max_len_cap():
    dfa = compile_ast(parse("a+"))
    with pytest.raises(ValueError):
        enumerate_accepted(dfa, max_len=300)


def test_analyze_non_string_pattern():
    with pytest.raises(RegexError):
        analyze(123)  # type: ignore[arg-type]


def test_unsupported_feature_lists_construct():
    with pytest.raises(UnsupportedFeature) as exc:
        parse("(?<=look)behind")
    assert "lookbehind" in str(exc.value)


def test_limits_reported_in_analysis():
    r = analyze("a+")
    assert r["limits"]["max_repeat_expansion"] > 0
    assert r["limits"]["max_dfa_states"] > 0
    assert r["limits"]["max_expanded_nodes"] > 0


def test_size_warning_fires_on_large_automaton():
    from grammarguard import analyses

    orig = analyses.WARN_DFA_STATES
    try:
        analyses.WARN_DFA_STATES = 2
        r = analyze("abc")
        assert r["size_warnings"], "expected size warning with tiny threshold"
        assert any("DFA has" in w for w in r["size_warnings"])
    finally:
        analyses.WARN_DFA_STATES = orig


def test_no_size_warning_on_small_automaton():
    r = analyze("\\d{4}-\\d{2}-\\d{2}")
    assert r["size_warnings"] == []
