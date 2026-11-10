"""ReDoS heuristic lint: known hazard shapes flagged, clean patterns quiet."""

import pytest

from grammarguard import redos_lint
from grammarguard.regex_ast import parse


def rules_for(pattern):
    return {(f["rule"], f["risk"]) for f in redos_lint(parse(pattern))}


def test_nested_unbounded_flagged_high():
    assert ("nested-unbounded", "high") in rules_for("(a+)+")
    assert ("nested-unbounded", "high") in rules_for("([a-zA-Z]+)*")
    assert ("nested-unbounded", "high") in rules_for("(?:\\d+)*")


def test_nullable_loop_flagged_high():
    assert ("nullable-loop", "high") in rules_for("(a?)*")
    assert ("nullable-loop", "high") in rules_for("(|ab)*")


def test_overlapping_alternatives_flagged_high():
    assert ("overlapping-alternatives", "high") in rules_for("(a|a)*")
    assert ("overlapping-alternatives", "high") in rules_for("(a|aa)+")
    assert ("overlapping-alternatives", "high") in rules_for("(\\d|\\w)+")


def test_adjacent_overlap_flagged_medium():
    assert ("adjacent-overlap", "medium") in rules_for(".*.*")
    assert ("adjacent-overlap", "medium") in rules_for("a*a")


def test_clean_patterns_quiet():
    clean = [
        "\\d{4}-\\d{2}-\\d{2}",
        "true|false",
        "[a-z]+@[a-z]+\\.[a-z]+",
        "(?:cat|dog)s?",
        "https://[a-z0-9.-]+",
        "[A-Z][a-z]*",
        "\\w+@\\w+\\.com",  # adjacent \w* and literal @ and literal .: disjoint
    ]
    for pattern in clean:
        findings = redos_lint(parse(pattern))
        assert findings == [], f"{pattern}: unexpected findings {findings}"


def test_findings_have_detail_and_dedupe():
    findings = redos_lint(parse("(a+)+"))
    for f in findings:
        assert f["detail"]
        assert f["risk"] in ("high", "medium")
    keys = [(f["rule"], f["risk"]) for f in findings]
    assert len(keys) == len(set(keys))
