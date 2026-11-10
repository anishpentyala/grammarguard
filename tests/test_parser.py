"""Parser unit tests: supported subset, classes, escapes, quantifiers."""

import pytest

from grammarguard.regex_ast import (
    Alt,
    Class,
    Dot,
    EmptyStr,
    Literal,
    RegexSyntaxError,
    Repeat,
    Seq,
    UnsupportedFeature,
    parse,
)


def test_literal_sequence():
    assert parse("abc") == Seq((Literal("a"), Literal("b"), Literal("c")))


def test_single_literal_not_wrapped():
    assert parse("a") == Literal("a")


def test_empty_pattern():
    assert parse("") == EmptyStr()


def test_anchors_preserved_as_nodes():
    from grammarguard.regex_ast import AnchorEnd, AnchorStart

    assert parse("^abc$") == Seq(
        (
            AnchorStart(),
            Seq((Literal("a"), Literal("b"), Literal("c"))),
            AnchorEnd(),
        )
    )
    assert parse("^$") == Seq((AnchorStart(), EmptyStr(), AnchorEnd()))


def test_anchors_noop_under_fullmatch():
    import re

    from grammarguard import compile_ast, simulate

    dfa = compile_ast(parse("^abc$"))
    assert simulate(dfa, "abc") and not simulate(dfa, "xabc")
    assert re.fullmatch("^abc$", "abc")


def test_trailing_escaped_dollar_kept():
    node = parse("a\\$")
    assert node == Seq((Literal("a"), Literal("$")))


def test_alternation():
    assert parse("a|b") == Alt((Literal("a"), Literal("b")))
    assert parse("a|b|c") == Alt((Literal("a"), Literal("b"), Literal("c")))


def test_alternation_precedence():
    node = parse("ab|cd")
    assert node == Alt(
        (Seq((Literal("a"), Literal("b"))), Seq((Literal("c"), Literal("d"))))
    )


def test_group_capture_and_noncapturing():
    assert parse("(a)") == Literal("a")
    assert parse("(?:a)") == Literal("a")


def test_empty_group():
    assert parse("()") == EmptyStr()


def test_star_plus_quest():
    assert parse("a*") == Repeat(Literal("a"), 0, None)
    assert parse("a+") == Repeat(Literal("a"), 1, None)
    assert parse("a?") == Repeat(Literal("a"), 0, 1)


def test_braced_quantifiers():
    assert parse("a{3}") == Repeat(Literal("a"), 3, 3)
    assert parse("a{2,}") == Repeat(Literal("a"), 2, None)
    assert parse("a{2,5}") == Repeat(Literal("a"), 2, 5)


def test_lazy_and_possessive_suffixes_accepted():
    assert parse("a*?") == Repeat(Literal("a"), 0, None)
    assert parse("a++") == Repeat(Literal("a"), 1, None)
    assert parse("a{2,5}?") == Repeat(Literal("a"), 2, 5)


def test_invalid_brace_is_literal():
    node = parse("a{b}")
    assert node == Seq((Literal("a"), Literal("{"), Literal("b"), Literal("}")))


def test_quantifier_min_gt_max():
    with pytest.raises(RegexSyntaxError):
        parse("a{5,2}")


def test_nothing_to_repeat():
    with pytest.raises(RegexSyntaxError):
        parse("*a")
    with pytest.raises(RegexSyntaxError):
        parse("+a")


def test_dot():
    assert parse(".") == Dot()


def test_class_basic_and_range():
    node = parse("[ab0-9]")
    assert isinstance(node, Class) and not node.negated
    assert node.ranges == ((ord("0"), ord("9")), (ord("a"), ord("b")))


def test_class_negation():
    node = parse("[^a]")
    assert node.negated


def test_class_predefined_and_escapes():
    node = parse("[\\d_]")
    assert (ord("0"), ord("9")) in node.ranges
    assert (ord("_"), ord("_")) in node.ranges


def test_class_negated_predefined_inside():
    # [\D] == everything except digits
    node = parse("[\\D]")
    assert not node.negated
    # complement pre-applied: '5' must not be covered
    assert all(not (lo <= ord("5") <= hi) for lo, hi in node.ranges)
    assert any(lo <= ord("a") <= hi for lo, hi in node.ranges)


def test_class_literal_bracket_first():
    node = parse("[]a]")
    assert (ord("]"), ord("]")) in node.ranges
    assert (ord("a"), ord("a")) in node.ranges


def test_class_range_min_gt_max():
    with pytest.raises(RegexSyntaxError):
        parse("[z-a]")


def test_escapes():
    assert parse("\\n") == Literal("\n")
    assert parse("\\t") == Literal("\t")
    assert parse("\\x41") == Literal("A")
    assert parse("\\u0041") == Literal("A")
    assert parse("\\U00000041") == Literal("A")
    assert parse("\\.") == Literal(".")
    assert parse("\\*") == Literal("*")
    assert parse("\\\\") == Literal("\\")


def test_predefined_classes():
    d = parse("\\d")
    assert isinstance(d, Class) and not d.negated
    nd = parse("\\D")
    assert nd.negated
    w = parse("\\w")
    assert not w.negated
    s = parse("\\s")
    assert not s.negated


def test_mid_pattern_anchors_unsupported():
    with pytest.raises(UnsupportedFeature):
        parse("a^b")
    with pytest.raises(UnsupportedFeature):
        parse("a$b")


def test_unsupported_constructs_named():
    cases = [
        ("a(?=b)", "lookahead"),
        ("a(?!b)", "negative lookahead"),
        ("(?<=a)b", "lookbehind"),
        ("(?<!a)b", "negative lookbehind"),
        ("(?P<name>a)", "named group"),
        ("(?P=name)", "named backreference"),
        ("(?>a)", "atomic group"),
        ("(?#comment)", "comment group"),
        ("(?(1)a|b)", "conditional group"),
        ("(?i)abc", "inline flags"),
        ("\\bword\\b", "boundary"),
        ("(a)\\1", "backreference"),
        ("\\Aabc\\Z", "anchor"),
    ]
    for pattern, feature in cases:
        with pytest.raises(UnsupportedFeature) as excinfo:
            parse(pattern)
        assert feature.split()[0] in str(excinfo.value)


def test_unbalanced_parens():
    with pytest.raises(RegexSyntaxError):
        parse("(a")
    with pytest.raises(RegexSyntaxError):
        parse("a)")


def test_unterminated_class():
    with pytest.raises(RegexSyntaxError):
        parse("[ab")


def test_trailing_backslash():
    with pytest.raises(RegexSyntaxError):
        parse("ab\\")


def test_nested_groups_and_repeats():
    node = parse("(a(b|c))*")
    assert isinstance(node, Repeat)
    assert node.min_count == 0 and node.max_count is None
