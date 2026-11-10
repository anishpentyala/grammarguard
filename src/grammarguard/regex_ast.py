"""Regex parsing into a small AST for language analysis.

Supported subset (full-match semantics, mirroring guided-decoding engines
such as vLLM/SGLang structured outputs and outlines):

  * literal characters
  * ``.`` (any character except newline)
  * character classes ``[...]`` with ranges, negation, escapes and the
    predefined classes ``\\d \\D \\w \\W \\s \\S`` -- ASCII-only by design
    (``\\d`` = ``[0-9]``, ``\\w`` = ``[0-9A-Za-z_]``,
    ``\\s`` = ``[ \\t\\n\\r\\f\\v]``); this deliberately diverges from
    Python ``re`` on non-ASCII text and claims no equivalence with
    ECMA-262 or any engine dialect
  * groups ``(...)`` and ``(?:...)``
  * alternation ``|``
  * quantifiers ``* + ? {n} {n,} {n,m}`` (lazy ``??`` and possessive ``?+``
    suffixes are accepted; they are identical at the language level)
  * anchors ``^`` and ``$`` at the pattern edges only; they are no-ops
    under full-match semantics

Unsupported constructs raise :class:`UnsupportedFeature` naming the
construct: lookarounds, backreferences, named groups, atomic groups,
conditionals, inline flags, comments, ``\\b \\B \\A \\Z \\z``.

Character ranges are inclusive ``(lo, hi)`` codepoint pairs, always kept
sorted and merged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

MAX_CODEPOINT = 0x10FFFF

DIGIT_RANGES = ((0x30, 0x39),)
WORD_RANGES = ((0x30, 0x39), (0x41, 0x5A), (0x5F, 0x5F), (0x61, 0x7A))
SPACE_RANGES = ((0x09, 0x0D), (0x20, 0x20))
DOT_RANGES = ((0x00, 0x09), (0x0B, MAX_CODEPOINT))  # everything except \n
DOTALL_RANGES = ((0x00, MAX_CODEPOINT),)


class RegexError(Exception):
    """Base class for grammarguard regex errors."""


class RegexSyntaxError(RegexError):
    """Malformed pattern."""

    def __init__(self, message: str, position: int):
        super().__init__(f"{message} (at offset {position})")
        self.message = message
        self.position = position


class UnsupportedFeature(RegexError):
    """A construct outside the supported subset."""

    def __init__(self, feature: str, position: int):
        super().__init__(
            f"unsupported regex feature: {feature} (at offset {position})"
        )
        self.feature = feature
        self.position = position


class PatternTooLarge(RegexError):
    """Pattern expands beyond analysis limits."""


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Literal:
    char: str


@dataclass(frozen=True)
class Dot:
    pass


@dataclass(frozen=True)
class Class:
    ranges: Tuple[Tuple[int, int], ...]
    negated: bool = False

    def resolved(self) -> Tuple[Tuple[int, int], ...]:
        if self.negated:
            return complement_ranges(self.ranges)
        return self.ranges


@dataclass(frozen=True)
class Seq:
    parts: Tuple


@dataclass(frozen=True)
class Alt:
    branches: Tuple


@dataclass(frozen=True)
class Repeat:
    node: object
    min_count: int
    max_count: Optional[int]  # None = unbounded


@dataclass(frozen=True)
class EmptyStr:
    pass


@dataclass(frozen=True)
class AnchorStart:
    """Leading ^; a no-op under full-match, honored in search mode."""


@dataclass(frozen=True)
class AnchorEnd:
    """Trailing $; a no-op under full-match, honored in search mode."""


# ---------------------------------------------------------------------------
# range helpers
# ---------------------------------------------------------------------------


def normalize_ranges(ranges):
    """Sort and merge overlapping/adjacent inclusive ranges."""
    pts = sorted((int(lo), int(hi)) for lo, hi in ranges)
    out = []
    for lo, hi in pts:
        if lo > hi:
            raise RegexSyntaxError("bad character range (min > max)", -1)
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return tuple(out)


def complement_ranges(ranges, lo: int = 0, hi: int = MAX_CODEPOINT):
    ranges = normalize_ranges(ranges)
    out = []
    cur = lo
    for a, b in ranges:
        if a > cur:
            out.append((cur, a - 1))
        cur = max(cur, b + 1)
    if cur <= hi:
        out.append((cur, hi))
    return tuple(out)


def ranges_overlap(a, b) -> bool:
    """True if any range in normalized list a overlaps any range in b."""
    i = j = 0
    a = normalize_ranges(a)
    b = normalize_ranges(b)
    while i < len(a) and j < len(b):
        alo, ahi = a[i]
        blo, bhi = b[j]
        if alo <= bhi and blo <= ahi:
            return True
        if ahi < bhi:
            i += 1
        else:
            j += 1
    return False


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

_SIMPLE_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "f": "\f",
    "v": "\v",
    "a": "\a",
    "0": "\0",
}

_CLASS_ESCAPES = {
    "d": (DIGIT_RANGES, False),
    "D": (DIGIT_RANGES, True),
    "w": (WORD_RANGES, False),
    "W": (WORD_RANGES, True),
    "s": (SPACE_RANGES, False),
    "S": (SPACE_RANGES, True),
}

_UNSUPPORTED_GROUP_PREFIXES = {
    "?<=": "lookbehind (?<=...)",
    "?<!": "negative lookbehind (?<!...)",
    "?:": None,  # handled
    "?=": "lookahead (?=...)",
    "?!": "negative lookahead (?!...)",
    "?>": "atomic group (?>...)",
    "?#": "comment group (?#...)",
    "?P": "named group / named backreference (?P...)",
    "?(": "conditional group (?(id)yes|no)",
    "?i": "inline flags (?i)",
    "?m": "inline flags (?m)",
    "?s": "inline flags (?s)",
    "?x": "inline flags (?x)",
    "?a": "inline flags (?a)",
    "?u": "inline flags (?u)",
    "?L": "inline flags (?L)",
    "?-": "inline flags (?-...)",
}


def parse(pattern: str):
    """Parse *pattern* into an AST. Full-match semantics.

    A single leading ``^`` and trailing ``$`` are preserved as
    :class:`AnchorStart` / :class:`AnchorEnd` nodes: no-ops under the
    full-match semantics of :func:`~grammarguard.automata.compile_ast`, but
    honored by search-mode wrapping (JSON Schema `pattern` semantics).
    Group nesting is capped (MAX_GROUP_DEPTH); deeper patterns raise
    PatternTooLarge instead of RecursionError.
    """
    if not isinstance(pattern, str):
        raise RegexError("pattern must be a str")
    s = pattern
    anchored_start = s.startswith("^")
    if anchored_start:
        s = s[1:]
    anchored_end = s.endswith("$") and not s.endswith("\\$")
    if anchored_end:
        s = s[:-1]
    parser = _Parser(s)
    try:
        node = parser.parse_alt()
    except RecursionError:
        raise PatternTooLarge(
            "pattern nesting too deep to parse (recursion limit)"
        ) from None
    if parser.i != len(s):
        ch = s[parser.i]
        if ch == ")":
            raise RegexSyntaxError("unbalanced ')'", parser.i)
        raise RegexSyntaxError(f"unexpected character {ch!r}", parser.i)
    if anchored_start or anchored_end:
        # ^ and $ bind to their own alternation branch only for a BARE
        # top-level alternation: ^a|b is (^a)|(b), but ^(a|b) anchors the
        # whole group. Groups unwrap to bare Alt nodes, so branch-local
        # attachment requires the parser's top-level-bar provenance, not
        # the node shape.
        if isinstance(node, Alt) and parser.top_level_bar:
            branches = list(node.branches)
            if anchored_start:
                branches[0] = _with_anchor(branches[0], AnchorStart(), True)
            if anchored_end:
                branches[-1] = _with_anchor(branches[-1], AnchorEnd(), False)
            node = Alt(tuple(branches))
        else:
            parts = []
            if anchored_start:
                parts.append(AnchorStart())
            parts.append(node)
            if anchored_end:
                parts.append(AnchorEnd())
            node = Seq(tuple(parts)) if len(parts) > 1 else node
    return node


def _with_anchor(branch, anchor, front):
    if isinstance(branch, Seq):
        parts = (
            (anchor,) + branch.parts if front else branch.parts + (anchor,)
        )
        return Seq(parts)
    return Seq((anchor, branch) if front else (branch, anchor))


MAX_GROUP_DEPTH = 100


class _Parser:
    def __init__(self, s: str):
        self.s = s
        self.i = 0
        self.n = len(s)
        self.depth = 0
        self.top_level_bar = False

    def peek(self, ahead: int = 0):
        j = self.i + ahead
        return self.s[j] if j < self.n else ""

    # alternation -----------------------------------------------------------
    def parse_alt(self):
        branches = [self.parse_concat()]
        while self.peek() == "|":
            self.i += 1
            if self.depth == 0:
                self.top_level_bar = True
            branches.append(self.parse_concat())
        if len(branches) == 1:
            return branches[0]
        return Alt(tuple(branches))

    def parse_concat(self):
        parts = []
        while self.i < self.n and self.peek() not in ")|":
            parts.append(self.parse_repeat())
        if not parts:
            return EmptyStr()
        if len(parts) == 1:
            return parts[0]
        return Seq(tuple(parts))

    # quantifiers -----------------------------------------------------------
    def parse_repeat(self):
        atom = self.parse_atom()
        while True:
            ch = self.peek()
            if ch == "*":
                self.i += 1
                node = Repeat(atom, 0, None)
            elif ch == "+":
                self.i += 1
                node = Repeat(atom, 1, None)
            elif ch == "?":
                self.i += 1
                node = Repeat(atom, 0, 1)
            elif ch == "{":
                parsed = self._parse_braced_quantifier()
                if parsed is None:
                    return atom  # literal '{'
                lo, hi = parsed
                node = Repeat(atom, lo, hi)
            else:
                return atom
            # lazy / possessive suffix: language-identical, accept and ignore
            if self.peek() in ("?", "+"):
                self.i += 1
            atom = node
            # reject stacked quantifiers like Python re ("multiple repeat")
            nxt = self.peek()
            if nxt and nxt in "*+?":
                raise RegexSyntaxError("multiple repeat", self.i)
            if nxt == "{":
                save = self.i
                if self._parse_braced_quantifier() is not None:
                    raise RegexSyntaxError("multiple repeat", save)
                self.i = save

    def _parse_braced_quantifier(self):
        """Parse {n} {n,} {n,m} starting at '{'. Return None if malformed."""
        start = self.i
        j = start + 1
        while j < self.n and self.s[j].isdigit():
            j += 1
        if j == start + 1:
            return None
        lo = int(self.s[start + 1 : j])
        if j < self.n and self.s[j] == "}":
            self.i = j + 1
            return (lo, lo)
        if j < self.n and self.s[j] == ",":
            j += 1
            k = j
            while k < self.n and self.s[k].isdigit():
                k += 1
            if k < self.n and self.s[k] == "}":
                if k == j:
                    hi = None
                else:
                    hi = int(self.s[j:k])
                    if hi < lo:
                        raise RegexSyntaxError(
                            "quantifier min greater than max", start
                        )
                self.i = k + 1
                return (lo, hi)
        return None

    # atoms -----------------------------------------------------------------
    def parse_atom(self):
        ch = self.peek()
        if ch == "":
            raise RegexSyntaxError("unexpected end of pattern", self.i)
        if ch in "*+":
            raise RegexSyntaxError(f"nothing to repeat: '{ch}'", self.i)
        if ch == "(":
            return self._parse_group()
        if ch == "[":
            return self._parse_class()
        if ch == ".":
            self.i += 1
            return Dot()
        if ch == "\\":
            return self._parse_escape_atom()
        if ch in "^$":
            raise UnsupportedFeature(
                f"mid-pattern anchor '{ch}' (anchors are no-ops under "
                "full-match semantics and are only accepted at the edges)",
                self.i,
            )
        self.i += 1
        return Literal(ch)

    def _parse_group(self):
        start = self.i
        self.i += 1  # consume '('
        self.depth += 1
        if self.depth > MAX_GROUP_DEPTH:
            raise PatternTooLarge(
                f"group nesting deeper than {MAX_GROUP_DEPTH}"
            )
        if self.peek() == "?":
            if self.peek(1) == ":":
                self.i += 2
            else:
                three = self.s[self.i : self.i + 3]
                two = self.s[self.i : self.i + 2]
                for prefix in (three, two):
                    feature = _UNSUPPORTED_GROUP_PREFIXES.get(prefix)
                    if feature:
                        raise UnsupportedFeature(feature, start)
                raise UnsupportedFeature(
                    "group construct (?%s...)" % self.s[self.i + 1 : self.i + 3],
                    start,
                )
        node = self.parse_alt()
        if self.peek() != ")":
            raise RegexSyntaxError("unbalanced '('", start)
        self.i += 1
        self.depth -= 1
        return node

    def _parse_escape_atom(self):
        start = self.i
        self.i += 1
        ch = self.peek()
        if ch == "":
            raise RegexSyntaxError("trailing backslash", start)
        self.i += 1
        if ch in _SIMPLE_ESCAPES:
            lit = Literal(_SIMPLE_ESCAPES[ch])
            if ch == "0" and self.peek().isdigit():
                raise UnsupportedFeature("octal escape \\0NN", start)
            return lit
        if ch in _CLASS_ESCAPES:
            ranges, negated = _CLASS_ESCAPES[ch]
            return Class(tuple(ranges), negated)
        if ch == "x":
            return Literal(self._read_hex(2, start))
        if ch == "u":
            return Literal(self._read_hex(4, start))
        if ch == "U":
            return Literal(self._read_hex(8, start))
        if ch in "bBAZz":
            raise UnsupportedFeature(f"anchor/boundary escape \\{ch}", start)
        if ch.isdigit():
            raise UnsupportedFeature(f"backreference \\{ch}", start)
        if ch.isascii() and ch.isalpha():
            raise RegexSyntaxError(f"bad escape \\{ch}", start)
        return Literal(ch)

    def _read_hex(self, digits: int, start: int) -> str:
        raw = self.s[self.i : self.i + digits]
        if len(raw) != digits or any(
            c not in "0123456789abcdefABCDEF" for c in raw
        ):
            raise RegexSyntaxError("invalid hex escape", start)
        self.i += digits
        return chr(int(raw, 16))

    # character classes -------------------------------------------------------
    def _parse_class(self):
        start = self.i
        self.i += 1  # consume '['
        negated = False
        if self.peek() == "^":
            negated = True
            self.i += 1
        items = []  # list of single ranges (negated predefined pre-complemented)
        first = True
        while True:
            ch = self.peek()
            if ch == "":
                raise RegexSyntaxError("unterminated character class", start)
            if ch == "]" and not first:
                self.i += 1
                break
            first = False
            lo_ranges = self._class_item(start)
            lo_single_char = (
                len(lo_ranges) == 1 and lo_ranges[0][0] == lo_ranges[0][1]
            )
            if (
                not lo_single_char
                and self.peek() == "-"
                and self.peek(1) not in ("", "]")
            ):
                raise RegexSyntaxError(
                    "bad character range (character class as range "
                    "endpoint)",
                    start,
                )
            if (
                lo_single_char
                and self.peek() == "-"
                and self.peek(1) not in ("", "]")
            ):
                self.i += 1  # consume '-'
                hi_ranges = self._class_item(start)
                if len(hi_ranges) != 1 or hi_ranges[0][0] != hi_ranges[0][1]:
                    raise RegexSyntaxError("bad character range endpoint", self.i)
                lo_cp = lo_ranges[0][0]
                hi_cp = hi_ranges[0][0]
                if lo_cp > hi_cp:
                    raise RegexSyntaxError(
                        "bad character range (min > max)", start
                    )
                items.append((lo_cp, hi_cp))
            else:
                items.extend(lo_ranges)
        merged = normalize_ranges(tuple(items))
        return Class(merged, negated)

    def _class_item(self, class_start):
        """Return the list of ranges this single class item contributes."""
        ch = self.peek()
        if ch == "\\":
            start = self.i
            self.i += 1
            e = self.peek()
            if e == "":
                raise RegexSyntaxError("trailing backslash in class", start)
            self.i += 1
            if e in _SIMPLE_ESCAPES:
                if e == "0" and self.peek().isdigit():
                    raise UnsupportedFeature("octal escape \\0NN", start)
                return [(ord(_SIMPLE_ESCAPES[e]), ord(_SIMPLE_ESCAPES[e]))]
            if e in _CLASS_ESCAPES:
                ranges, negated = _CLASS_ESCAPES[e]
                if negated:
                    return list(complement_ranges(ranges))
                return list(ranges)
            if e == "x":
                c = self._read_hex(2, start)
                return [(ord(c), ord(c))]
            if e == "u":
                c = self._read_hex(4, start)
                return [(ord(c), ord(c))]
            if e == "U":
                c = self._read_hex(8, start)
                return [(ord(c), ord(c))]
            if e in "bBAZz":
                raise UnsupportedFeature(f"anchor/boundary escape \\{e}", start)
            if e.isdigit():
                raise UnsupportedFeature(f"backreference \\{e}", start)
            if e.isascii() and e.isalpha():
                raise RegexSyntaxError(f"bad escape \\{e}", start)
            return [(ord(e), ord(e))]
        self.i += 1
        return [(ord(ch), ord(ch))]
