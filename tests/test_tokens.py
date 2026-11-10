"""Dead-token analysis: brute-force verified on finite languages."""

from grammarguard import compile_ast, enumerate_accepted
from grammarguard.regex_ast import parse
from grammarguard.tokens import dead_tokens, load_vocab


def brute_force_dead(pattern, vocab, max_len):
    """Ground truth by full enumeration (finite languages only)."""
    dfa = compile_ast(parse(pattern))
    matches, truncated = enumerate_accepted(dfa, max_len=max_len, limit=10000)
    assert not truncated, "test setup: language too big to enumerate"
    dead = []
    for tok in vocab:
        if not any(tok in m for m in matches):
            dead.append(tok)
    return sorted(dead)


def check(pattern, vocab, max_len=6):
    dfa = compile_ast(parse(pattern))
    result = dead_tokens(dfa, vocab)
    assert not result["truncated"]
    assert sorted(result["dead"]) == brute_force_dead(pattern, vocab, max_len)
    assert result["dead_count"] + result["live_count"] == len(vocab)
    return result


def test_simple_vocab():
    res = check("a+", ["a", "aa", "b", "ab", "ba"], max_len=4)
    assert sorted(res["dead"]) == ["ab", "b", "ba"]
    assert res["live_count"] == 2


def test_finite_language():
    vocab = ["cat", "cats", "dog", "dogs", "at", "catsup", "x"]
    res = check("(?:cat|dog)s?", vocab, max_len=4)
    assert sorted(res["dead"]) == ["catsup", "x"]


def test_all_dead_when_unsatisfiable():
    dfa = compile_ast(parse("[^\\x00-\\U0010FFFF]"))
    res = dead_tokens(dfa, ["a", "b"])
    assert res["dead_count"] == 2 and res["live_count"] == 0


def test_budget_truncation():
    dfa = compile_ast(parse("a+"))
    vocab = [f"tok{i}" for i in range(100)]
    res = dead_tokens(dfa, vocab, max_checks=10)
    assert res["truncated"] is True
    assert res["checked"] < len(vocab)


def test_load_vocab(tmp_path):
    p = tmp_path / "vocab.txt"
    p.write_text("a\nb\na\n\nc\n", encoding="utf-8")
    assert load_vocab(str(p)) == ["a", "b", "c"]


def test_substring_semantics_documented_case():
    # "aa" is a substring of accepted "aaa" -> live even though a decoder
    # might never emit it; we only promise dead tokens are provably unusable.
    res = check("a{3}", ["aa"], max_len=3)
    assert res["live_count"] == 1
