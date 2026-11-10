"""CLI end-to-end tests: text output, JSON output, exit codes."""

import json

import pytest

from grammarguard.cli import main


def test_cli_text_basic(capsys):
    code = main(["\\d{4}-\\d{2}-\\d{2}"])
    assert code == 0
    out = capsys.readouterr().out
    assert "satisfiable: True" in out
    assert "min=10" in out and "max=10" in out
    assert "DFA states" in out


def test_cli_json_basic(capsys):
    code = main(["true|false", "--json"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["satisfiable"] is True
    assert report["finite_language"] is True
    assert sorted(report["witnesses"]) == ["false", "true"]


def test_cli_parse_error_exit_2(capsys):
    code = main(["a(?=b)"])
    assert code == 2
    err = capsys.readouterr().err
    assert "unsupported regex feature" in err


def test_cli_strict_exit_1_on_hazard(capsys):
    code = main(["(a+)+$", "--strict"])
    assert code == 1
    out = capsys.readouterr().out
    assert "nested-unbounded" in out


def test_cli_strict_clean_exit_0(capsys):
    code = main(["\\d{4}-\\d{2}-\\d{2}", "--strict"])
    assert code == 0


def test_cli_unsatisfiable_strict(capsys):
    code = main(["[^\\x00-\\U0010FFFF]", "--strict"])
    assert code == 1


def test_cli_intersect(capsys):
    code = main(["a+", "--intersect", "aa+"])
    assert code == 0
    out = capsys.readouterr().out
    assert "intersection empty: False" in out
    assert '"aa"' in out


def test_cli_intersect_empty_strict(capsys):
    code = main(["a+", "--intersect", "b+", "--strict"])
    assert code == 1
    out = capsys.readouterr().out
    assert "intersection empty: True" in out


def test_cli_vocab(tmp_path, capsys):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("a\naa\nb\n", encoding="utf-8")
    code = main(["a+", "--vocab", str(vocab), "--json"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dead_tokens"]["dead"] == ["b"]
    assert report["dead_tokens"]["live_count"] == 2


def test_cli_search_mode(capsys):
    code = main(["\\d{4}", "--mode", "search", "--json"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["max_length"] is None
