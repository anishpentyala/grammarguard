"""Command-line interface for grammarguard."""

from __future__ import annotations

import argparse
import json
import sys

from .analyses import analyze, analyze_intersection
from .regex_ast import RegexError


def _show(value):
    return json.dumps(value, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="grammarguard",
        description=(
            "Offline preflight checks for regex constraints used in "
            "guided/structured generation (vLLM/SGLang structured outputs, "
            "outlines, JSON Schema `pattern` fields)."
        ),
    )
    p.add_argument("pattern", help="regex pattern to analyze")
    p.add_argument(
        "--mode",
        choices=["fullmatch", "search"],
        default="fullmatch",
        help="fullmatch = guided-decoding semantics (default); search = "
        "JSON Schema `pattern` substring semantics",
    )
    p.add_argument(
        "--intersect",
        metavar="OTHER",
        help="instead of a full report, check whether PATTERN and OTHER "
        "accept a common string",
    )
    p.add_argument("--witnesses", type=int, default=3, metavar="K")
    p.add_argument("--enum", type=int, default=16, metavar="N",
                   help="max accepted-string samples")
    p.add_argument("--enum-max-len", type=int, default=24, metavar="L",
                   help="max length of enumerated strings (cap 256)")
    p.add_argument("--near-miss", type=int, default=8, metavar="N",
                   help="max near-miss (rejected) control strings")
    p.add_argument("--vocab", metavar="PATH",
                   help="plain-text token vocabulary (one token per line) "
                   "for dead-token analysis")
    p.add_argument("--max-dfa-states", type=int, default=20000)
    p.add_argument("--json", action="store_true", help="JSON output")
    p.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when any hazard is found (unsatisfiable, high-risk "
        "ReDoS finding, dead tokens, size warning)",
    )
    return p


def _hazards(report: dict) -> int:
    n = 0
    if not report["satisfiable"]:
        n += 1
    n += sum(1 for f in report["redos_findings"] if f["risk"] == "high")
    n += len(report["size_warnings"])
    if report.get("dead_tokens", {}).get("dead_count"):
        n += 1
    return n


def _print_report(r: dict) -> None:
    print(f"pattern: {r['pattern']}   mode: {r['mode']}")
    print(f"satisfiable: {r['satisfiable']}")
    if r["satisfiable"]:
        print(f"shortest witnesses: {_show(r['witnesses'])}")
        print(
            f"sample matches (<= {len(r['sample_matches'])}, "
            f"truncated={r['enumeration_truncated']}): "
            f"{_show(r['sample_matches'])}"
        )
        print(f"near-miss controls (all rejected): {_show(r['near_misses'])}")
        print(f"common literal prefix: {_show(r['common_prefix'])}")
        print(
            f"length bounds: min={r['min_length']} "
            f"max={r['max_length'] if r['max_length'] is not None else 'unbounded'}"
            f"  finite={r['finite_language']}"
        )
    print(
        f"automaton: {r['dfa_states']} DFA states "
        f"({r['min_dfa_states']} minimized), {r['dfa_transitions']} transitions"
    )
    for w in r["size_warnings"]:
        print(f"size warning: {w}")
    if r["redos_findings"]:
        print("ReDoS hazard lint (heuristic, not a safety proof):")
        for f in r["redos_findings"]:
            print(f"  [{f['risk']}] {f['rule']}: {f['detail']}")
    else:
        print("ReDoS hazard lint: no known hazard shapes found "
              "(heuristic; not a safety proof)")
    if "dead_tokens" in r:
        dt = r["dead_tokens"]
        print(
            f"dead tokens: {dt['dead_count']} provably unusable, "
            f"{dt['live_count']} possibly usable, truncated={dt['truncated']}"
        )
        if dt["dead"]:
            print(f"  examples: {_show(dt['dead'][:10])}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.intersect is not None:
            result = analyze_intersection(
                args.pattern,
                args.intersect,
                mode=args.mode,
                max_dfa_states=args.max_dfa_states,
            )
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(f"pattern A: {result['pattern_a']}")
                print(f"pattern B: {result['pattern_b']}")
                print(f"intersection empty: {result['intersection_empty']}")
                if result["witness"] is not None:
                    print(f"shortest common witness: {_show(result['witness'])}")
            if args.strict and result["intersection_empty"]:
                return 1
            return 0
        report = analyze(
            args.pattern,
            mode=args.mode,
            witness_count=args.witnesses,
            enum_limit=args.enum,
            enum_max_len=args.enum_max_len,
            near_miss_limit=args.near_miss,
            vocab_path=args.vocab,
            max_dfa_states=args.max_dfa_states,
        )
    except RegexError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_report(report)
    if args.strict and _hazards(report) > 0:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
