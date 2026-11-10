"""grammarguard - offline preflight for regex constraints used in
guided/structured generation.

Check a regex *before* handing it to a constrained-decoding engine
(vLLM/SGLang structured outputs, outlines, JSON Schema `pattern` fields):
is it satisfiable, what do matches look like, how large is the compiled
automaton, does it carry known catastrophic-backtracking shapes, and which
vocabulary tokens can never be emitted.

Supported regex subset and all analysis limits are documented in
``grammarguard.regex_ast`` and the README. Unsupported constructs raise
:class:`grammarguard.regex_ast.UnsupportedFeature` loudly instead of being
silently mis-analyzed.
"""

from .analyses import analyze, analyze_intersection
from .automata import (
    compile_ast,
    common_prefix,
    enumerate_accepted,
    intersect_witness,
    is_satisfiable,
    length_bounds,
    minimize,
    near_misses,
    shortest_witnesses,
    simulate,
)
from .redos import lint as redos_lint
from .regex_ast import (
    PatternTooLarge,
    RegexError,
    RegexSyntaxError,
    UnsupportedFeature,
    parse,
)
from .tokens import dead_tokens, load_vocab

__version__ = "0.1.0"

__all__ = [
    "analyze",
    "analyze_intersection",
    "compile_ast",
    "common_prefix",
    "dead_tokens",
    "enumerate_accepted",
    "intersect_witness",
    "is_satisfiable",
    "length_bounds",
    "load_vocab",
    "minimize",
    "near_misses",
    "parse",
    "redos_lint",
    "shortest_witnesses",
    "simulate",
    "PatternTooLarge",
    "RegexError",
    "RegexSyntaxError",
    "UnsupportedFeature",
    "__version__",
]
