"""NFA/DFA construction and language analyses for grammarguard.

Pipeline: AST -> (bounded expansion of counted repeats) -> Thompson NFA ->
symbolic DFA (interval alphabet) -> optional minimization.

All potentially expensive operations are bounded:

  * counted-repeat expansion is capped (``MAX_REPEAT_EXPANSION`` per repeat,
    ``MAX_EXPANDED_NODES`` total AST nodes)
  * determinization is capped at ``MAX_DFA_STATES``
  * enumeration and witness searches are depth/count capped

Analyses provided: simulation, shortest witnesses, bounded enumeration,
single-edit near misses, common literal prefix, length bounds + finiteness,
productive-state trimming, and lazy product emptiness/witness for
intersection of two DFAs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .regex_ast import (
    DOT_RANGES,
    Alt,
    AnchorEnd,
    AnchorStart,
    Class,
    Dot,
    EmptyStr,
    Literal,
    PatternTooLarge,
    Repeat,
    Seq,
)

_ANCHOR_OR_EMPTY = (EmptyStr, AnchorStart, AnchorEnd)

MAX_REPEAT_EXPANSION = 512      # max bound allowed on one counted repeat
MAX_EXPANDED_NODES = 20000      # total AST nodes after expansion
MAX_DFA_STATES = 20000          # determinization cap
MAX_WITNESS_DEPTH_FACTOR = 2    # BFS depth cap = factor * dfa states


# ---------------------------------------------------------------------------
# counted-repeat expansion
# ---------------------------------------------------------------------------


def _count_nodes(node) -> int:
    if isinstance(node, (Literal, Dot, Class) + _ANCHOR_OR_EMPTY):
        return 1
    if isinstance(node, Seq):
        return 1 + sum(_count_nodes(p) for p in node.parts)
    if isinstance(node, Alt):
        return 1 + sum(_count_nodes(b) for b in node.branches)
    if isinstance(node, Repeat):
        return 1 + _count_nodes(node.node)
    raise TypeError(f"unknown AST node {node!r}")


def expand_repeats(node):
    """Rewrite Repeat(min, max) with finite max into Seq/Alt structure.

    Only star (Repeat(min=0, max=None)) survives expansion. Bounded repeats
    above MAX_REPEAT_EXPANSION or expansions beyond MAX_EXPANDED_NODES raise
    PatternTooLarge. Size is tracked bottom-up and checked BEFORE building
    each expanded subtree, so deeply nested repeats abort in time
    proportional to the unexpanded pattern, not the expansion.
    """
    expanded, _ = _expand(node)
    return expanded


def _expand(node):
    """Return (expanded_node, node_count), aborting early over the cap."""
    if isinstance(node, (Literal, Dot, Class) + _ANCHOR_OR_EMPTY):
        return node, 1
    if isinstance(node, Seq):
        total = 1
        parts = []
        for p in node.parts:
            e, c = _expand(p)
            parts.append(e)
            total += c
            if total > MAX_EXPANDED_NODES:
                raise PatternTooLarge(
                    f"expanded pattern exceeds {MAX_EXPANDED_NODES} nodes"
                )
        return Seq(tuple(parts)), total
    if isinstance(node, Alt):
        total = 1
        branches = []
        for b in node.branches:
            e, c = _expand(b)
            branches.append(e)
            total += c
            if total > MAX_EXPANDED_NODES:
                raise PatternTooLarge(
                    f"expanded pattern exceeds {MAX_EXPANDED_NODES} nodes"
                )
        return Alt(tuple(branches)), total
    if isinstance(node, Repeat):
        inner, inner_count = _expand(node.node)
        lo, hi = node.min_count, node.max_count
        if hi is None:
            if lo > MAX_REPEAT_EXPANSION:
                raise PatternTooLarge(
                    f"repeat lower bound {lo} exceeds limit "
                    f"{MAX_REPEAT_EXPANSION}"
                )
            factor = lo + 1
        else:
            if hi > MAX_REPEAT_EXPANSION:
                raise PatternTooLarge(
                    f"repeat upper bound {hi} exceeds limit "
                    f"{MAX_REPEAT_EXPANSION}"
                )
            factor = hi
        total = 1 + factor * inner_count
        if total > MAX_EXPANDED_NODES:
            raise PatternTooLarge(
                f"expanded pattern exceeds {MAX_EXPANDED_NODES} nodes"
            )
        parts = [inner] * lo
        if hi is None:
            parts.append(Repeat(inner, 0, None))
        else:
            for _ in range(hi - lo):
                parts.append(Alt((EmptyStr(), inner)))
        return _seq_of(parts), total
    raise TypeError(f"unknown AST node {node!r}")


def _seq_of(parts):
    if not parts:
        return EmptyStr()
    if len(parts) == 1:
        return parts[0]
    return Seq(tuple(parts))





# ---------------------------------------------------------------------------
# NFA
# ---------------------------------------------------------------------------


class NFA:
    def __init__(self):
        self.sym: Dict[int, List[Tuple[int, int, int]]] = {}
        self.eps: Dict[int, List[int]] = {}
        self.n_states = 0
        self.start = self._new()
        self.accept = self._new()

    def _new(self) -> int:
        s = self.n_states
        self.n_states += 1
        self.sym.setdefault(s, [])
        self.eps.setdefault(s, [])
        return s

    def add(self, frm: int, lo: int, hi: int, to: int):
        self.sym[frm].append((lo, hi, to))


def build_nfa(node) -> NFA:
    nfa = NFA()
    s, a = _build(nfa, node)
    nfa.eps[nfa.start].append(s)
    nfa.eps[a].append(nfa.accept)
    return nfa


def _build(nfa: NFA, node) -> Tuple[int, int]:
    if isinstance(node, _ANCHOR_OR_EMPTY):
        s, a = nfa._new(), nfa._new()
        nfa.eps[s].append(a)
        return s, a
    if isinstance(node, Literal):
        s, a = nfa._new(), nfa._new()
        nfa.add(s, ord(node.char), ord(node.char), a)
        return s, a
    if isinstance(node, Dot):
        s, a = nfa._new(), nfa._new()
        for lo, hi in DOT_RANGES:
            nfa.add(s, lo, hi, a)
        return s, a
    if isinstance(node, Class):
        s, a = nfa._new(), nfa._new()
        for lo, hi in node.resolved():
            nfa.add(s, lo, hi, a)
        return s, a
    if isinstance(node, Seq):
        parts = [_build(nfa, p) for p in node.parts]
        for (s1, a1), (s2, a2) in zip(parts, parts[1:]):
            nfa.eps[a1].append(s2)
        return parts[0][0], parts[-1][1]
    if isinstance(node, Alt):
        s, a = nfa._new(), nfa._new()
        for b in node.branches:
            bs, ba = _build(nfa, b)
            nfa.eps[s].append(bs)
            nfa.eps[ba].append(a)
        return s, a
    if isinstance(node, Repeat):
        if node.min_count != 0 or node.max_count is not None:
            raise AssertionError("expand_repeats must run first")
        s, a = nfa._new(), nfa._new()
        bs, ba = _build(nfa, node.node)
        nfa.eps[s].append(a)       # zero times
        nfa.eps[s].append(bs)
        nfa.eps[ba].append(bs)     # loop
        nfa.eps[ba].append(a)
        return s, a
    raise TypeError(f"unknown AST node {node!r}")


def _closure(nfa: NFA, states) -> frozenset:
    stack = list(states)
    seen = set(states)
    while stack:
        s = stack.pop()
        for t in nfa.eps.get(s, ()):
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return frozenset(seen)


def _atomic_intervals(nfa: NFA) -> List[Tuple[int, int]]:
    cuts = set()
    for trans in nfa.sym.values():
        for lo, hi, _ in trans:
            cuts.add(lo)
            cuts.add(hi + 1)
    pts = sorted(cuts)
    return [(pts[i], pts[i + 1] - 1) for i in range(len(pts) - 1)]


# ---------------------------------------------------------------------------
# DFA
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DFA:
    n_states: int
    start: int
    accepts: frozenset
    # trans[state] = tuple of (lo, hi, target), sorted by lo, non-overlapping
    trans: tuple

    def step(self, state: int, codepoint: int) -> Optional[int]:
        lo, hi = 0, len(self.trans[state]) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            t_lo, t_hi, target = self.trans[state][mid]
            if codepoint < t_lo:
                hi = mid - 1
            elif codepoint > t_hi:
                lo = mid + 1
            else:
                return target
        return None

    def n_transitions(self) -> int:
        return sum(len(t) for t in self.trans)


def determinize(nfa: NFA, max_states: int = MAX_DFA_STATES) -> DFA:
    intervals = _atomic_intervals(nfa)
    # per NFA state, per interval -> target set
    start_set = _closure(nfa, {nfa.start})
    ids = {start_set: 0}
    trans_out: List[List[Tuple[int, int, int]]] = [[]]
    accepts = []
    queue = [start_set]
    while queue:
        current = queue.pop()
        cid = ids[current]
        accepts.append(cid if nfa.accept in current else -1)
        # interval index -> target frozenset
        iv_targets: List[frozenset] = []
        for lo, hi in intervals:
            targets = set()
            for s in current:
                for r_lo, r_hi, t in nfa.sym.get(s, ()):
                    if r_lo <= lo and hi <= r_hi:
                        targets.add(t)
            iv_targets.append(_closure(nfa, targets) if targets else frozenset())
        # merge adjacent intervals with equal targets
        merged: List[Tuple[int, int, Optional[frozenset]]] = []
        for (lo, hi), tgt in zip(intervals, iv_targets):
            if merged and merged[-1][2] == tgt and merged[-1][1] + 1 == lo:
                merged[-1] = (merged[-1][0], hi, tgt)
            else:
                merged.append((lo, hi, tgt))
        for lo, hi, tgt in merged:
            if not tgt:
                continue
            if tgt not in ids:
                if len(ids) >= max_states:
                    raise PatternTooLarge(
                        f"DFA exceeds {max_states} states; refusing to build "
                        "further (pattern automaton too large)"
                    )
                ids[tgt] = len(ids)
                trans_out.append([])
                queue.append(tgt)
            trans_out[cid].append((lo, hi, ids[tgt]))
    accept_set = frozenset(x for x in accepts if x >= 0)
    return DFA(
        n_states=len(ids),
        start=0,
        accepts=accept_set,
        trans=tuple(tuple(sorted(t)) for t in trans_out),
    )


def compile_ast(node, max_states: int = MAX_DFA_STATES) -> DFA:
    try:
        expanded = expand_repeats(node)
        nfa = build_nfa(expanded)
        return determinize(nfa, max_states=max_states)
    except RecursionError:
        raise PatternTooLarge(
            "pattern nesting too deep to analyze (recursion limit)"
        ) from None


def _partitions_equal(a: List[int], b: List[int]) -> bool:
    """True if block-id lists a and b induce the same set partition."""
    map_ab: Dict[int, int] = {}
    map_ba: Dict[int, int] = {}
    for x, y in zip(a, b):
        if map_ab.setdefault(x, y) != y or map_ba.setdefault(y, x) != x:
            return False
    return True


def minimize(dfa: DFA) -> DFA:
    """Partition-refinement DFA minimization."""
    # atomic interval boundaries from transitions
    cuts = set()
    for trans in dfa.trans:
        for lo, hi, _ in trans:
            cuts.add(lo)
            cuts.add(hi + 1)
    pts = sorted(cuts)
    intervals = [(pts[i], pts[i + 1] - 1) for i in range(len(pts) - 1)]

    def interval_of(cp: int) -> int:
        lo_i, hi_i = 0, len(intervals) - 1
        while lo_i <= hi_i:
            mid = (lo_i + hi_i) // 2
            a, b = intervals[mid]
            if cp < a:
                hi_i = mid - 1
            elif cp > b:
                lo_i = mid + 1
            else:
                return mid
        return -1

    blocks: List[int] = [1 if s in dfa.accepts else 0 for s in range(dfa.n_states)]
    while True:
        # map (block, per-interval target block) -> new block
        sig_map: Dict[tuple, int] = {}
        new_blocks = []
        for s in range(dfa.n_states):
            row = [-1] * len(intervals)
            for lo, hi, tgt in dfa.trans[s]:
                i = interval_of(lo)
                if i >= 0:
                    row[i] = blocks[tgt]
            sig = (blocks[s], tuple(row))
            new_blocks.append(sig_map.setdefault(sig, len(sig_map)))
        if _partitions_equal(blocks, new_blocks):
            break
        blocks = new_blocks
    # rebuild
    order = sorted(set(blocks), key=lambda b: (b != blocks[dfa.start], b))
    remap = {b: i for i, b in enumerate(order)}
    new_trans: List[List[Tuple[int, int, int]]] = [[] for _ in order]
    for s in range(dfa.n_states):
        for lo, hi, tgt in dfa.trans[s]:
            new_trans[remap[blocks[s]]].append((lo, hi, remap[blocks[tgt]]))
    new_trans = [sorted(set(t)) for t in new_trans]
    return DFA(
        n_states=len(order),
        start=remap[blocks[dfa.start]],
        accepts=frozenset(
            remap[blocks[s]] for s in dfa.accepts
        ),
        trans=tuple(tuple(t) for t in new_trans),
    )


def productive_states(dfa: DFA) -> frozenset:
    """States that can reach an accept state."""
    rev: Dict[int, List[int]] = {s: [] for s in range(dfa.n_states)}
    for s in range(dfa.n_states):
        for _, _, tgt in dfa.trans[s]:
            rev[tgt].append(s)
    seen = set(dfa.accepts)
    stack = list(dfa.accepts)
    while stack:
        s = stack.pop()
        for t in rev[s]:
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return frozenset(seen)


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------


def simulate(dfa: DFA, text: str) -> bool:
    state = dfa.start
    for ch in text:
        state = dfa.step(state, ord(ch))
        if state is None:
            return False
    return state in dfa.accepts


def is_satisfiable(dfa: DFA) -> bool:
    return dfa.start in productive_states(dfa)


def shortest_witnesses(dfa: DFA, k: int = 3) -> List[str]:
    """Up to k shortest accepted strings (ascending length)."""
    out: List[str] = []
    if not is_satisfiable(dfa):
        return out
    visited = {dfa.start}
    frontier = [(dfa.start, "")]
    max_depth = MAX_WITNESS_DEPTH_FACTOR * dfa.n_states + 1
    depth = 0
    while frontier and len(out) < k and depth <= max_depth:
        depth += 1
        nxt = []
        for state, prefix in frontier:
            if state in dfa.accepts:
                out.append(prefix)
                if len(out) >= k:
                    break
        if len(out) >= k:
            break
        for state, prefix in frontier:
            for lo, hi, tgt in dfa.trans[state]:
                if tgt in visited:
                    continue
                visited.add(tgt)
                nxt.append((tgt, prefix + chr(lo)))
        frontier = nxt
    return out


def enumerate_accepted(
    dfa: DFA, max_len: int, limit: int = 64
) -> Tuple[List[str], bool]:
    """Sample accepted strings of length <= *max_len*, at most *limit*.

    Breadth-first by string length with per-state prefix buckets capped at
    *limit* and a per-level work budget, so wide ranges (e.g. dotall loops
    from search-mode patterns) cannot explode. Returns (strings,
    truncated); truncated=True means accepted strings within max_len exist
    that were not returned. max_len is capped at 256.
    """
    if max_len > 256:
        raise ValueError("max_len capped at 256")
    out: List[str] = []
    truncated = False
    level_budget = max(limit * 40, 20000)
    prefixes: Dict[int, List[str]] = {dfa.start: [""]}
    for depth in range(max_len + 1):
        # collect accepted strings at this level under one global limit
        for state in sorted(prefixes):
            if state not in dfa.accepts:
                continue
            for p in prefixes[state]:
                if len(out) >= limit:
                    truncated = True  # uncollected accepted strings remain
                    break
                out.append(p)
        if depth == max_len or not prefixes:
            break
        nxt: Dict[int, List[str]] = {}
        used = 0
        for state in sorted(prefixes):
            for p in prefixes[state]:
                trans = dfa.trans[state]
                # pass 1: one character per transition, so every reachable
                # target is represented even when the level budget or a
                # bucket fills (no starvation of later transitions)
                for lo, hi, tgt in trans:
                    bucket = nxt.setdefault(tgt, [])
                    if len(bucket) < limit and used < level_budget:
                        bucket.append(p + chr(lo))
                        used += 1
                    else:
                        truncated = True
                # pass 2: spend remaining budget round-robin
                offsets = [1] * len(trans)
                while used < level_budget:
                    progressed = False
                    for k, (lo, hi, tgt) in enumerate(trans):
                        cp = lo + offsets[k]
                        if cp > hi:
                            continue
                        bucket = nxt[tgt]
                        if len(bucket) >= limit:
                            truncated = True
                            continue
                        bucket.append(p + chr(cp))
                        offsets[k] += 1
                        used += 1
                        progressed = True
                        if used >= level_budget:
                            break
                    if not progressed or used >= level_budget:
                        if used >= level_budget and any(
                            lo + offsets[k] <= hi
                            for k, (lo, hi, _) in enumerate(trans)
                        ):
                            truncated = True
                        break
        prefixes = nxt
    return out, truncated


_PROBE_CHARS = ("a", "b", "0", "1", " ", "-", "_", "A", ".", "/", "9", ":")


def near_misses(dfa: DFA, witnesses, limit: int = 8) -> List[str]:
    """Single-edit variants of witnesses that are NOT accepted (controls)."""
    out: List[str] = []
    seen = set()
    for w in witnesses:
        if len(out) >= limit:
            break
        candidates = []
        for i in range(len(w)):
            candidates.append(w[:i] + w[i + 1 :])  # deletion
            for c in _PROBE_CHARS:
                candidates.append(w[:i] + c + w[i + 1 :])  # substitution
        for c in _PROBE_CHARS:
            candidates.append(c + w)
            candidates.append(w + c)
        for cand in candidates:
            if len(out) >= limit:
                break
            if cand in seen or cand == w:
                continue
            seen.add(cand)
            if not simulate(dfa, cand):
                out.append(cand)
    return out


def common_prefix(dfa: DFA, max_len: int = 256) -> str:
    """Longest literal prefix shared by every accepted string."""
    chars = []
    state = dfa.start
    while len(chars) < max_len:
        if state in dfa.accepts:
            break  # empty suffix accepted -> divergence point
        trans = dfa.trans[state]
        if len(trans) != 1:
            break
        lo, hi, tgt = trans[0]
        if lo != hi:
            break
        chars.append(chr(lo))
        state = tgt
    return "".join(chars)


def length_bounds(dfa: DFA) -> Tuple[Optional[int], Optional[int], bool]:
    """Return (min_length, max_length, finite).

    min_length is None if the language is empty; max_length is None if
    unbounded. finite=True iff the language has finitely many strings.
    Iterative throughout (no recursion-depth limits).
    """
    if not is_satisfiable(dfa):
        return None, None, True
    prod = productive_states(dfa)
    # BFS for min
    dist = {dfa.start: 0}
    queue = [dfa.start]
    min_len: Optional[int] = None
    while queue and min_len is None:
        nxt = []
        for s in queue:
            if s in dfa.accepts:
                min_len = dist[s]
                break
            for _, _, tgt in dfa.trans[s]:
                if tgt in prod and tgt not in dist:
                    dist[tgt] = dist[s] + 1
                    nxt.append(tgt)
        queue = nxt
    # iterative cycle detection within productive subgraph -> unbounded
    color: Dict[int, int] = {}  # 1 = in stack, 2 = done
    cyclic = False
    for root in prod:
        if cyclic or color.get(root) == 2:
            continue
        stack = [(root, iter(dfa.trans[root]))]
        color[root] = 1
        while stack and not cyclic:
            s, it = stack[-1]
            advanced = False
            for _, _, tgt in it:
                if tgt not in prod:
                    continue
                c = color.get(tgt, 0)
                if c == 1:
                    cyclic = True
                    break
                if c == 0:
                    color[tgt] = 1
                    stack.append((tgt, iter(dfa.trans[tgt])))
                    advanced = True
                    break
            if not advanced:
                color[s] = 2
                stack.pop()
    if cyclic:
        return min_len, None, False
    # DAG: longest path from start via topological order (Kahn) on the
    # productive subgraph. Sinks of the productive subgraph are accept
    # states, so the maximum distance is the longest accepted string.
    indeg: Dict[int, int] = {s: 0 for s in prod}
    for s in prod:
        for _, _, tgt in dfa.trans[s]:
            if tgt in prod:
                indeg[tgt] += 1
    dist_from_start: Dict[int, int] = {dfa.start: 0}
    queue = [s for s in prod if indeg[s] == 0]
    while queue:
        s = queue.pop()
        if s not in dist_from_start:
            continue
        for _, _, tgt in dfa.trans[s]:
            if tgt not in prod:
                continue
            if dist_from_start[s] + 1 > dist_from_start.get(tgt, -1):
                dist_from_start[tgt] = dist_from_start[s] + 1
            indeg[tgt] -= 1
            if indeg[tgt] == 0:
                queue.append(tgt)
    return min_len, max(dist_from_start.values()), True


def intersect_witness(dfa1: DFA, dfa2: DFA) -> Optional[str]:
    """Shortest string accepted by BOTH dfas, or None if intersection empty.

    Lazy product construction with BFS, so no full product DFA is built.
    """
    start = (dfa1.start, dfa2.start)
    visited = {start}
    frontier = [(start, "")]
    max_depth = MAX_WITNESS_DEPTH_FACTOR * max(
        dfa1.n_states, dfa2.n_states
    ) ** 2 + 1
    depth = 0
    while frontier and depth <= max_depth:
        depth += 1
        nxt = []
        for (s1, s2), prefix in frontier:
            if s1 in dfa1.accepts and s2 in dfa2.accepts:
                return prefix
        for (s1, s2), prefix in frontier:
            # pair transitions on shared codepoints via interval walking
            for lo1, hi1, t1 in dfa1.trans[s1]:
                for lo2, hi2, t2 in dfa2.trans[s2]:
                    lo = max(lo1, lo2)
                    hi = min(hi1, hi2)
                    if lo > hi:
                        continue
                    pair = (t1, t2)
                    if pair in visited:
                        continue
                    visited.add(pair)
                    nxt.append((pair, prefix + chr(lo)))
        frontier = nxt
    return None
