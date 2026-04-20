"""Grammar-based graph matching engine.

Two matching modes, selected automatically by quantifier:

- **Chain mode** (``"1"`` / ``"?"`` only): walks forward from a seed node
  along fan-out-1 consumer edges. Used by decomposition and optimization
  rules that match one node at a time.

- **Region mode** (any ``"*"`` / ``"+"``): forward BFS + backward cone.
  Absorbs all reachable nodes matching the grammar's productions, respecting
  ``bind`` constraints (e.g. all ReduceOps must share the same axis) and
  ``where`` predicates. Used by fusion rules.

Both modes produce the same ``ChainMatch`` result type.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from deplodock.compiler.ir.graph import Graph

# ---------------------------------------------------------------------------
# Grammar types
# ---------------------------------------------------------------------------


@dataclass
class Production:
    """Match nodes by op type, constraints, and optional predicates.

    ``quantifier`` controls how many nodes to consume:
      - ``"1"``: exactly one (fail if no match).
      - ``"?"``: zero or one.
      - ``"*"``: zero or more — uses region growing (all reachable).
      - ``"+"``: one or more — uses region growing (fail if none).

    ``constraints``: field-level equality checks on ``node.op``.
    ``where``: extra predicate ``(op, node, graph) -> bool``.
    ``bind``: unify op fields across all matches. E.g.
    ``{"axis": "reduce_axis"}`` extracts ``node.op.axis`` (normalized)
    and rejects if a prior match bound a different value. The special
    key ``"input_shape"`` reads the shape of the node's first input.
    """

    name: str
    op_type: type
    quantifier: str = "1"
    constraints: dict = field(default_factory=dict)
    where: object | None = None  # Callable[[Op, Node, Graph], bool] | None
    bind: dict = field(default_factory=dict)


@dataclass
class Group:
    """Repeating group of productions with backtracking (chain mode only)."""

    name: str
    productions: list[Production]
    quantifier: str = "*"


type ChainGrammar = list[Production | Group]


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------


@dataclass
class ChainSegment:
    """One named segment of matched nodes."""

    name: str
    node_ids: list[str] = field(default_factory=list)


@dataclass
class ChainMatch:
    """Result of matching a grammar against a graph."""

    root_node_id: str
    segments: list[ChainSegment] = field(default_factory=list)
    consumed: set[str] = field(default_factory=set)
    output: str | None = None
    bindings: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> list[str]:
        """All node ids for segments matching ``name`` (flattened)."""
        out: list[str] = []
        for seg in self.segments:
            if seg.name == name or seg.name.endswith(f".{name}"):
                out.extend(seg.node_ids)
        return out

    def get_groups(self, group_name: str) -> list[list[ChainSegment]]:
        """Return segment lists per group iteration."""
        buckets: dict[int, list[ChainSegment]] = {}
        for seg in self.segments:
            if not seg.name.startswith(f"{group_name}["):
                continue
            bracket_end = seg.name.index("]")
            idx = int(seg.name[len(group_name) + 1 : bracket_end])
            buckets.setdefault(idx, []).append(seg)
        return [buckets[i] for i in sorted(buckets)]


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------


def match_grammar(graph: Graph, grammar: ChainGrammar) -> list[ChainMatch]:
    """Find all non-overlapping grammar matches, scanning in topo order.

    Dispatches to region mode if any Production uses ``"*"`` or ``"+"``,
    otherwise uses the chain-walking mode.
    """
    use_region = any(isinstance(p, Production) and p.quantifier in ("*", "+") for p in grammar)
    if use_region:
        return _match_grammar_region(graph, grammar)
    return _match_grammar_chain(graph, grammar)


# ---------------------------------------------------------------------------
# Region mode: forward BFS + backward cone
# ---------------------------------------------------------------------------


def _match_grammar_region(graph: Graph, grammar: ChainGrammar) -> list[ChainMatch]:
    """Find non-overlapping region matches using BFS + backward cone."""
    productions = [p for p in grammar if isinstance(p, Production)]
    consumed: set[str] = set()
    results: list[ChainMatch] = []
    for nid in graph.topological_order():
        if nid in consumed:
            continue
        node = graph.nodes.get(nid)
        if node is None:
            continue
        if not _match_any_production(node, productions, {}, graph):
            continue
        match = _grow_region(graph, nid, productions, consumed)
        if match is not None:
            consumed.update(match.consumed)
            results.append(match)
    return results


def _grow_region(
    graph: Graph,
    seed: str,
    productions: list[Production],
    skip: set[str],
) -> ChainMatch | None:
    """Grow a region from seed using the grammar's productions."""
    topo = graph.topological_order()
    topo_idx = {nid: i for i, nid in enumerate(topo)}

    # Phase 1: forward BFS
    forward: set[str] = set()
    bindings: dict[str, Any] = {}
    seg_map: dict[str, list[str]] = {}  # production name → node ids
    last_bind_nid: str | None = None  # last node with bind constraints
    queue: deque[str] = deque([seed])

    while queue:
        nid = queue.popleft()
        if nid in forward or nid in skip:
            continue
        node = graph.nodes.get(nid)
        if node is None:
            continue
        prod = _match_any_production(node, productions, bindings, graph)
        if prod is None:
            continue

        # Try bind — extract values and unify
        new_bindings = dict(bindings)
        if prod.bind and not _try_bind(node, prod, new_bindings, graph):
            # Bind conflict — stop exploring from the parent that enqueued this
            continue

        bindings = new_bindings
        forward.add(nid)
        seg_map.setdefault(prod.name, []).append(nid)
        if prod.bind:
            last_bind_nid = nid

        # Don't cross geometry boundaries: if any consumer is a rejected
        # bind-constrained node, stop exploring from this node.
        if last_bind_nid is not None and _has_rejected_bind_consumer(graph, nid, productions, bindings):
            continue
        for cid in graph.consumers(nid):
            queue.append(cid)

    if not forward:
        return None

    # Phase 2: find output
    sorted_fwd = sorted(forward, key=lambda n: topo_idx[n])
    output: str | None = None
    for nid in reversed(sorted_fwd):
        if not any(c in forward for c in graph.consumers(nid)):
            output = nid
            break
    if output is None:
        return None

    # Phase 3: backward cone
    region: set[str] = set()
    for nid in reversed(sorted_fwd):
        if nid == output:
            region.add(nid)
            continue
        effective = _effective_consumers(graph, nid, forward, productions)
        if effective is not None and effective and all(c in region for c in effective):
            region.add(nid)

    if not region:
        return None

    # Build segments from region (topo order within each production name)
    segments: list[ChainSegment] = []
    for prod in productions:
        ids = [nid for nid in topo if nid in region and nid in seg_map.get(prod.name, [])]
        if ids:
            segments.append(ChainSegment(name=prod.name, node_ids=ids))

    return ChainMatch(
        root_node_id=seed,
        segments=segments,
        consumed=region,
        output=output,
        bindings=bindings,
    )


def _match_any_production(node, productions: list[Production], bindings: dict[str, Any], graph: Graph) -> Production | None:
    """Return the first Production matching this node, or None."""
    for prod in productions:
        if not isinstance(node.op, prod.op_type):
            continue
        if not _check_constraints(node, prod):
            continue
        if prod.where is not None and not prod.where(node.op, node, graph):
            continue
        return prod
    return None


def _check_constraints(node, prod: Production) -> bool:
    for field_name, expected in prod.constraints.items():
        actual = getattr(node.op, field_name, None)
        if actual is None or str(actual) != str(expected):
            return False
    return True


def _try_bind(node, prod: Production, bindings: dict[str, Any], graph: Graph) -> bool:
    """Extract bind values from node and unify with existing bindings."""
    for field_key, bind_name in prod.bind.items():
        if field_key == "input_shape":
            # Special: read shape of the node's first input
            if not node.inputs:
                return False
            inp_node = graph.nodes.get(node.inputs[0])
            if inp_node is None:
                return False
            value = tuple(inp_node.output.shape)
        else:
            raw = getattr(node.op, field_key, None)
            if raw is None:
                return False
            # Normalize axis using input shape
            if field_key == "axis" and isinstance(raw, int) and node.inputs:
                inp_node = graph.nodes.get(node.inputs[0])
                if inp_node is not None:
                    ndim = len(inp_node.output.shape)
                    raw = raw % ndim if ndim else raw
            value = raw

        if bind_name in bindings:
            if bindings[bind_name] != value:
                return False
        else:
            bindings[bind_name] = value
    return True


def _has_rejected_bind_consumer(graph: Graph, nid: str, productions: list[Production], bindings: dict[str, Any]) -> bool:
    """Check if any consumer is a bind-constrained node that would be rejected."""
    for cid in graph.consumers(nid):
        node = graph.nodes.get(cid)
        if node is None:
            continue
        for prod in productions:
            if not prod.bind:
                continue
            if not isinstance(node.op, prod.op_type):
                continue
            if not _check_constraints(node, prod):
                continue
            # This consumer matches a bind-constrained production — would it pass?
            test = dict(bindings)
            if not _try_bind(node, prod, test, graph):
                return True  # rejected bind
    return False


def _effective_consumers(graph: Graph, nid: str, forward: set[str], productions: list[Production]) -> list[str] | None:
    """Get consumers in forward set, looking through non-bind production nodes.

    Returns None if any consumer path leads to a dead end (signals a
    side output that prevents absorption).
    """
    result: list[str] = []
    stack = list(graph.consumers(nid))
    visited: set[str] = set()
    while stack:
        cid = stack.pop()
        if cid in visited:
            continue
        visited.add(cid)
        if cid in forward:
            result.append(cid)
        else:
            # Look through nodes that match a non-bind production
            cnode = graph.nodes.get(cid)
            if cnode is not None and _is_lookthrough(cnode, productions, graph):
                stack.extend(graph.consumers(cid))
            else:
                return None  # dead end
    return result


def _is_lookthrough(node, productions: list[Production], graph: Graph) -> bool:
    """A node is look-through if it matches a production without bind constraints."""
    for prod in productions:
        if prod.bind:
            continue
        if not isinstance(node.op, prod.op_type):
            continue
        if not _check_constraints(node, prod):
            continue
        if prod.where is not None and not prod.where(node.op, node, graph):
            continue
        return True
    return False


# ---------------------------------------------------------------------------
# Chain mode: original sole-consumer walking (for "1"/"?" grammars)
# ---------------------------------------------------------------------------


def _match_grammar_chain(graph: Graph, grammar: ChainGrammar) -> list[ChainMatch]:
    """Return every chain match rooted at a node in topo order.

    Matches are allowed to overlap — e.g. both ``{A, B}`` and ``{B, C}``
    for a two-production ``producer → consumer`` grammar. The rewriter
    applies at most one match per iteration (breaks after the first
    successful ``rewrite``), so overlap is only a candidate enumeration
    concern, not a correctness one. Returning overlapping matches lets
    rule 5 (``LoopOp → LoopOp``) try ``{B, C}`` after ``{A, B}`` fails
    to splice, instead of skipping ``B`` for the rest of the pass.
    """
    results: list[ChainMatch] = []
    for nid in graph.topological_order():
        match = parse_chain(graph, nid, grammar)
        if match is not None:
            results.append(match)
    return results


def parse_chain(
    graph: Graph,
    start_nid: str,
    grammar: ChainGrammar,
    already_consumed: set[str] | None = None,
) -> ChainMatch | None:
    """Walk forward from ``start_nid`` along fan-out-1 edges, matching
    nodes against ``grammar`` productions (chain mode).
    """
    skip = already_consumed or set()
    if start_nid not in graph.nodes or start_nid in skip:
        return None

    ctx = _ParseCtx(graph=graph, skip=skip)
    ctx.cursor = start_nid

    for item in grammar:
        if isinstance(item, Production):
            ok = _parse_production(ctx, item, prefix="")
            if not ok:
                return None
        elif isinstance(item, Group):
            ok = _parse_group(ctx, item)
            if not ok:
                return None

    if not ctx.segments:
        return None

    return ChainMatch(
        root_node_id=start_nid,
        segments=ctx.segments,
        consumed=ctx.consumed,
    )


# ---------------------------------------------------------------------------
# Chain mode internals
# ---------------------------------------------------------------------------


@dataclass
class _ParseCtx:
    graph: Graph
    skip: set[str]
    cursor: str | None = None
    segments: list[ChainSegment] = field(default_factory=list)
    consumed: set[str] = field(default_factory=set)

    def snapshot(self) -> tuple[str | None, int, int]:
        return self.cursor, len(self.segments), len(self.consumed)

    def restore(self, snap: tuple[str | None, int, int]) -> None:
        cursor, seg_len, consumed_len = snap
        self.cursor = cursor
        removed_segs = self.segments[seg_len:]
        self.segments = self.segments[:seg_len]
        for seg in removed_segs:
            for nid in seg.node_ids:
                self.consumed.discard(nid)


def _parse_production(ctx: _ParseCtx, prod: Production, prefix: str) -> bool:
    seg_name = f"{prefix}{prod.name}" if prefix else prod.name
    count = 0
    min_count = 1 if prod.quantifier in ("1", "+") else 0
    max_count = 1 if prod.quantifier in ("1", "?") else float("inf")

    seg = ChainSegment(name=seg_name)

    while count < max_count:
        if ctx.cursor is None:
            break
        if not _node_matches(ctx, ctx.cursor, prod):
            break
        seg.node_ids.append(ctx.cursor)
        ctx.consumed.add(ctx.cursor)
        count += 1
        ctx.cursor = _sole_consumer(ctx.graph, ctx.cursor, ctx.skip | ctx.consumed)

    if count < min_count:
        for nid in seg.node_ids:
            ctx.consumed.discard(nid)
        return False

    if seg.node_ids:
        ctx.segments.append(seg)
    return True


def _parse_group(ctx: _ParseCtx, group: Group) -> bool:
    min_iters = 1 if group.quantifier == "+" else 0
    iteration_count = 0

    while True:
        snap = ctx.snapshot()
        prefix = f"{group.name}[{iteration_count}]."
        ok = True
        for sub in group.productions:
            if not _parse_production(ctx, sub, prefix=prefix):
                ok = False
                break
        if ok:
            iteration_count += 1
            continue
        ctx.restore(snap)
        break

    return iteration_count >= min_iters


def _node_matches(ctx: _ParseCtx, nid: str, prod: Production) -> bool:
    node = ctx.graph.nodes.get(nid)
    if node is None or nid in ctx.skip:
        return False
    if not isinstance(node.op, prod.op_type):
        return False
    return _check_constraints(node, prod)


def _sole_consumer(graph: Graph, nid: str, exclude: set[str]) -> str | None:
    consumers = [cid for cid in graph.consumers(nid) if cid not in exclude]
    if len(consumers) == 1:
        return consumers[0]
    return None
