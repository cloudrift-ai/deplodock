"""Chain-grammar graph matching engine.

Replaces the tree-pattern matcher with a forward-walking chain parser.
Every rule — decomposition, optimization, fusion — declares a ``GRAMMAR``
(list of ``Production`` / ``Group``) instead of a ``PATTERN`` string.

The parser walks forward from a seed node along fan-out-1 consumer edges,
matching each node against the grammar's current production. Groups
enable repeating sub-sequences with all-or-nothing backtracking.

Conceptual framing: the grammar is a **regular expression over the
dataflow graph**, where the "string" is the fan-out-1 path from a seed
node and each "character" is a graph node matched by its op type +
optional field constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir import Graph

# ---------------------------------------------------------------------------
# Grammar types
# ---------------------------------------------------------------------------


@dataclass
class Production:
    """Match nodes of a given op type along a fan-out-1 chain.

    ``quantifier`` controls how many consecutive nodes to consume:
      - ``"1"``: exactly one (fail if no match).
      - ``"?"``: zero or one.
      - ``"*"``: zero or more (greedy).
      - ``"+"``: one or more (greedy; fail if no match on first).

    ``constraints``: field-level checks on the matched node's ``op``.
    E.g. ``{"fn": "mul"}`` requires ``node.op.fn == "mul"``.
    """

    name: str
    op_type: type
    quantifier: str = "1"
    constraints: dict = field(default_factory=dict)


@dataclass
class Group:
    """Repeating group of productions with backtracking.

    The parser tries all sub-productions in sequence. If any required
    sub-production (quantifier ``"1"`` or ``"+"``) fails, the whole
    group iteration fails and the cursor backtracks to before the
    attempt.

    ``quantifier`` controls group repetition:
      - ``"?"``: try once; skip if fails (all-or-nothing).
      - ``"*"``: repeat while the group fully matches (zero or more).
      - ``"+"``: like ``"*"`` but must succeed at least once.
    """

    name: str
    productions: list[Production]
    quantifier: str = "*"


type ChainGrammar = list[Production | Group]


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------


@dataclass
class ChainSegment:
    """One named segment of a parsed chain."""

    name: str
    node_ids: list[str] = field(default_factory=list)


@dataclass
class ChainMatch:
    """Result of parsing a fan-out-1 chain against a grammar."""

    root_node_id: str
    segments: list[ChainSegment] = field(default_factory=list)
    consumed: set[str] = field(default_factory=set)

    def get(self, name: str) -> list[str]:
        """All node ids for segments matching ``name`` (flattened).

        For group-scoped segments (e.g. ``"stage[0].pre_ops"``), pass the
        full scoped name. For convenience, passing just ``"pre_ops"``
        matches any segment whose name ends with ``.pre_ops`` or equals
        ``"pre_ops"``.
        """
        out: list[str] = []
        for seg in self.segments:
            if seg.name == name or seg.name.endswith(f".{name}"):
                out.extend(seg.node_ids)
        return out

    def get_groups(self, group_name: str) -> list[list[ChainSegment]]:
        """Return segment lists per group iteration.

        Segments from ``Group("stage", ...)`` iteration *i* are prefixed
        ``"stage[i]."``; this method collects them by index.
        """
        buckets: dict[int, list[ChainSegment]] = {}
        for seg in self.segments:
            # Match "group_name[N].anything"
            if not seg.name.startswith(f"{group_name}["):
                continue
            bracket_end = seg.name.index("]")
            idx = int(seg.name[len(group_name) + 1 : bracket_end])
            buckets.setdefault(idx, []).append(seg)
        return [buckets[i] for i in sorted(buckets)]


# ---------------------------------------------------------------------------
# Chain parser
# ---------------------------------------------------------------------------


def parse_chain(
    graph: Graph,
    start_nid: str,
    grammar: ChainGrammar,
    already_consumed: set[str] | None = None,
) -> ChainMatch | None:
    """Walk forward from ``start_nid`` along fan-out-1 edges, matching
    nodes against ``grammar`` productions.

    Returns the maximal match, or ``None`` if a required production
    (quantifier ``"1"`` or ``"+"``) can't be satisfied.

    The parser is **greedy**: for ``"*"`` / ``"+"``, it consumes as many
    nodes as possible before advancing to the next production.
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
# Top-level: find all non-overlapping matches in a graph
# ---------------------------------------------------------------------------


def match_grammar(graph: Graph, grammar: ChainGrammar) -> list[ChainMatch]:
    """Find all non-overlapping grammar matches, scanning in topo order.

    Replaces the old ``match_pattern()`` function. Each match consumes
    nodes; subsequent seeds skip consumed nodes.
    """
    consumed: set[str] = set()
    results: list[ChainMatch] = []
    for nid in graph.topological_order():
        if nid in consumed:
            continue
        match = parse_chain(graph, nid, grammar, consumed)
        if match is not None:
            consumed.update(match.consumed)
            results.append(match)
    return results


# ---------------------------------------------------------------------------
# Internal parser state + helpers
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
        # Undo segments and consumed additions since snapshot.
        removed_segs = self.segments[seg_len:]
        self.segments = self.segments[:seg_len]
        for seg in removed_segs:
            for nid in seg.node_ids:
                self.consumed.discard(nid)


def _parse_production(ctx: _ParseCtx, prod: Production, prefix: str) -> bool:
    """Try to match ``prod`` at ``ctx.cursor``. Returns True if the
    quantifier is satisfied. Advances cursor on each consumed node.

    ``prefix`` is prepended to segment names for group scoping
    (e.g. ``"stage."``).
    """
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
        # Undo any partial consumption.
        for nid in seg.node_ids:
            ctx.consumed.discard(nid)
        return False

    if seg.node_ids:
        ctx.segments.append(seg)
    return True


def _parse_group(ctx: _ParseCtx, group: Group) -> bool:
    """Try to match a ``Group`` — repeat its sub-productions as a unit.

    Each iteration's segments are prefixed with ``"group_name[i]."`` so
    ``get_groups()`` can split them by iteration index.
    """
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
    """Check whether a graph node matches a production."""
    node = ctx.graph.nodes.get(nid)
    if node is None:
        return False
    if nid in ctx.skip:
        return False
    if not isinstance(node.op, prod.op_type):
        return False
    for field_name, expected in prod.constraints.items():
        actual = getattr(node.op, field_name, None)
        if actual is None or str(actual) != str(expected):
            return False
    return True


def _sole_consumer(graph: Graph, nid: str, exclude: set[str]) -> str | None:
    """Return the sole consumer of ``nid``, or None if fan-out != 1.

    Consumers in ``exclude`` are not counted (they've been consumed by
    a different match or are in the skip set).
    """
    consumers = [cid for cid in graph.consumers(nid) if cid not in exclude]
    if len(consumers) == 1:
        return consumers[0]
    return None
