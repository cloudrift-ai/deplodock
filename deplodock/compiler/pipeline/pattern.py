"""Pattern matcher value types — ``Pattern``, ``Match``, ``match_pattern``,
plus :class:`Pipeline`, the per-run description of pass + rule layout.

Lives in its own module so the engine *and* the search policies can
both import these without cycling. ``engine.py`` owns the rule-loader
and rewrite dispatcher (which depend on ``Match``); ``Search.push``
takes a ``Match`` so policies that care about the rewrite site can
read it off the engine hand-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.pipeline.rule import Rule


@dataclass
class Pattern:
    """One node in a chain-match pattern.

    ``constraints`` is a dict of ``field_name → expected_value`` checks
    applied to ``node.op`` (e.g. ``{"fn": "softmax"}``).
    """

    name: str
    op_type: type
    constraints: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Pipeline:
    """Frozen per-run layout of the rewrite pipeline: pass directory
    names and the rules in each pass. Constructed once per
    ``run_autotune`` call and threaded onto every :class:`Match` so
    rule / pass metadata has a single source of truth instead of
    being re-stamped on each match.

    ``passes`` may include ``""`` placeholders for empty / nameless
    pass slots (e.g. early-exit stubs); :meth:`pass_name` returns
    ``None`` for those so dump bookkeeping skips them.

    :meth:`match` is the only entry point for pattern matching: it
    locates the rule via ``(pass_idx, rule_idx)``, walks the graph,
    and stamps every Match with this Pipeline + indices so callers
    can derive rule / pass metadata. Tests / standalone callers that
    just want pattern matching can build a one-rule Pipeline via
    :meth:`from_pattern`."""

    passes: list[str]
    rules_per_pass: list[list[Rule]]

    def rule(self, pass_idx: int, rule_idx: int) -> Rule:
        return self.rules_per_pass[pass_idx][rule_idx]

    def pass_name(self, pass_idx: int) -> str | None:
        return self.passes[pass_idx] or None

    def n_rules(self, pass_idx: int) -> int:
        return len(self.rules_per_pass[pass_idx])

    def n_passes(self) -> int:
        return len(self.passes)

    def match(self, graph: Graph, pass_idx: int, rule_idx: int) -> list[Match]:
        """Enumerate every live pattern match for the rule at
        ``(pass_idx, rule_idx)`` against ``graph``. Stamps
        ``is_last=True`` on the last surviving match so the rewriter
        knows which apply closes out the rule batch (cursor advance
        flows through ``Candidate.try_rewrite`` /
        ``Candidate.apply``). Drops matches that fail
        :meth:`Match.is_alive` — an earlier match in the same batch
        may have removed a consumed node."""
        pattern = self.rule(pass_idx, rule_idx).pattern
        results: list[Match] = []
        for nid in graph.topological_order():
            m = _match_at(graph, nid, pattern, self, pass_idx, rule_idx)
            if m is not None and m.is_alive():
                results.append(m)
        if results:
            results[-1].is_last = True
        return results

    @classmethod
    def from_pattern(cls, pattern: list[Pattern]) -> Pipeline:
        """Test/standalone helper: build a single-pass, single-rule
        Pipeline whose only rule wraps ``pattern`` (no ``rewrite``).
        Lets pattern-matching tests drive :meth:`match` without
        setting up the full engine pipeline."""
        return cls(passes=["__test__"], rules_per_pass=[[Rule(name="__test__", pattern=pattern)]])


@dataclass
class Match:
    """Result of matching a pattern against a graph.

    ``graph`` is the graph this match was built against (rules access
    it via ``match.graph`` for ad-hoc lookups). ``nodes`` maps each
    pattern entry's name to the matched node id. ``consumed`` and
    ``output`` may be overwritten by the rewrite function to control
    which nodes the rewriter removes and which node its edges get
    redirected to. ``output`` defaults to ``root_node_id`` when left
    as ``None``.

    ``pipeline`` + ``pass_idx`` + ``rule_idx`` locate this match in
    the run's pipeline so the rewriter can derive ``rule`` /
    ``rule_name`` / ``pass_name`` / ``n_rules`` from one source.
    Set by :meth:`Pipeline.match` at construction time (use
    :meth:`Pipeline.from_pattern` to build a one-rule Pipeline for
    standalone / test callers). ``is_last`` is stamped on the last
    live match returned by :meth:`Pipeline.match` so
    ``Candidate.try_rewrite`` knows when to advance the cursor.

    Use the helpers (``root``, ``node()``, ``input()``, ``is_alive()``)
    to resolve ids to ``Node`` objects through ``graph`` — they're the
    intended access pattern for rules that need graph-wide lookups.
    """

    graph: Graph
    root_node_id: str
    pipeline: Pipeline | None = None
    pass_idx: int = 0
    rule_idx: int = 0
    nodes: dict[str, str] = field(default_factory=dict)
    consumed: set[str] = field(default_factory=set)
    output: str | None = None
    is_last: bool = False
    # Snapshot of id(Node) at match time for every consumed node. The
    # ``is_alive`` check uses this to detect the case where an earlier
    # match in the same batch removed a consumed node and a different
    # node was added at the same id (e.g. splicer auto-rename hitting
    # a recently-freed name). Pure id-existence wouldn't catch that.
    _identities: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def rule(self) -> Rule | None:
        if self.pipeline is None:
            return None
        return self.pipeline.rule(self.pass_idx, self.rule_idx)

    @property
    def rule_name(self) -> str | None:
        rule = self.rule
        return rule.name if rule is not None else None

    @property
    def pass_name(self) -> str | None:
        if self.pipeline is None:
            return None
        return self.pipeline.pass_name(self.pass_idx)

    @property
    def n_rules(self) -> int:
        if self.pipeline is None:
            return 0
        return self.pipeline.n_rules(self.pass_idx)

    @property
    def root(self) -> Node:
        """The root ``Node`` (matched by the first ``Pattern`` entry)."""
        return self.graph.nodes[self.root_node_id]

    def node(self, name_or_id: str) -> Node:
        """Resolve a pattern name (e.g. ``"producer"``) OR a raw node id
        to the current ``Node`` in ``graph``. Raises ``KeyError`` if the
        node has been removed."""
        nid = self.nodes.get(name_or_id, name_or_id)
        return self.graph.nodes[nid]

    def input(self, i: int) -> Node | None:
        """Root's ``i``-th input as a ``Node``, or ``None`` when ``i``
        exceeds the input count or the input node was removed."""
        root = self.root
        if i >= len(root.inputs):
            return None
        return self.graph.nodes.get(root.inputs[i])

    def is_alive(self) -> bool:
        """``True`` when every consumed node still resolves to the same
        ``Node`` object captured at match time. Catches both removal and
        the "removed-then-re-added under same id" case."""
        for nid in self.consumed:
            n = self.graph.nodes.get(nid)
            if n is None or id(n) != self._identities.get(nid):
                return False
        return True

    def remap(self, graph: Graph) -> Match:
        """Build a fresh ``Match`` against ``graph`` (a copy of the
        original) that mirrors this match's ids. Re-snapshots
        ``_identities`` against the new graph's nodes so ``is_alive``
        still works after the copy. Used when materializing a lazy
        fork — the fork copies the parent's snapshot, then needs a
        match anchored on the copy."""
        identities = {nid: id(graph.nodes[nid]) for nid in self.consumed if nid in graph.nodes}
        return Match(
            graph=graph,
            root_node_id=self.root_node_id,
            pipeline=self.pipeline,
            pass_idx=self.pass_idx,
            rule_idx=self.rule_idx,
            nodes=dict(self.nodes),
            consumed=set(self.consumed),
            output=self.output,
            is_last=self.is_last,
            _identities=identities,
        )


def _match_at(
    graph: Graph,
    start: str,
    pattern: list[Pattern],
    pipeline: Pipeline | None,
    pass_idx: int,
    rule_idx: int,
) -> Match | None:
    cursor: str | None = start
    nodes: dict[str, str] = {}
    consumed: set[str] = set()
    identities: dict[str, int] = {}
    for prod in pattern:
        if cursor is None:
            return None
        node = graph.nodes.get(cursor)
        if node is None or not isinstance(node.op, prod.op_type):
            return None
        if not _check_constraints(node, prod):
            return None
        nodes[prod.name] = cursor
        consumed.add(cursor)
        identities[cursor] = id(node)
        cursor = _sole_consumer(graph, cursor)
    return Match(
        graph=graph,
        root_node_id=start,
        pipeline=pipeline,
        pass_idx=pass_idx,
        rule_idx=rule_idx,
        nodes=nodes,
        consumed=consumed,
        _identities=identities,
    )


def _check_constraints(node: Node, prod: Pattern) -> bool:
    return all(str(getattr(node.op, k, None)) == str(v) for k, v in prod.constraints.items())


def _sole_consumer(graph: Graph, nid: str) -> str | None:
    consumers = graph.consumers(nid)
    return consumers[0] if len(consumers) == 1 else None


__all__ = ["Match", "Pattern", "Pipeline", "Rule"]
