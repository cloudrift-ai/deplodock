"""Pattern matcher value types — ``Pattern``, ``Match``, plus
:class:`Pass` and :class:`Pipeline`, the per-run description of pass +
rule layout.

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


@dataclass
class Pass:
    """One pipeline pass: a named, indexed list of rules.

    * ``name`` — pass directory name (e.g. ``"frontend/decomposition"``),
      or ``""`` for empty / nameless pass slots (early-exit stubs).
    * ``rules`` — the rules in this pass, in load order.
    * ``index`` — 0-based position in the pipeline.

    Stamps each rule's ``pass_`` backref on construction so a ``Match``
    that carries a ``Rule`` can resolve pass metadata without holding a
    separate index.
    """

    name: str
    rules: list[Rule]
    index: int = 0

    def __post_init__(self) -> None:
        for r in self.rules:
            r.pass_ = self


@dataclass(frozen=True)
class Pipeline:
    """Frozen per-run layout of the rewrite pipeline. Constructed once
    per ``run_autotune`` call and reached from every :class:`Match` via
    ``match.rule.pass_``.

    :meth:`match` is the only entry point for pattern matching: it
    walks the graph for one rule and stamps the rule onto every Match.
    Tests / standalone callers that just want pattern matching can
    build a one-rule Pipeline via :meth:`from_pattern`."""

    passes: list[Pass]

    def rule_at(self, pass_idx: int, rule_idx: int) -> Rule:
        return self.passes[pass_idx].rules[rule_idx]

    def pass_at(self, pass_idx: int) -> Pass:
        return self.passes[pass_idx]

    def n_rules(self, pass_idx: int) -> int:
        return len(self.passes[pass_idx].rules)

    def n_passes(self) -> int:
        return len(self.passes)

    def match(self, graph: Graph, rule: Rule) -> list[Match]:
        """Enumerate every live pattern match for ``rule`` against
        ``graph``. Stamps ``is_last=True`` on the last surviving match
        so the rewriter knows which apply closes out the rule batch
        (cursor advance flows through ``Candidate.try_rewrite`` /
        ``Candidate.apply``). Drops matches that fail
        :meth:`Match.is_alive` — an earlier match in the same batch
        may have removed a consumed node."""
        results: list[Match] = []
        for nid in graph.topological_order():
            m = _match_at(graph, nid, rule)
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
        rule = Rule(name="__test__", pattern=pattern)
        return cls(passes=[Pass(name="__test__", rules=[rule], index=0)])


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

    ``rule`` locates this match in the run's pipeline so the rewriter
    can derive ``rule_name`` / ``pass_name`` / ``n_rules`` /
    ``pass_idx`` from one source. Set by :meth:`Pipeline.match` at
    construction time (use :meth:`Pipeline.from_pattern` to build a
    one-rule Pipeline for standalone / test callers). ``is_last`` is
    stamped on the last live match returned by :meth:`Pipeline.match`
    so ``Candidate.try_rewrite`` knows when to advance the cursor.

    Use the helpers (``root``, ``node()``, ``input()``, ``is_alive()``)
    to resolve ids to ``Node`` objects through ``graph`` — they're the
    intended access pattern for rules that need graph-wide lookups.
    """

    graph: Graph
    root_node_id: str
    rule: Rule | None = None
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
    def rule_name(self) -> str | None:
        return self.rule.name if self.rule is not None else None

    @property
    def pass_(self) -> Pass | None:
        return self.rule.pass_ if self.rule is not None else None

    @property
    def pass_name(self) -> str | None:
        p = self.pass_
        return (p.name or None) if p is not None else None

    @property
    def pass_idx(self) -> int | None:
        p = self.pass_
        return p.index if p is not None else None

    @property
    def n_rules(self) -> int:
        p = self.pass_
        return len(p.rules) if p is not None else 0

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
            rule=self.rule,
            nodes=dict(self.nodes),
            consumed=set(self.consumed),
            output=self.output,
            is_last=self.is_last,
            _identities=identities,
        )


def _match_at(graph: Graph, start: str, rule: Rule) -> Match | None:
    cursor: str | None = start
    nodes: dict[str, str] = {}
    consumed: set[str] = set()
    identities: dict[str, int] = {}
    for prod in rule.pattern:
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
        rule=rule,
        nodes=nodes,
        consumed=consumed,
        _identities=identities,
    )


def _check_constraints(node: Node, prod: Pattern) -> bool:
    return all(str(getattr(node.op, k, None)) == str(v) for k, v in prod.constraints.items())


def _sole_consumer(graph: Graph, nid: str) -> str | None:
    consumers = graph.consumers(nid)
    return consumers[0] if len(consumers) == 1 else None


__all__ = ["Match", "Pass", "Pattern", "Pipeline", "Rule"]
