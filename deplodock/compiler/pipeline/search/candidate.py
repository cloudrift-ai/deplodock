"""Search-tree data classes — one ``Candidate`` per point in the
autotune space, plus its cursor / trace / per-step result records.

A ``Candidate`` is what the engine pops, advances by one rule
application, and pushes back. Forks at multi-option rewrite points
share a single deep-copied snapshot of the parent's graph and
materialize lazily on first ``.graph`` access — see :class:`ForkOrigin`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph

if TYPE_CHECKING:
    from deplodock.compiler.ir.base import Op
    from deplodock.compiler.pipeline.pattern import Match


@dataclass(frozen=True)
class TraceEntry:
    """One rule application in a candidate's history. ``choice_idx`` is
    the option picked at fork points (0 for deterministic single-option
    rules)."""

    rule_name: str
    choice_idx: int = 0


@dataclass
class Cursor:
    """Pipeline resume state for a ``Candidate``.

    * ``pass_idx`` — index of the pass to apply next.
    * ``rule_idx`` — index of the rule within the current pass to try next.
    * ``n_applied`` — number of functional rewrites in the current
      pass scan. When ``rule_idx`` wraps past the last rule with this
      counter ``> 0``, the engine restarts the scan (changes happened);
      with the counter ``== 0``, the engine advances to the next pass."""

    pass_idx: int = 0
    rule_idx: int = 0
    n_applied: int = 0

    def advance(
        self,
        result: RuleResult,
        n_rules: int,
        on_pass_finish: Callable[[int], None] | None = None,
    ) -> None:
        """Drive the cursor forward by one rule attempt.

        First, update ``n_applied`` (functional fires drive end-of-scan
        restart logic) and ``rule_idx`` (only advanced when no
        functional fire happened — in-place rebinds and zero-fire
        batches stay on the same graph state, so re-scanning the rule
        would loop or no-op). Then, when ``rule_idx`` reaches
        ``n_rules``, transition: restart from rule 0 if functional
        rewrites accumulated this scan, otherwise invoke ``on_pass_end``
        with the just-finished ``pass_idx`` and advance to the next
        pass."""
        self.n_applied += result.n_functional
        if result.n_functional == 0:
            self.rule_idx += 1
        if self.rule_idx < n_rules:
            return
        finished = self.n_applied == 0
        self.rule_idx = 0
        self.n_applied = 0
        if finished:
            if on_pass_finish is not None:
                on_pass_finish(self.pass_idx)
            self.pass_idx += 1

    def fork(self, n_applied_delta: int) -> Cursor:
        """A copy at the same ``(pass_idx, rule_idx)`` with ``n_applied``
        shifted by ``n_applied_delta`` — used when spawning autotune
        alternatives mid-batch."""
        return Cursor(self.pass_idx, self.rule_idx, self.n_applied + n_applied_delta)


@dataclass
class RuleResult:
    """Outcome of one rule-application iteration.

    * ``forks`` — alternative candidates spawned at autotune fork points
      (empty for deterministic rules).
    * ``n_functional`` — count of ``Graph`` (functional) rewrites applied
      to the candidate's own graph in this batch.
    * ``n_inplace`` — count of ``Op`` (in-place rebind) rewrites applied
      to the candidate's own graph in this batch."""

    forks: list[Candidate] = field(default_factory=list)
    n_functional: int = 0
    n_inplace: int = 0

    @property
    def fired(self) -> bool:
        return (self.n_functional + self.n_inplace) > 0


@dataclass(frozen=True)
class ForkOrigin:
    """Lazy-fork recipe. Materializing a Candidate from this means:
    deep-copy ``parent_snapshot``, build a fresh :class:`Match` against
    the copy, apply ``option`` to it, return the resulting graph.

    Multiple sibling forks share the same ``parent_snapshot`` via Python
    references — refcount frees it when the last lazy sibling has been
    materialized or dropped from the search queue.

    ``tip_key`` is precomputed as ``op_cache_key(option)`` so the search
    can file the fork into its ``_by_tip`` bucket without materializing
    the graph.
    """

    parent_snapshot: Graph
    rule_name: str
    match: Match
    option: Op | Graph
    choice_idx: int  # option index picked at the fork point; ``Candidate.apply`` records it on the trace
    # ``op_cache_key(option)`` when the option is an Op (the autotune
    # knob case); ``None`` for Graph-returning rewrites whose tip can't
    # be computed without materializing. ``None`` tips route to the
    # search's FIFO fallback queue, bypassing UCB selection — fine for
    # the rare functional fork point.
    tip_key: str | None


@dataclass
class Candidate:
    """A single point in the search space. The engine pops a candidate,
    advances it by one rule application attempt, pushes the resulting
    successor(s) back onto the search queue, and yields the candidate
    when ``cursor.pass_idx`` reaches the end of the pipeline.

    ``graph`` is owned by this candidate. The active rollout's
    Candidate holds a concrete ``_graph``; forks at multi-option
    rewrite points are *lazy* — they hold an ``_origin`` (see
    :class:`ForkOrigin`) and materialize on first ``.graph`` access.

    ``ctx`` is shared by reference. ``trace`` is the immutable history
    of rule applications on this branch. ``cursor`` is the pipeline
    cursor.

    ``last_rewritten`` lists the node IDs whose op was replaced by the
    most recent rule application — empty until the first rule fires."""

    ctx: Context
    trace: tuple[TraceEntry, ...] = ()
    cursor: Cursor = field(default_factory=Cursor)
    last_rewritten: tuple[str, ...] = ()
    _graph: Graph | None = field(default=None, repr=False)
    _origin: ForkOrigin | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        assert (self._graph is None) ^ (self._origin is None), "Candidate must have exactly one of _graph (concrete) or _origin (lazy)"

    @property
    def graph(self) -> Graph:
        """Materialize the graph on demand. Lazy forks copy the shared
        ``parent_snapshot`` once and apply their stored option in
        place; subsequent accesses return the cached result."""
        if self._graph is None:
            origin = self._origin
            assert origin is not None  # invariant enforced by __post_init__
            self._graph = origin.parent_snapshot.copy()
            self._origin = None  # drop the shared snapshot reference; refcount-GC kicks in
            self.apply(origin.rule_name, origin.match.remap(self._graph), origin.option, choice_idx=origin.choice_idx)
        return self._graph

    @graph.setter
    def graph(self, g: Graph) -> None:
        self._graph = g
        self._origin = None

    def apply(self, rule_name: str, match: Match, option: Op | Graph, *, choice_idx: int = 0) -> None:
        """Apply one rewrite outcome to ``self.graph`` in place and
        update ``trace`` + ``last_rewritten``. ``Op`` rebinds
        ``root.op`` (id, inputs, hints kept); ``Graph`` is a fragment
        spliced via ``Graph.splice``.

        On the ``Op`` path the chain ``Op.source`` is stamped with the
        op being replaced (unless the rule already set it) and the
        predecessor's ``knobs`` are merged forward — so the rewrite
        chain threads through every in-place rebind for free.

        ``choice_idx`` is the option index picked at this fork point
        (0 for the primary path, ``>0`` for the recorded fork that this
        Candidate descended from).
        """
        # Local imports to break the cycle with ``compiler.ir.base``.
        from deplodock.compiler.ir.base import Op as _Op  # noqa: PLC0415

        graph = self.graph
        if isinstance(option, _Op):
            old_op = graph.nodes[match.root_node_id].op
            if option is not old_op and option.source is None:
                option.source = old_op
                option.knobs = {**old_op.knobs, **option.knobs}
            graph.nodes[match.root_node_id].op = option
        else:
            assert isinstance(option, Graph), f"rule {rule_name} returned {type(option).__name__}; expected Graph, Op, list, or RuleSkipped"
            graph.splice(option, consumed=match.consumed, output=match.output or match.root_node_id)
        self.trace = (*self.trace, TraceEntry(rule_name, choice_idx))
        self.last_rewritten = (match.root_node_id,)
