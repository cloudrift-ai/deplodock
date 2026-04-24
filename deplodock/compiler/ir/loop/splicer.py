"""Worklist-driven splicer for a DAG of ``LoopOp``s.

Three public entry points wrap the same underlying ``_Splicer``:

- :func:`splice_loop_ops` — pairwise producer / consumer helper.
- :func:`splice_loops` — tag-generic N-way: caller supplies ``loops``
  (tag → ``LoopOp``), ``splice_edges`` ((origin_tag, src) →
  (target_tag, target_output)), and ``input_remap``. Sink is derived
  as the one tag that never appears as a splice target.
- :func:`splice_graph` — consumes a ``Graph`` fragment directly;
  classifies each Load by its node.inputs edge (LoopOp → splice,
  otherwise → external slot in first-seen order).

Algorithm. Seed: every ``Write`` of the sink loop. Each iteration pops
one pending dep and emits its def, queueing that def's own deps.
Resolution dispatches on stmt kind:

- **Load on a splice edge** — emit a copy alias at the demand scope;
  σ is solved by pairing target's ``Write.index`` against the reader's
  σ-substituted index, and the target's ``Write.value`` is queued under
  the solved σ. The target's expression chain reconstructs piecemeal.
- **Accum** — freshen its reduce axis, place
  ``Loop(fresh_reduce_axis, Accum(...))`` at
  ``_scope_for_axes(ref_scope, required_c_axes)``, queue the Accum's
  ``value`` under σ extended with the fresh reduce binding.
- **Plain Assign / Select / Load** (non-splice source) — ``rewrite``
  the original stmt through ``(rename_ssa, sigma)`` and insert at the
  demand scope.

Unified dedup. A single table keyed on
``(origin, name, emit_scope, σ.restrict(live_axes))`` decides whether
to share an existing emission or allocate a fresh one. ``live_axes``
comes from ``LoopMeta`` and is the set of axes transitively reachable
through the stmt's Expr subtrees — σ bindings outside that set are
irrelevant and collapsed. Same key → share; different emit scope or
different live-σ → emit twice. This handles plain-stmt sharing, Accum
scope multiplicity (SDPA QK^T at softmax-max vs softmax-output), and
multi-output splice targets uniformly.

``LoopBuilder.insert`` is pure tree-splicing: descend the body along
the enclosure path, creating ``Loop`` nodes if missing, prepend at the
leaf. Always-prepend yields defined-before-use ordering naturally,
because the worklist resolves deps in reverse-topological order —
consumers demand before producers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from deplodock.compiler.ir.expr import Expr, Literal, Var, free_vars
from deplodock.compiler.ir.loop.builder import LoopBuilder
from deplodock.compiler.ir.loop.ir import (
    Accum,
    Assign,
    Axis,
    Load,
    Loop,
    LoopMeta,
    LoopOp,
    Scope,
    Select,
    Stmt,
    Write,
)
from deplodock.compiler.ir.loop.sigma import Sigma


class _NotSupported(Exception):
    """Pattern not handled — caller converts to ``None`` return."""


# Unified binding key: ``(origin, name, emit_scope, sigma.restrict(enclosing))``.
# ``emit_scope`` is where the stmt lands in the merged body; ``sigma`` is
# restricted to the stmt's own enclosing axis names — the only bindings that
# affect its rewrite (Load.index / Select.select) or its dep resolution.
_BindKey = tuple[str, str, Scope, Sigma]


@dataclass
class _Demand:
    """A pending dep in the worklist.

    ``bound_as`` is the fresh name the dep's def will bind in the merged
    body — allocated at queue time so callers can reference it without
    waiting for resolution.
    """

    name: str
    origin: str  # tag identifying which loop this def came from
    sigma: Sigma
    demand_scope: Scope
    bound_as: str


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def splice_loop_ops(producer: LoopOp, consumer: LoopOp, source: str) -> LoopOp | None:
    """Pairwise splicer: inline ``producer`` into every ``consumer`` Load
    whose ``source`` (buf name) matches the producer's output. Returns
    ``None`` when the pattern isn't supported.

    Thin wrapper over ``splice_loops``. The merged kernel's Loads keep
    their original buf names — no remap needed since names are stable
    across kernels. The producer's output buf name comes from its (sole)
    Write.
    """
    prod_writes = [s for s in producer if isinstance(s, Write)]
    if len(prod_writes) != 1:
        return None
    prod_buf = prod_writes[0].output
    return splice_loops(
        loops={"producer": producer, "consumer": consumer},
        splice_edges={("consumer", source): ("producer", prod_buf)},
    )


def splice_loops(
    loops: dict[str, LoopOp],
    splice_edges: dict[tuple[str, str], tuple[str, str]],
) -> LoopOp | None:
    """Splice a DAG of ``LoopOp``s into one merged kernel.

    ``loops`` maps an opaque tag to each participating ``LoopOp``.
    ``splice_edges`` identifies which Loads are inlined from another
    registered loop: key ``(origin_tag, source_buf)`` → value
    ``(target_tag, target_output_buf)`` meaning "this loop's Load whose
    ``source`` is ``source_buf`` reads ``target_tag``'s Write whose
    ``output`` is ``target_output_buf`` and should be inlined."

    Non-splice Loads keep their original ``source`` buf names — buf
    identity is global, no remap needed. The sink — the loop whose Writes
    seed the traversal — is derived from ``splice_edges``: it's the
    unique tag in ``loops`` that never appears as a splice target.
    Returns ``None`` if the sink is ambiguous (cycle or multiple sinks)
    or if any splice edge hits an unsupported pattern.
    """
    target_tags = {tag for tag, _out in splice_edges.values()}
    candidates = [tag for tag in loops if tag not in target_tags]
    if len(candidates) != 1:
        return None
    root = candidates[0]
    try:
        return _Splicer(
            loops={tag: op.analyze() for tag, op in loops.items()},
            splice_edges=splice_edges,
            root=root,
        ).run()
    except (_NotSupported, ValueError):
        return None


def splice_graph(graph) -> tuple[LoopOp, list[str]] | None:
    """Splice a subgraph of ``LoopOp`` nodes into one merged kernel.

    Each ``LoopOp`` node in ``graph`` becomes a registered loop tagged
    by its node id. Within each LoopOp node, a Load whose source points
    at another ``LoopOp`` node becomes a splice edge; a Load whose
    source points at a non-``LoopOp`` node (e.g. ``InputOp``) becomes
    an external read, assigned a slot in first-seen order.

    Returns ``(merged_op, external_node_ids)`` where ``external_node_ids``
    is the list of non-``LoopOp`` input node ids in merged-slot order.
    Returns ``None`` if the graph has zero / multiple outputs, the sink
    is not a ``LoopOp``, or any splice edge hits an unsupported pattern.
    """
    if len(graph.outputs) != 1:
        return None
    root = graph.outputs[0]
    root_node = graph.nodes.get(root)
    if root_node is None or not isinstance(root_node.op, LoopOp):
        return None

    loop_ids = {nid for nid, n in graph.nodes.items() if isinstance(n.op, LoopOp)}
    loops: dict[str, LoopOp] = {nid: graph.nodes[nid].op for nid in loop_ids}
    splice_edges: dict[tuple[str, str], tuple[str, str]] = {}
    external_order: list[str] = []
    seen_external: set[str] = set()

    for nid in loop_ids:
        node_op = loops[nid]
        for ld in node_op.loads:
            inp = ld.source
            # A Load is a splice edge if its source buf names another LoopOp node;
            # otherwise it's an external input. We key edges off the buf name
            # (Load.source is now the producing node's id), not a positional
            # input index — so a single edge entry covers every Load that reads
            # the same producer.
            if inp in loop_ids:
                splice_edges[(nid, inp)] = (inp, inp)  # producer's Write.output is its node id
            elif inp not in seen_external:
                seen_external.add(inp)
                external_order.append(inp)

    merged = splice_loops(loops=loops, splice_edges=splice_edges)
    if merged is None:
        return None
    return merged, external_order


# ---------------------------------------------------------------------------
# _Splicer — all per-splice state + the worklist loop
# ---------------------------------------------------------------------------


class _Splicer(LoopBuilder):
    """Multi-loop splicer driven by an explicit splice-edge graph.

    Each registered loop has a tag (opaque string). ``splice_edges``
    identifies which Loads are inlined from another registered loop;
    all other Loads are re-indexed into the merged kernel's external
    input list via ``input_remap``. ``root`` names the tag whose Writes
    seed the traversal — typically the chain's final consumer.

    Inherits body building (``insert`` / ``fresh`` / ``finish``) from
    ``LoopBuilder``; adds the worklist of pending demands and the dedup
    table that keeps the merged body minimal. Worklist dep-resolution
    is reverse-topological — producers demanded after consumers — so
    the builder's prepend-at-leaf behavior yields defined-before-use
    ordering naturally.
    """

    def __init__(
        self,
        *,
        loops: dict[str, LoopMeta],
        splice_edges: dict[tuple[str, str], tuple[str, str]],
        root: str,
    ) -> None:
        used: set[str] = set()
        for meta in loops.values():
            used |= _collect_names(meta.op)
        super().__init__(used_names=used)
        self.loops = loops
        self.splice_edges = splice_edges
        self.root = root
        self._pending: deque[_Demand] = deque()
        # Dedup: a stmt is uniquely identified by its (origin, name), the
        # emit scope it lands at in the merged body, and the σ restricted
        # to its own enclosing — the only bindings that affect its rewrite.
        # Same key → share a single emission.
        self._binding: dict[_BindKey, str] = {}

    def run(self) -> LoopOp:
        self._seed()
        while self._pending:
            self._resolve(self._pending.popleft())
        return LoopOp(body=self.finish())

    # -- Seed: every root Write, with its value queued ----------------------

    def _seed(self) -> None:
        for w, scope in self.loops[self.root].writes:
            v_bound = self._ensure_dep(w.value, self.root, Sigma(), scope)
            self.insert(Write(output=w.output, index=w.index, value=v_bound), scope)

    # -- Dep binding: look up or queue --------------------------------------

    def _ensure_dep(self, name: str, origin: str, sigma: Sigma, ref_scope: Scope) -> str:
        """Return the merged-body name for ``(origin, name)`` at the emit
        scope induced by ``ref_scope`` and σ. Queue a new demand the first
        time the key is seen.
        """
        meta = self.loops[origin]
        if name not in meta.defs:
            raise _NotSupported

        required_axes = tuple(_remap_axis_name(a, sigma) for a in meta.scopes[name].enclosing)
        emit_scope = _scope_for_axes(ref_scope, required_axes)

        # σ restricted to axes transitively used in Expr subtrees reachable
        # from this stmt. Bindings outside this set don't affect any emitted
        # stmt, so keeping them in the key would cause spurious duplicate
        # emissions.
        restricted = sigma.restrict(meta.live_axes[name])
        key = (origin, name, emit_scope, restricted)
        existing = self._binding.get(key)
        if existing is not None:
            return existing
        bound = self.fresh(name)
        self._binding[key] = bound
        self._pending.append(_Demand(name=name, origin=origin, sigma=sigma, demand_scope=emit_scope, bound_as=bound))
        return bound

    # -- Resolution dispatch -------------------------------------------------

    def _resolve(self, d: _Demand) -> None:
        stmt = self.loops[d.origin].defs[d.name]

        if isinstance(stmt, Load):
            edge = self.splice_edges.get((d.origin, stmt.source))
            if edge is not None:
                target_tag, target_output_buf = edge
                self._resolve_splice_load(stmt, d, target_tag, target_output_buf)
            else:
                self._resolve_external_load(stmt, d)
        elif isinstance(stmt, Accum):
            self._resolve_accum(stmt, d)
        elif isinstance(stmt, (Assign, Select)):
            self._resolve_plain(stmt, d)
        else:
            raise _NotSupported

    def _resolve_plain(self, stmt: Stmt, d: _Demand) -> None:
        """Generic Assign / Select emission — rewrite the stmt with fresh args
        and σ-substituted Exprs, insert at ``d.demand_scope``."""
        rename = {arg: self._ensure_dep(arg, d.origin, d.sigma, d.demand_scope) for arg in stmt.deps()}
        rename[stmt.name] = d.bound_as  # type: ignore[attr-defined]
        self.insert(stmt.rewrite(lambda n: rename.get(n, n), d.sigma), d.demand_scope)

    def _resolve_external_load(self, stmt: Load, d: _Demand) -> None:
        """A Load that isn't a splice edge — keep its buf name as-is (buf
        identity is global across kernels), σ-sub the index, emit."""
        new_index = tuple(d.sigma.apply(e) for e in stmt.index)
        self.insert(Load(name=d.bound_as, source=stmt.source, index=new_index), d.demand_scope)

    def _resolve_splice_load(self, stmt: Load, d: _Demand, target_tag: str, target_output_buf: str) -> None:
        """A Load that's a splice edge to another registered loop — emit a
        copy alias and queue the target loop's ``Write.value`` under the
        solved σ. The target's expression chain reconstructs piecemeal over
        subsequent iterations. ``target_output_buf`` selects which ``Write``
        of the target is the splice source when the target has multiple outputs."""
        target = self.loops[target_tag]
        target_write = next((w for w, _ in target.writes if w.output == target_output_buf), None)
        if target_write is None:
            raise _NotSupported
        effective_index = tuple(d.sigma.apply(e) for e in stmt.index)
        sigma = _solve_sigma(target_write.index, effective_index, {a.name for a in target.op.axes})
        if sigma is None:
            raise _NotSupported
        v_bound = self._ensure_dep(target_write.value, target_tag, sigma, d.demand_scope)
        self.insert(Assign(name=d.bound_as, op="copy", args=(v_bound,)), d.demand_scope)

    def _resolve_accum(self, stmt: Accum, d: _Demand) -> None:
        """Emit ``Loop(fresh_reduce_axis, [Accum(bound, value_bound, op)])`` at
        ``d.demand_scope``. The Accum's value is queued under σ extended with
        the fresh reduce-axis binding."""
        orig_axis = self.loops[d.origin].reduce_axes[stmt.name]
        fresh_name = self.fresh(orig_axis.name)
        reduce_axis = Axis(name=fresh_name, extent=orig_axis.extent)
        inner_sigma = d.sigma.extend(orig_axis.name, Var(fresh_name))
        inner_scope = Scope(enclosing=d.demand_scope.enclosing + (reduce_axis,))
        value_bound = self._ensure_dep(stmt.value, d.origin, inner_sigma, inner_scope)
        self.insert(Accum(name=d.bound_as, value=value_bound, op=stmt.op), inner_scope)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _scope_for_axes(ref_scope: Scope, required: tuple[str, ...]) -> Scope:
    """Shortest prefix of ``ref_scope`` whose axis set contains ``required``.

    Used two ways:
    - For Accums: places the reduce ``Loop`` at the innermost consumer
      scope where all σ-mapped producer enclosing axes are visible (today's
      behavior — further hoisting is left to later passes).
    - For plain producer stmts: picks the emit scope from the consumer's
      nest, tolerating producer's free-axis order differing from consumer's.
      A matmul producer ``(a0, a1, a2)`` σ-maps to consumer ``(a0, a2, a1)``
      (shuffled), but the consumer's scope ``(a0, a1, a2)`` covers the same
      axis set; emitting at the consumer's nest avoids a duplicate Loop tree.
    """
    names = tuple(a.name for a in ref_scope.enclosing)
    remaining = set(required)
    k = 0
    while remaining and k < len(names):
        remaining.discard(names[k])
        k += 1
    if remaining:
        raise _NotSupported
    return Scope(enclosing=ref_scope.enclosing[:k])


def _remap_axis_name(axis: Axis, sigma: Sigma) -> str:
    """Pick the merged-kernel axis that ``axis`` lands under, given σ's target.

    σ can map a producer axis to any single-variable expression built from the
    consumer's axes — ``Var(b)`` (plain rename), ``Var(b) + 5`` (offset slice),
    ``2*Var(b)`` (strided slice), ``3*Var(b) + 1`` (strided+offset), etc. In
    all these cases the merged kernel's enclosing loop iterates ``b``; every
    occurrence of ``Var(a)`` in the producer body is substituted to the full
    expression by the caller's σ-rewrite, so the arithmetic lands inside the
    stmt's Exprs rather than on the loop axis itself.

    Rejected shapes: targets that reference two or more variables (would need
    the merged kernel to iterate over multiple consumer axes here — requires
    a scope-splitting refactor beyond this hook) and targets with zero
    variables (σ fixes the axis to a constant — the def should be hoisted
    out of any loop, which isn't representable as a single axis name).
    """
    target = sigma.get(axis.name)
    if target is None:
        return axis.name
    if isinstance(target, Var):
        return target.name
    vars_in_target = free_vars(target)
    if len(vars_in_target) == 1:
        return next(iter(vars_in_target))
    raise _NotSupported


def _solve_sigma(
    writer: tuple[Expr, ...],
    reader: tuple[Expr, ...],
    producer_axes: set[str],
) -> Sigma | None:
    """Solve per-dim pairing ``writer[k] == reader[k]``. Supported writer
    forms: ``Var(a)`` (``a`` in ``producer_axes``) → bind ``a → reader[k]``;
    ``Literal(c)`` → no binding. Anything else → ``None``."""
    if len(writer) != len(reader):
        return None
    mapping: dict[str, Expr] = {}
    for w, r in zip(writer, reader, strict=True):
        if isinstance(w, Literal):
            continue
        if isinstance(w, Var) and w.name in producer_axes:
            existing = mapping.get(w.name)
            if existing is not None and existing != r:
                return None
            mapping[w.name] = r
            continue
        return None
    return Sigma(mapping)


def _collect_names(op: LoopOp) -> set[str]:
    """All SSA names plus all axis names used anywhere in ``op``."""
    names: set[str] = set()
    for s in op:
        if isinstance(s, Loop):
            names.add(s.axis.name)
        elif isinstance(s, (Load, Assign, Select, Accum)):
            names.add(s.name)
    return names
