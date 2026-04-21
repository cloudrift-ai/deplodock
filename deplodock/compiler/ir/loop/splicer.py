"""Worklist-driven splicer for adjacent ``LoopOp``s.

Builds the merged body one statement at a time. Seed: every consumer
``Write``. Each iteration pops one pending dep from the worklist and
emits its def, queueing that def's own deps in turn. Three dep shapes:

- **Consumer Load from the producer** — emit a copy alias at the demand
  scope; queue the producer's ``Write.value`` under σ = solve(writer,
  reader). The producer's expression chain lands piecemeal over
  subsequent iterations.
- **Accum (producer or consumer)** — freshen the reduce axis, place a
  new ``Loop(reduce_axis, Accum(...))`` at ``_accum_enclosure`` of the
  demand scope, and queue the Accum's ``value``. Scope-keyed dedup:
  the same Accum demanded at different ``required_c_axes`` gets two
  separate materializations (SDPA's QK^T at softmax-max vs output).
- **Plain Assign / Select / Load(non-producer source)** — ``rewrite``
  the original stmt with fresh names and σ-substituted Exprs; insert at
  the stmt's natural scope (consumer: σ-remapped enclosing, which for
  consumer origin is usually identity; producer: σ-remapped enclosing).

``LoopBuilder.insert`` is pure tree-splicing: descend the body along
the enclosure path, creating ``Loop`` nodes if missing, prepend at the
leaf. Always-prepend yields defined-before-use ordering naturally,
because the worklist resolves deps in reverse-topological order —
consumers demand before producers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from deplodock.compiler.ir.expr import Expr, Literal, Sigma, Var
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
from deplodock.compiler.ir.tensor_ir import ElementwiseOp


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


def splice_loop_chain(
    loops: dict[str, LoopOp],
    splice_edges: dict[tuple[str, int], str],
    input_remap: dict[tuple[str, int], int],
    root: str,
) -> LoopOp | None:
    """Splice a graph of LoopOps into one merged kernel.

    ``loops`` maps an opaque tag to each participating ``LoopOp``.
    ``splice_edges`` identifies which Loads are inlined from another
    registered loop: key ``(origin_tag, source_idx)`` → value ``target_tag``,
    meaning "the origin loop's Load at source_idx reads target_tag's
    output and should be inlined."
    ``input_remap`` assigns every non-splice Load's ``(origin_tag,
    source_idx)`` to a slot in the merged kernel's external input list.
    ``root`` names the tag whose ``Write``s seed the traversal — typically
    the chain's final consumer.

    Returns ``None`` if any splice edge hits an unsupported pattern
    (non-Var/Literal writer index, arithmetic σ targets, etc.)."""
    try:
        return _Splicer(
            loops={tag: op.analyze() for tag, op in loops.items()},
            splice_edges=splice_edges,
            input_remap=input_remap,
            root=root,
        ).run()
    except (_NotSupported, ValueError):
        return None


def splice_loop_ops(producer: LoopOp, consumer: LoopOp, source: int) -> LoopOp | None:
    """Pairwise splicer: inline ``producer`` into every ``consumer`` Load
    that targets ``source``. Returns ``None`` when the pattern isn't
    supported.

    Thin wrapper over ``splice_loop_chain``: producer inputs occupy slots
    ``[0, n_prod)`` in the merged kernel; the consumer's surviving inputs
    follow in declaration order, skipping the spliced slot.
    """
    n_prod = producer.num_inputs
    input_remap: dict[tuple[str, int], int] = {("producer", i): i for i in range(n_prod)}
    next_slot = n_prod
    for i in range(consumer.num_inputs):
        if i == source:
            continue
        input_remap[("consumer", i)] = next_slot
        next_slot += 1
    return splice_loop_chain(
        loops={"producer": producer, "consumer": consumer},
        splice_edges={("consumer", source): "producer"},
        input_remap=input_remap,
        root="consumer",
    )


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
        splice_edges: dict[tuple[str, int], str],
        input_remap: dict[tuple[str, int], int],
        root: str,
    ) -> None:
        used: set[str] = set()
        for meta in loops.values():
            used |= _collect_names(meta.op)
        super().__init__(used_names=used)
        self.loops = loops
        self.splice_edges = splice_edges
        self.input_remap = input_remap
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
            target_tag = self.splice_edges.get((d.origin, stmt.source))
            if target_tag is not None:
                self._resolve_splice_load(stmt, d, target_tag)
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
        resolved = {arg: self._ensure_dep(arg, d.origin, d.sigma, d.demand_scope) for arg in stmt.deps()}
        self.insert(stmt.rewrite(d.bound_as, resolved, d.sigma), d.demand_scope)

    def _resolve_external_load(self, stmt: Load, d: _Demand) -> None:
        """A Load that isn't a splice edge — remap source via ``input_remap``,
        σ-sub the index, emit as-is."""
        new_src = self.input_remap[(d.origin, stmt.source)]
        new_index = tuple(d.sigma.apply(e) for e in stmt.index)
        self.insert(Load(name=d.bound_as, source=new_src, index=new_index), d.demand_scope)

    def _resolve_splice_load(self, stmt: Load, d: _Demand, target_tag: str) -> None:
        """A Load that's a splice edge to another registered loop — emit a
        copy alias and queue the target loop's ``Write.value`` under the
        solved σ. The target's expression chain reconstructs piecemeal over
        subsequent iterations."""
        target = self.loops[target_tag]
        target_write = next((w for w, _ in target.writes if w.output == 0), None)
        if target_write is None:
            raise _NotSupported
        effective_index = tuple(d.sigma.apply(e) for e in stmt.index)
        sigma = _solve_sigma(target_write.index, effective_index, {a.name for a in target.op.axes})
        if sigma is None:
            raise _NotSupported
        v_bound = self._ensure_dep(target_write.value, target_tag, sigma, d.demand_scope)
        self.insert(Assign(name=d.bound_as, op=ElementwiseOp("copy"), args=(v_bound,)), d.demand_scope)

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
    target = sigma.get(axis.name)
    if target is None:
        return axis.name
    if isinstance(target, Var):
        return target.name
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

    def walk(stmts: tuple[Stmt, ...]) -> None:
        for s in stmts:
            if isinstance(s, Loop):
                names.add(s.axis.name)
                walk(s.body)
            elif isinstance(s, (Load, Assign, Select, Accum)):
                names.add(s.name)

    walk(op.body)
    return names
