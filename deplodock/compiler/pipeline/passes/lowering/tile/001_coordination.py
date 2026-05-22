"""Post-planner coordination for ``GridTile.splitk_axes`` /
``ThreadTile.cooperative_axes``.

The partition planner (``000_partition_planner``) constructs typed tile
flavors directly. When it commits to a split-K block axis or a
cooperative-K thread axis, the resulting tile carries that decision via
``GridTile.splitk_axes`` (subset of block-axis names that aggregate
cross-CTA via atomic) or ``ThreadTile.cooperative_axes`` (subset of
thread-axis names that aggregate cross-thread via Combine).

This pass walks each ``TileOp.body`` for those triggers and materializes
the matching coordination:

- **``splitk_axes``** (cross-CTA reduction): every ``Write`` inside the
  ``GridTile`` whose index doesn't reference the split-K axis is
  rewritten to atomic-add. Splittable ``add(indep, dep)`` Writes
  decompose into ``atomic(dep) + Cond(axis == 0, atomic(indep))`` so
  the axis-independent term lands once.
- **``cooperative_axes``** (cross-thread reduction): a ``Combine`` is
  emitted after each reduce subtree (the materializer lowers ``Combine``
  to smem tree-halve / warp-shuffle); scalar ``Write``s that don't
  reference the cooperative axis are wrapped in
  ``Cond(coop_axis == 0)`` so only one thread of the cooperative group
  writes the broadcast value.

When neither trigger fires, the pass ``RuleSkipped``s so the engine
moves on without rewriting the body.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.stmt import Accum, Assign, Cond, Write
from deplodock.compiler.ir.stmt import Stmt as LoopStmt
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.tile.ir import (
    Combine,
    GridTile,
    RegisterTile,
    SerialTile,
    Stmt,
    StridedTile,
    ThreadTile,
    TileOp,
)
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(root: Node) -> Graph | None:
    return _coordinate(root.op)


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------


def _coordinate(tile_op: TileOp) -> TileOp | None:
    """Walk ``tile_op.body`` for ``GridTile.splitk_axes`` and
    ``ThreadTile.cooperative_axes``; emit atomic-Write / Combine
    rewrites. ``RuleSkipped`` when nothing to do (idempotent on
    already-coordinated TileOps and on TileOps with no splitk / coop)."""
    if not _coordination_needed(tile_op.body):
        raise RuleSkipped("no splitk / cooperative coordination needed")

    new_body: list[Stmt] = []
    for s in tile_op.body:
        if isinstance(s, GridTile) and s.splitk_axes:
            new_inner = list(s.body)
            for axis_name in s.splitk_axes:
                new_inner = list(_rewrite_for_lifted_axis(tuple(new_inner), axis_name, mode="atomic"))
            new_grid = GridTile(axes=s.axes, body=Body(new_inner), splitk_axes=s.splitk_axes)
            new_body.append(_coordinate_thread(new_grid))
        elif isinstance(s, GridTile):
            new_body.append(_coordinate_thread(s))
        elif isinstance(s, ThreadTile):
            new_body.append(_coordinate_thread(s))
        else:
            new_body.append(s)
    return TileOp(body=tuple(new_body), name=tile_op.name, knobs=tile_op.knobs)


def _coordinate_thread(parent: GridTile | ThreadTile) -> GridTile | ThreadTile:
    """If ``parent`` (or its inner ThreadTile child) carries cooperative
    axes, emit Combine + ``Cond(coop == 0)``-guarded scalar Writes."""
    if isinstance(parent, ThreadTile):
        tt = parent
    else:
        # GridTile — find inner ThreadTile child
        tt = next((c for c in parent.body if isinstance(c, ThreadTile)), None)
        if tt is None or not tt.cooperative_axes:
            return parent
    if not tt.cooperative_axes:
        return parent
    new_inner = list(tt.body)
    for axis_name in tt.cooperative_axes:
        new_inner = list(_rewrite_for_lifted_axis(tuple(new_inner), axis_name, mode="cooperative"))
    new_tt = ThreadTile(axes=tt.axes, body=Body(new_inner), cooperative_axes=tt.cooperative_axes)
    if isinstance(parent, ThreadTile):
        return new_tt
    new_grid_body = tuple(new_tt if c is tt else c for c in parent.body)
    return GridTile(axes=parent.axes, body=Body(new_grid_body), splitk_axes=parent.splitk_axes)


def _coordination_needed(body) -> bool:
    """True iff any top-level GridTile / ThreadTile carries splitk /
    cooperative axes that require coordination rewrites."""
    for s in body:
        if isinstance(s, GridTile):
            if s.splitk_axes:
                return True
            for c in s.body:
                if isinstance(c, ThreadTile) and c.cooperative_axes:
                    return True
        elif isinstance(s, ThreadTile) and s.cooperative_axes:
            return True
    return False


# ---------------------------------------------------------------------------
# Atomic-Write rewrite for split-K block-axis lifts
# ---------------------------------------------------------------------------


def _rewrite_for_lifted_axis(body: tuple[LoopStmt, ...], axis_name: str, *, mode: str) -> tuple[LoopStmt, ...]:
    """Dispatch to atomic-Write or cooperative-Combine rewrite.

    Both modes share the trigger ("Write.index doesn't reference the
    lifted axis") and the def-DAG analysis (``_compute_axis_dep_set``).
    They differ in the primitive that materializes the coordination:

    - ``mode="atomic"`` (``splitk_axes``): descend single-stmt wrappers,
      rewrite Writes to atomic-add at the kernel level.
    - ``mode="cooperative"`` (``cooperative_axes``): walk recursively to
      find every reduce subtree, emit ``Combine`` siblings after each,
      then Cond-wrap every Write whose index doesn't reference the
      cooperative axis.
    """
    if mode == "atomic":
        return _rewrite_for_atomic_lift(body, axis_name)
    if mode == "cooperative":
        return _rewrite_for_cooperative_lift(body, axis_name)
    raise ValueError(f"unknown mode: {mode!r}")


def _rewrite_for_atomic_lift(body: tuple[LoopStmt, ...], axis_name: str) -> tuple[LoopStmt, ...]:
    """Descend through wrapper tiles until we reach the level where
    ``Write`` stmts live, then rewrite each Write whose index doesn't
    reference ``axis_name`` — those are being raced by multiple CTAs to
    the same output cell now that the axis is split-K.

    Writes whose index *does* mention ``axis_name`` write distinct cells
    across CTAs (the regular output-tile case) and pass through unchanged.
    """
    if any(isinstance(s, Write) for s in body):
        return _rewrite_kernel_writes_atomic(tuple(body), axis_name)

    changed = False
    out: list[LoopStmt] = []
    for s in body:
        if isinstance(s, (SerialTile, StridedTile, RegisterTile, ThreadTile)) and _contains_write(s):
            new_inner = _rewrite_for_atomic_lift(tuple(s.body), axis_name)
            if new_inner != tuple(s.body):
                changed = True
            out.append(s.with_bodies((Body(new_inner),)))
        elif isinstance(s, Cond) and (_contains_write(s) or any(_contains_write(c) for c in s.else_body)):
            new_b = _rewrite_for_atomic_lift(tuple(s.body), axis_name)
            new_e = _rewrite_for_atomic_lift(tuple(s.else_body), axis_name)
            if new_b != tuple(s.body) or new_e != tuple(s.else_body):
                changed = True
            out.append(Cond(cond=s.cond, body=Body(new_b), else_body=Body(new_e)))
        else:
            out.append(s)
    if not changed:
        return tuple(body)
    return tuple(out)


def _contains_write(stmt: LoopStmt) -> bool:
    """True if ``stmt`` is or transitively contains a ``Write``."""
    if isinstance(stmt, Write):
        return True
    for sub in stmt.nested():
        for c in sub:
            if _contains_write(c):
                return True
    return False


def _rewrite_kernel_writes_atomic(stmts: tuple[LoopStmt, ...], axis_name: str) -> tuple[LoopStmt, ...]:
    """At the kernel level, find every Write whose index doesn't
    reference ``axis_name`` and classify its value via def-DAG analysis
    for the atomic-add rewrite. Writes whose index *does* mention the
    axis (the typical M_b / N_b output-tile case) pass through."""
    defs: dict[str, LoopStmt] = {s.name: s for s in stmts if hasattr(s, "name")}
    axis_dep = _compute_axis_dep_set(stmts)

    out: list[LoopStmt] = []
    for s in stmts:
        if isinstance(s, Write) and not _write_indexed_by(s, axis_name):
            rewritten = _rewrite_write_atomic(s, defs, axis_dep, axis_name)
            if rewritten is not None:
                out.extend(rewritten)
                continue
        out.append(s)
    return tuple(out)


def _rewrite_write_atomic(
    write: Write,
    defs: dict[str, LoopStmt],
    axis_dep: set[str],
    axis_name: str,
) -> tuple[LoopStmt, ...] | None:
    """Classify ``write.value`` and emit the appropriate atomic-Write
    rewrite. Returns ``None`` to leave the Write alone."""
    if write.reduce_op is not None:
        return None  # already atomic (idempotence)

    value = write.value

    # Splittable: immediate producer is ``add(indep, dep)``. Check this
    # BEFORE the pure-dep path so ``v = add(r, acc)`` (which has
    # v ∈ axis_dep) gets decomposed instead of naively atomic-added
    # (which would double-count the axis-independent term).
    producer = defs.get(value)
    if isinstance(producer, Assign) and producer.op.name == "add" and len(producer.args) == 2:
        a, b = producer.args
        a_dep = a in axis_dep
        b_dep = b in axis_dep
        if a_dep != b_dep:
            indep_arg, dep_arg = (b, a) if a_dep else (a, b)
            from dataclasses import replace as dc_replace  # noqa: PLC0415

            atomic_dep = dc_replace(write, value=dep_arg, reduce_op=ElementwiseImpl("add"))
            atomic_indep = Write(
                output=write.output,
                index=write.index,
                value=indep_arg,
                reduce_op=ElementwiseImpl("add"),
            )
            cond = Cond(
                cond=BinaryExpr("==", Var(axis_name), Literal(0, "int")),
                body=Body((atomic_indep,)),
                else_body=Body(()),
            )
            return (atomic_dep, cond)

    # Pure axis-dep (plain matmul ``Write(acc)`` and mult-chain
    # ``Write(c·acc)`` both end up here): mark atomic. ``sum_i (c·a_i)
    # = c·sum_i a_i`` so atomic-add gives the correct result for any
    # linear-in-acc chain.
    if value in axis_dep:
        from dataclasses import replace as dc_replace  # noqa: PLC0415

        return (dc_replace(write, reduce_op=ElementwiseImpl("add")),)

    return None


def _write_indexed_by(write: Write, axis_name: str) -> bool:
    """True iff any index expression of the Write mentions ``axis_name``."""
    return any(axis_name in e.free_vars() for e in write.index)


def _compute_axis_dep_set(stmts: tuple[LoopStmt, ...]) -> set[str]:
    """SSA names whose value transitively touches an ``Accum`` at the
    kernel level. Since the lifted axis wraps every kernel-level stmt,
    any value that flows through an ``Accum`` is by construction
    axis-dependent (the Accum aggregates iterations of an inner loop
    that itself iterates inside the lifted axis's scope)."""
    axis_dep: set[str] = set()

    def _collect_accums(body: tuple[LoopStmt, ...]) -> None:
        for s in body:
            if isinstance(s, Accum):
                axis_dep.add(s.name)
            for child in s.nested():
                _collect_accums(tuple(child))

    _collect_accums(stmts)
    changed = True
    while changed:
        changed = False
        for s in stmts:
            if isinstance(s, Assign) and s.name not in axis_dep:
                if any(arg in axis_dep for arg in s.args):
                    axis_dep.add(s.name)
                    changed = True
    return axis_dep


# ---------------------------------------------------------------------------
# Cooperative-K Combine emission + Cond-guarded Write
# ---------------------------------------------------------------------------


def _rewrite_for_cooperative_lift(body: tuple[LoopStmt, ...], axis_name: str) -> tuple[LoopStmt, ...]:
    """Cooperative-K coordination: emit ``Combine`` after each reduce
    subtree (or bare ``stage_inner`` reduce when K_o was inlined), then
    Cond-wrap each scalar Write whose index doesn't reference the
    cooperative axis.

    Idempotence: if a ``Combine`` sibling is already present in ``body``,
    skip — re-firing would emit duplicate tree-halves.
    """
    if any(isinstance(s, Combine) for s in body):
        return tuple(body)

    new_body = _insert_combines_after_reduces(tuple(body))
    new_body = tuple(_guard_scalar_write(s, axis_name) for s in new_body)
    return new_body


def _insert_combines_after_reduces(stmts: tuple[LoopStmt, ...]) -> tuple[LoopStmt, ...]:
    """Walk ``stmts`` once, emitting one ``Combine(name, op)`` sibling
    after each cooperative-K reduce subtree. The reduce subtree shape is
    one of:

    - ``SerialTile(K_o, kind="serial_outer", body=[SerialTile(K_i,
      kind="stage_inner", reduce, [Accum])])`` — the canonical
      multi-chunk form.
    - ``SerialTile(K_i, kind="stage_inner", reduce, [Accum])`` — same
      shape but K_o collapsed to extent 1 and inlined.

    If neither shape matches at this level, descend one wrapper level
    and try again. Returns ``stmts`` unchanged if no reduce subtree
    is found anywhere reachable.
    """
    new_stmts: list[LoopStmt] = []
    emitted = False
    for s in stmts:
        new_stmts.append(s)
        accums = _accums_under_reduce_subtree(s)
        if accums:
            new_stmts.extend(Combine(name=a.name, op=a.op) for a in accums)
            emitted = True
    if emitted:
        return tuple(new_stmts)

    # No subtree at this level. Try descending one wrapper.
    for i, s in enumerate(stmts):
        if isinstance(s, (SerialTile, StridedTile, RegisterTile)):
            inner = _insert_combines_after_reduces(tuple(s.body))
            if inner != tuple(s.body):
                return stmts[:i] + (s.with_bodies((Body(inner),)),) + stmts[i + 1 :]
    return tuple(stmts)


def _accums_under_reduce_subtree(s: LoopStmt) -> list[Accum]:
    """Return the Accums of a cooperative-K reduce subtree rooted at ``s``,
    or ``[]`` if ``s`` isn't one.

    Recognized shapes:

    - **Bare Accum** at the cooperative scope. Happens when both K_o and
      K_i collapsed to size-1 (e.g. K=32 with BR=32 cooperative threads
      each handling one element) and ``drop_size_one_free_axes`` inlined
      both wrapping loops.
    - **stage_inner SerialTile** with Accums in its immediate body — the
      canonical single-chunk shape.
    - **serial_outer SerialTile wrapping stage_inner** — the canonical
      multi-chunk shape.

    Returns the immediate Accums in the innermost reduce body of the
    matched subtree."""
    if isinstance(s, Accum):
        return [s]
    if isinstance(s, SerialTile) and s.is_reduce and s.kind == "stage_inner":
        return [c for c in s.body if isinstance(c, Accum)]
    if isinstance(s, SerialTile) and not s.is_reduce and s.kind == "serial_outer":
        cur = tuple(s.body)
        while len(cur) == 1 and isinstance(cur[0], (SerialTile, StridedTile, RegisterTile)):
            inner = cur[0]
            if isinstance(inner, SerialTile) and inner.is_reduce and inner.kind == "stage_inner":
                accums = [c for c in inner.body if isinstance(c, Accum)]
                if accums:
                    return accums
            cur = tuple(inner.body)
    return []


def _guard_scalar_write(s: LoopStmt, coop_name: str) -> LoopStmt:
    """Wrap ``s`` in ``Cond(coop == 0)`` when it's a Write whose index
    doesn't reference the cooperative axis. Otherwise (per-thread Write
    or non-Write stmt) returns ``s`` unchanged. Descends into block stmts
    so multi-line post-reduce epilogues get guarded uniformly."""
    if isinstance(s, Write):
        free: set[str] = set()
        for e in s.index:
            free |= e.free_vars()
        if coop_name not in free:
            return Cond(
                cond=BinaryExpr("==", Var(coop_name), Literal(0, "int")),
                body=Body((s,)),
                else_body=Body(()),
            )
        return s
    if isinstance(s, (SerialTile, StridedTile, RegisterTile)):
        inner = tuple(_guard_scalar_write(c, coop_name) for c in s.body)
        if inner != tuple(s.body):
            return s.with_bodies((Body(inner),))
        return s
    if isinstance(s, Cond):
        b = tuple(_guard_scalar_write(c, coop_name) for c in s.body)
        e = tuple(_guard_scalar_write(c, coop_name) for c in s.else_body)
        if b != tuple(s.body) or e != tuple(s.else_body):
            return Cond(cond=s.cond, body=Body(b), else_body=Body(e))
        return s
    return s
