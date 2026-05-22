"""Form a ``TileOp`` from each ``LoopOp`` — decide launch geometry,
materialize post-lift coordination primitives.

Mechanical translation: outer free-Loop chain becomes ``Tile.axes``,
leaves and inner Loops pass through unchanged. The planner
(``000_partition_planner``) stamps Role tags on body Loops before
launch_geometry runs; this pass just reads the tags via
:func:`_bind_for_role` and lifts tagged Loops with the appropriate
``BoundAxis.bind``: ``Role.BLOCK`` and ``Role.SPLITK_BLOCK`` →
``BIND_BLOCK``; ``Role.THREAD`` and ``Role.COOPERATIVE_STRIDE`` →
``BIND_THREAD``; untagged Loops default to ``BIND_THREAD`` (safety net
for kernels the planner currently can't partition).

**Outer free-Loop chain → ``Tile.axes``**. After stripping leading
non-Loop stmts (scalar Loads) into the TileOp body prefix, the chain
walker descends single-stmt nests and lifts every Loop whose role
isn't body-resident (REGISTER / SERIAL_OUTER / STAGE_INNER / PIPELINE).
The chain ends at: a level with multiple sibling stmts, a reduce
Loop, a body-resident-role Loop, or no Loop at all.

**Body free Loops over output dims → ``Tile.axes``**. After the
outer chain is stripped, top-level body stmts may still contain free
Loops whose iteration writes distinct output positions (e.g. fused
SDPA's head-dim loop sits as a sibling to two softmax reduces). Each
such Loop is lifted into ``Tile.axes`` (bind resolved from its role,
defaulting to ``BIND_THREAD``) and replaced by its body.

**Post-lift coordination — generic.** Whenever a Loop is lifted to a
parallel coord (BIND_BLOCK or BIND_THREAD with COOPERATIVE_STRIDE),
every Write inside the lifted scope whose index doesn't reference the
lifted axis becomes potentially raced — multiple coords share an output
cell. The trigger check (``Write.index`` doesn't reference the axis) is
identical for both modes; the primitive that materializes the
coordination differs:

- **BIND_BLOCK** (cross-CTA): atomic-add Write, decomposing
  ``add(indep, dep)`` into ``atomic(dep) + Cond(axis == 0, atomic(indep))``.
  Subsumes the special-case SPLITK_BLOCK handling.
- **COOPERATIVE_STRIDE** (cross-thread): emit ``Combine`` siblings
  after each reduce subtree (smem tree-halve / warp-shuffle at
  materialize time), then ``Cond(axis == 0, [Write])``-wrap the Write.
  Post-Combine the broadcast value is uniform across the cooperative
  group, so no per-term decomposition is needed.

The node's id, inputs, and output tensor are preserved — only the op
changes.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis, Role
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, Cond, Loop, StridedLoop, Write
from deplodock.compiler.ir.stmt import Stmt as LoopStmt
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.tile.ir import Combine, Stmt, Tile, TileOp
from deplodock.compiler.pipeline import Pattern

PATTERN = [Pattern("root", LoopOp)]


def _bind_for_role(role: Role | None) -> str:
    """Resolve the ``BoundAxis.bind`` for a chain-lifted Loop. Untagged
    Loops default to ``BIND_THREAD`` — exercised only for kernels the
    planner skips (currently: SDPA V matmul edge case).

    ``Role.COOPERATIVE_STRIDE`` (cooperative-K thread axis K_c) binds to
    ``BIND_THREAD`` — the same as a regular thread axis. The
    post-lift rewrite (``_rewrite_for_lifted_axis`` with
    ``mode="cooperative"``) reads the role off the BoundAxis to emit
    ``Combine`` after each reduce subtree.
    """
    if role is Role.BLOCK or role is Role.SPLITK_BLOCK:
        return BIND_BLOCK
    if role is Role.THREAD or role is Role.COOPERATIVE_STRIDE:
        return BIND_THREAD
    return BIND_THREAD


def rewrite(root: Node) -> Graph | None:
    kname = _kernel_name_for(root.op, root.id)
    return launch_geometry(root.op, kname)


def launch_geometry(loop_op: LoopOp, kernel_name: str = "") -> TileOp:
    """Translate a ``LoopOp`` into a ``TileOp`` holding a logical ``Tile``.

    Steps:

    1. Pull leading non-Loop stmts (typically scalar Loads) off the LoopOp
       body — they sit at the start of ``TileOp.body``, above any Tile.
    2. Descend the outer free-Loop chain, collecting ``(axis, bind)`` pairs
       (bind resolved from each Loop's role via :func:`_bind_for_role`)
       until the chain breaks (multi-stmt level, reduce Loop, body-
       resident role, or no more Loops).
    3. If any axes were collected, wrap the remaining inner body in a
       ``Tile`` with those ``BoundAxis``es. Otherwise, lower the inner
       body in place (single-thread serial — degenerate).

    Inner ``Loop``s pass through unchanged.
    """
    leading: list[LoopStmt] = []
    rest: tuple[LoopStmt, ...] = loop_op.body
    while rest and not isinstance(rest[0], Loop):
        leading.append(rest[0])
        rest = rest[1:]

    output_axes_with_bind, inner = _strip_outer_free_chain(rest)

    # Post-lift coordination. The inner check ("axis in Write.index?")
    # filters out axes whose coords already write distinct cells (M_b,
    # N_b, etc.) — for those, this loop is a no-op. It engages when the
    # lifted axis is absent from Write.index, meaning multiple coords
    # share an output cell:
    #   - BIND_BLOCK: rewrite Write to atomic-add (cross-CTA reduce).
    #   - COOPERATIVE_STRIDE: emit Combine after each reduce subtree
    #     and Cond(axis == 0)-wrap the scalar Write (cross-thread
    #     reduce via smem tree-halve / warp-shuffle).
    for axis, bind, role in output_axes_with_bind:
        if bind == BIND_BLOCK:
            inner = _rewrite_for_lifted_axis(inner, axis.name, mode="atomic")
        elif role is Role.COOPERATIVE_STRIDE:
            inner = _rewrite_for_lifted_axis(inner, axis.name, mode="cooperative")

    body: list[Stmt] = list(leading)
    if output_axes_with_bind:
        bound = tuple(BoundAxis(axis=ax, bind=bind, role=role) for ax, bind, role in output_axes_with_bind)
        body.append(_lift_output_loops(Tile(axes=bound, body=inner)))
    else:
        body.extend(inner)

    return TileOp(body=body, name=kernel_name)


def _lift_output_loops(tile: Tile) -> Tile:
    """Lift top-level free Loops in ``tile.body`` into ``Tile.axes``.

    Picks up the case where the chain walker stopped early (multi-stmt
    level, e.g. fused SDPA's head-dim Loop sitting as a sibling to two
    softmax reduces). Tagged Loops bind via :func:`_bind_for_role`;
    untagged Loops default to ``BIND_THREAD``. Body-resident roles
    (``REGISTER`` / ``SERIAL_OUTER`` / ``STAGE_INNER`` / ``PIPELINE``)
    are skipped — they're handled in body lowering, not as launch dims.

    The body-resident role check is the only filter — any free
    non-reduce Loop reaching this point at the Tile-body top level is
    an output-dim iterator by LoopOp construction (LoopOps come from
    graph nodes; non-reduce Loops iterate the output's shape dims).
    """
    new_axes = list(tile.axes)
    new_body: list[Stmt] = []
    changed = False
    for s in tile.body:
        if isinstance(s, Loop) and not s.is_reduce and s.role not in (Role.REGISTER, Role.SERIAL_OUTER, Role.STAGE_INNER, Role.PIPELINE):
            new_axes.append(BoundAxis(axis=s.axis, bind=_bind_for_role(s.role), role=s.role))
            new_body.extend(s.body)
            changed = True
        else:
            new_body.append(s)
    if not changed:
        return tile
    return Tile(axes=tuple(new_axes), body=new_body)


def _strip_outer_free_chain(
    stmts: tuple[LoopStmt, ...],
) -> tuple[tuple[tuple[Axis, str, Role | None], ...], tuple[LoopStmt, ...]]:
    """Walk the outer free-Loop chain and return
    ``(stripped_axes_with_bind_and_role, remainder)``.

    Stops when the current level has more than one stmt, isn't a Loop, or
    is a reduce Loop, or carries a body-resident role (REGISTER /
    SERIAL_OUTER / STAGE_INNER / PIPELINE).

    Tagged Loops (BLOCK / THREAD / SPLITK_BLOCK / COOPERATIVE_STRIDE) bind
    via :func:`_bind_for_role`. Untagged Loops default to ``BIND_THREAD``
    — a safety net for kernels the planner currently can't partition
    (e.g. SDPA V matmul where M/N detection from the chain is wrong).

    The third tuple element preserves each Loop's role so the caller can
    forward it to ``BoundAxis.role`` for downstream consumption."""
    axes_with_bind: list[tuple[Axis, str, Role | None]] = []
    cur = stmts
    while (
        len(cur) == 1
        and isinstance(cur[0], Loop)
        and not cur[0].is_reduce
        and cur[0].role not in (Role.REGISTER, Role.SERIAL_OUTER, Role.STAGE_INNER, Role.PIPELINE)
    ):
        axes_with_bind.append((cur[0].axis, _bind_for_role(cur[0].role), cur[0].role))
        cur = cur[0].body
    return tuple(axes_with_bind), cur


# ---------------------------------------------------------------------------
# Post-lift coordination
# ---------------------------------------------------------------------------


def _rewrite_for_lifted_axis(body: tuple[LoopStmt, ...], axis_name: str, *, mode: str) -> tuple[LoopStmt, ...]:
    """Materialize cross-coord coordination for a lifted parallel axis.

    Both modes share the trigger ("Write.index doesn't reference the
    lifted axis") and the def-DAG analysis (``_compute_axis_dep_set``).
    They differ in the primitive that materializes the coordination:

    - ``mode="atomic"`` (BIND_BLOCK lift): descend single-stmt
      wrappers, rewrite Writes to atomic-add at the kernel level.
    - ``mode="cooperative"`` (COOPERATIVE_STRIDE lift): walk recursively
      to find every reduce subtree, emit ``Combine`` siblings after
      each, then Cond-wrap every Write whose index doesn't reference
      the cooperative axis.

    Both rewrites are scoped to ``body`` (typically the ``Tile.body``
    just after launch_geometry stripped the outer chain)."""
    if mode == "atomic":
        return _rewrite_for_atomic_lift(body, axis_name)
    if mode == "cooperative":
        return _rewrite_for_cooperative_lift(body, axis_name)
    raise ValueError(f"unknown mode: {mode!r}")


# --- Atomic mode (BIND_BLOCK lifts) ----------------------------------


def _rewrite_for_atomic_lift(body: tuple[LoopStmt, ...], axis_name: str) -> tuple[LoopStmt, ...]:
    """Descend through wrapper Loops until we reach the level where
    ``Write`` stmts live, then rewrite each Write whose index doesn't
    reference ``axis_name`` — those are being raced by multiple CTAs
    to the same output cell now that the axis is BIND_BLOCK.

    Writes whose index *does* mention ``axis_name`` write distinct
    cells across CTAs (the regular output-tile case) and pass through
    unchanged.

    Multi-stmt scopes (e.g. SDPA kernel 1 — sibling Init / reduce-Loop
    / output-Loop at top of Tile.body) are handled by recursing into
    every nested Loop / StridedLoop / Cond that contains a Write; the
    rewrite happens at the deepest level that still has Writes as
    immediate siblings of any post-reduce Accum-dep stmts.
    """
    if any(isinstance(s, Write) for s in body):
        return _rewrite_kernel_writes_atomic(tuple(body), axis_name)

    changed = False
    out: list[LoopStmt] = []
    for s in body:
        if isinstance(s, (Loop, StridedLoop)) and _contains_write(s):
            new_inner = _rewrite_for_atomic_lift(tuple(s.body), axis_name)
            if new_inner != tuple(s.body):
                changed = True
            out.append(dc_replace(s, body=new_inner))
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
    # ``Write(c·acc)`` both end up here): mark atomic.
    # ``sum_i (c·a_i) = c·sum_i a_i`` so atomic-add gives the correct
    # result for any linear-in-acc chain.
    if value in axis_dep:
        return (dc_replace(write, reduce_op=ElementwiseImpl("add")),)

    return None


# --- Shared dataflow analysis ---------------------------------------


def _write_indexed_by(write: Write, axis_name: str) -> bool:
    """True iff any index expression of the Write mentions ``axis_name``."""
    return any(axis_name in e.free_vars() for e in write.index)


def _compute_axis_dep_set(stmts: tuple[LoopStmt, ...]) -> set[str]:
    """SSA names whose value transitively touches an ``Accum`` at the
    kernel level. Since the lifted axis wraps every kernel-level stmt,
    any value that flows through an ``Accum`` is by construction
    axis-dependent (the Accum aggregates iterations of an inner loop
    that itself iterates inside the lifted axis's scope).

    Accums sit inside reduce Loops (one or more nesting levels below
    the kernel-level Writes), so collect them by descending through
    every nested ``Loop`` / ``StridedLoop`` / ``Cond`` body — the
    aggregated value is still visible at the kernel scope through the
    Accum's SSA name."""
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


# --- Cooperative mode (COOPERATIVE_STRIDE lifts) --------------------


def _rewrite_for_cooperative_lift(body: tuple[LoopStmt, ...], axis_name: str) -> tuple[LoopStmt, ...]:
    """Cooperative-K coordination: emit ``Combine`` after each K_o reduce
    subtree (or bare STAGE_INNER reduce when K_o was inlined), then
    Cond-wrap each scalar Write whose index doesn't reference the
    cooperative axis.

    Idempotence: if a ``Combine`` sibling is already present in
    ``body``, skip — re-firing would emit duplicate tree-halves.
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

    - ``Loop(K_o, SERIAL_OUTER, Loop(K_i, STAGE_INNER, reduce, [Accum]))``
      — the canonical multi-chunk form.
    - ``Loop(K_i, STAGE_INNER, reduce, [Accum])`` — same shape but K_o
      collapsed to extent 1 and inlined by ``drop_size_one_free_axes``.

    If neither shape matches at this level, descend one wrapper level
    and try again (handles cases where an output THREAD layer didn't
    fully collapse). Returns ``stmts`` unchanged if no reduce subtree
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
        if isinstance(s, (Loop, StridedLoop)):
            inner = _insert_combines_after_reduces(tuple(s.body))
            if inner != tuple(s.body):
                return stmts[:i] + (dc_replace(s, body=inner),) + stmts[i + 1 :]
    return tuple(stmts)


def _accums_under_reduce_subtree(s: LoopStmt) -> list[Accum]:
    """If ``s`` is a cooperative-K reduce subtree (SERIAL_OUTER wrapping
    STAGE_INNER, or bare STAGE_INNER), return the immediate Accums of
    the STAGE_INNER. Otherwise return an empty list."""
    if isinstance(s, Loop) and s.is_reduce and s.role is Role.STAGE_INNER:
        return [c for c in s.body if isinstance(c, Accum)]
    if isinstance(s, Loop) and not s.is_reduce and s.role is Role.SERIAL_OUTER:
        cur = tuple(s.body)
        while len(cur) == 1 and isinstance(cur[0], (Loop, StridedLoop)):
            inner = cur[0]
            if inner.is_reduce and inner.role is Role.STAGE_INNER:
                accums = [c for c in inner.body if isinstance(c, Accum)]
                if accums:
                    return accums
            cur = tuple(inner.body)
    return []


def _guard_scalar_write(s: LoopStmt, coop_name: str) -> LoopStmt:
    """Wrap ``s`` in ``Cond(coop == 0)`` when it's a Write whose index
    doesn't reference the cooperative axis. Otherwise (per-thread Write
    or non-Write stmt) returns ``s`` unchanged. Descends into block
    stmts so multi-line post-reduce epilogues get guarded uniformly."""
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
    if isinstance(s, (Loop, StridedLoop)):
        inner = tuple(_guard_scalar_write(c, coop_name) for c in s.body)
        if inner != tuple(s.body):
            return dc_replace(s, body=inner)
        return s
    if isinstance(s, Cond):
        b = tuple(_guard_scalar_write(c, coop_name) for c in s.body)
        e = tuple(_guard_scalar_write(c, coop_name) for c in s.else_body)
        if b != tuple(s.body) or e != tuple(s.else_body):
            return Cond(cond=s.cond, body=Body(b), else_body=Body(e))
        return s
    return s


# ---------------------------------------------------------------------------
# Kernel naming
# ---------------------------------------------------------------------------


def _kernel_name_for(loop: LoopOp, base_name: str) -> str:
    suffix = "reduce" if any(isinstance(s, Accum) for s in loop) else "pointwise"
    return f"k_{_dedup_tokens(base_name)}_{suffix}"


def _dedup_tokens(name: str) -> str:
    """Drop consecutive duplicate ``_``-separated tokens.

    ``softmax_softmax_max`` → ``softmax_max``; ``rms_rms_norm`` → ``rms_norm``.
    Preserves order; only collapses adjacent duplicates so structurally
    distinct repeats (``add_mul_add``) survive.
    """
    out: list[str] = []
    for tok in name.split("_"):
        if not tok or (out and out[-1] == tok):
            continue
        out.append(tok)
    return "_".join(out) if out else name
