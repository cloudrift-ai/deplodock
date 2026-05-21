"""Form a ``TileOp`` from each ``LoopOp`` — decide launch geometry.

Mechanical translation: outer free-Loop chain becomes ``Tile.axes``,
leaves and inner Loops pass through unchanged. The planner
(``000_partition_planner``) stamps Role tags on body Loops before
launch_geometry runs; this pass just reads the tags via
:func:`_bind_for_role` and lifts tagged Loops with the appropriate
``BoundAxis.bind``: ``Role.BLOCK`` and ``Role.SPLITK_BLOCK`` →
``BIND_BLOCK``; ``Role.THREAD`` → ``BIND_THREAD``; untagged Loops
default to ``BIND_THREAD`` (safety net for kernels the planner
currently can't partition — see ``_bind_for_role``).

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
defaulting to ``BIND_THREAD``) and replaced by its body. Any free
non-reduce Loop at this level is an output-dim iterator by LoopOp
construction, so the lift gate is purely the body-resident-role
filter — no per-Write index inspection needed.

**BLOCK lifts → atomic Writes (generic).** Whenever the chain walker
lifts a Loop into ``BIND_BLOCK``, every Write inside the lifted scope
becomes potentially raced — multiple CTAs share the loop's axis.
The test is structural: if the lifted axis name appears in
``Write.index``, each CTA writes a different cell and there's no
conflict (typical output-tile axes M_b / N_b). If the axis name is
absent, the CTAs are reducing across that axis and we rewrite the
Write to commit its partial via atomic-add, decomposing
``add(indep, dep)`` shapes into
``atomic(dep) + Cond(axis == 0, atomic(indep))`` so axis-independent
contributions are added exactly once. This subsumes the special-case
SPLITK_BLOCK handling: K_s simply happens to be the only Role today
whose axis is reliably absent from Write.index by σ-construction.

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
from deplodock.compiler.ir.stmt import Accum, Assign, Cond, Loop, Write
from deplodock.compiler.ir.stmt import Stmt as LoopStmt
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.tile.ir import Stmt, Tile, TileOp
from deplodock.compiler.pipeline import Pattern

PATTERN = [Pattern("root", LoopOp)]


def _bind_for_role(role: Role | None) -> str:
    """Resolve the ``BoundAxis.bind`` for a chain-lifted Loop. Untagged
    Loops default to ``BIND_THREAD`` — exercised only for kernels the
    planner skips (currently: SDPA V matmul edge case)."""
    if role is Role.BLOCK or role is Role.SPLITK_BLOCK:
        return BIND_BLOCK
    if role is Role.THREAD:
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

    # For each BIND_BLOCK axis we just lifted, rewrite Writes inside
    # the lifted scope. The inner check ("axis in Write.index?")
    # filters out axes whose CTAs already write distinct cells (M_b,
    # N_b, etc.), so this loop is a no-op for them and only engages
    # for reduction-style lifts (K_s — but the logic doesn't know or
    # care about K_s specifically).
    for axis, bind in output_axes_with_bind:
        if bind != BIND_BLOCK:
            continue
        inner = _rewrite_writes_for_lifted_axis(inner, axis.name)

    body: list[Stmt] = list(leading)
    if output_axes_with_bind:
        bound = tuple(BoundAxis(axis=ax, bind=bind) for ax, bind in output_axes_with_bind)
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
            new_axes.append(BoundAxis(axis=s.axis, bind=_bind_for_role(s.role)))
            new_body.extend(s.body)
            changed = True
        else:
            new_body.append(s)
    if not changed:
        return tile
    return Tile(axes=tuple(new_axes), body=new_body)


def _strip_outer_free_chain(stmts: tuple[LoopStmt, ...]) -> tuple[tuple[tuple[Axis, str], ...], tuple[LoopStmt, ...]]:
    """Walk the outer free-Loop chain and return
    ``(stripped_axes_with_bind, remainder)``.

    Stops when the current level has more than one stmt, isn't a Loop, or
    is a reduce Loop, or carries a body-resident role (REGISTER /
    SERIAL_OUTER / STAGE_INNER / PIPELINE).

    Tagged Loops (BLOCK / THREAD / SPLITK_BLOCK) bind via
    :func:`_bind_for_role`. Untagged Loops default to ``BIND_THREAD``
    — a safety net for kernels the planner currently can't partition
    (e.g. SDPA V matmul where M/N detection from the chain is wrong)."""
    axes_with_bind: list[tuple[Axis, str]] = []
    cur = stmts
    while (
        len(cur) == 1
        and isinstance(cur[0], Loop)
        and not cur[0].is_reduce
        and cur[0].role not in (Role.REGISTER, Role.SERIAL_OUTER, Role.STAGE_INNER, Role.PIPELINE)
    ):
        axes_with_bind.append((cur[0].axis, _bind_for_role(cur[0].role)))
        cur = cur[0].body
    return tuple(axes_with_bind), cur


# ---------------------------------------------------------------------------
# Atomic-Write rewrite for BLOCK lifts
# ---------------------------------------------------------------------------


def _rewrite_writes_for_lifted_axis(body: tuple[LoopStmt, ...], axis_name: str) -> tuple[LoopStmt, ...]:
    """Descend through wrapper Loops (REGISTER / SERIAL_OUTER / etc.)
    until we reach the level where ``Write`` stmts live, then rewrite
    each Write whose index doesn't reference ``axis_name`` — those
    are being raced by multiple CTAs to the same output cell now that
    the axis is BIND_BLOCK.

    Writes whose index *does* mention ``axis_name`` write distinct
    cells across CTAs (the regular output-tile case) and pass through
    unchanged."""
    wrappers: list[Loop] = []
    cur: tuple[LoopStmt, ...] = tuple(body)
    while not any(isinstance(s, Write) for s in cur):
        if len(cur) != 1 or not isinstance(cur[0], Loop):
            return tuple(body)  # can't locate the kernel level
        wrappers.append(cur[0])
        cur = tuple(cur[0].body)

    new_kernel_stmts = _rewrite_kernel_writes(cur, axis_name)

    current: tuple[LoopStmt, ...] = new_kernel_stmts
    for w in reversed(wrappers):
        current = (dc_replace(w, body=current),)
    return current


def _rewrite_kernel_writes(stmts: tuple[LoopStmt, ...], axis_name: str) -> tuple[LoopStmt, ...]:
    """At the kernel level, find every Write whose index doesn't
    reference ``axis_name`` and classify its value via def-DAG analysis
    for the atomic-add rewrite. Writes whose index *does* mention the
    axis (the typical M_b / N_b output-tile case) pass through."""
    defs: dict[str, LoopStmt] = {s.name: s for s in stmts if hasattr(s, "name")}
    axis_dep = _compute_axis_dep_set(stmts)

    out: list[LoopStmt] = []
    for s in stmts:
        if isinstance(s, Write) and not _write_indexed_by(s, axis_name):
            rewritten = _rewrite_write_for_lifted_axis(s, defs, axis_dep, axis_name)
            if rewritten is not None:
                out.extend(rewritten)
                continue
        out.append(s)
    return tuple(out)


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
    for s in stmts:
        if isinstance(s, Accum):
            axis_dep.add(s.name)
    changed = True
    while changed:
        changed = False
        for s in stmts:
            if isinstance(s, Assign) and s.name not in axis_dep:
                if any(arg in axis_dep for arg in s.args):
                    axis_dep.add(s.name)
                    changed = True
    return axis_dep


def _rewrite_write_for_lifted_axis(
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
