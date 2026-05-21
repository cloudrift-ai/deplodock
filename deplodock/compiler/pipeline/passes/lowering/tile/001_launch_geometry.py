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
defaulting to ``BIND_THREAD``) and replaced by its body. Detection:
top-level body stmts only; the loop's subtree must contain a ``Write``
whose index expression references the loop's axis.

**SPLITK_BLOCK → atomic Writes.** When the chain walker lifts a Loop
with ``Role.SPLITK_BLOCK`` into ``BIND_BLOCK``, the body inside is
now executed by multiple CTAs racing toward the same output cells.
We rewrite every Write in that body so it commits its partial via
atomic-add, decomposing ``add(K_s-indep, K_s-dep)`` shapes into
``atomic(dep) + Cond(K_s == 0, atomic(indep))`` so K_s-independent
contributions are added exactly once. The K_s-dep classification is
a def-DAG walk: a value is K_s-dep iff its def chain reaches an
``Accum`` inside the lifted scope.

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

    output_axes_with_bind, k_s_name, inner = _strip_outer_free_chain(rest)

    # If a SPLITK_BLOCK axis got lifted (the chain walker remembered
    # its name), the body inside is now multi-CTA. Rewrite Writes so
    # CTAs can safely commit their partials via atomic-add.
    if k_s_name is not None:
        inner = _rewrite_writes_for_splitk(inner, k_s_name)

    body: list[Stmt] = list(leading)
    if output_axes_with_bind:
        bound = tuple(BoundAxis(axis=ax, bind=bind) for ax, bind in output_axes_with_bind)
        body.append(_lift_output_loops(Tile(axes=bound, body=inner)))
    else:
        body.extend(inner)

    return TileOp(body=body, name=kernel_name)


def _lift_output_loops(tile: Tile) -> Tile:
    """Lift top-level free Loops that wrap a Write whose index varies
    with the loop's axis into ``Tile.axes``. Tagged Loops bind via
    :func:`_bind_for_role`; untagged Loops default to ``BIND_THREAD``.

    REGISTER / SERIAL_OUTER / STAGE_INNER tags are skipped — they
    belong in the body."""
    new_axes = list(tile.axes)
    new_body: list[Stmt] = []
    changed = False
    for s in tile.body:
        if (
            isinstance(s, Loop)
            and not s.is_reduce
            and s.role not in (Role.REGISTER, Role.SERIAL_OUTER, Role.STAGE_INNER, Role.PIPELINE)
            and _writes_with_axis(s.body, s.axis.name)
        ):
            new_axes.append(BoundAxis(axis=s.axis, bind=_bind_for_role(s.role)))
            new_body.extend(s.body)
            changed = True
        else:
            new_body.append(s)
    if not changed:
        return tile
    return Tile(axes=tuple(new_axes), body=new_body)


def _writes_with_axis(stmts: tuple, axis_name: str) -> bool:
    for s in stmts:
        if isinstance(s, Write):
            free: set[str] = set()
            for e in s.index:
                free |= e.free_vars()
            if axis_name in free:
                return True
        if isinstance(s, (Loop, StridedLoop)) and _writes_with_axis(s.body, axis_name):
            return True
        if isinstance(s, Cond):
            if _writes_with_axis(s.body, axis_name) or _writes_with_axis(s.else_body, axis_name):
                return True
    return False


def _strip_outer_free_chain(
    stmts: tuple[LoopStmt, ...],
) -> tuple[tuple[tuple[Axis, str], ...], str | None, tuple[LoopStmt, ...]]:
    """Walk the outer free-Loop chain and return
    ``(stripped_axes_with_bind, splitk_axis_name, remainder)``.

    Stops when the current level has more than one stmt, isn't a Loop, or
    is a reduce Loop, or carries a body-resident role (REGISTER /
    SERIAL_OUTER / STAGE_INNER / PIPELINE).

    Tagged Loops (BLOCK / THREAD / SPLITK_BLOCK) bind via
    :func:`_bind_for_role`. Untagged Loops default to ``BIND_THREAD``
    — a safety net for kernels the planner currently can't partition
    (e.g. SDPA V matmul where M/N detection from the chain is wrong).

    The middle return value is the axis name of a lifted SPLITK_BLOCK
    Loop, or ``None`` if no SPLITK was in the chain — used by the
    caller to drive the atomic-Write rewrite for the body inside."""
    axes_with_bind: list[tuple[Axis, str]] = []
    splitk_name: str | None = None
    cur = stmts
    while (
        len(cur) == 1
        and isinstance(cur[0], Loop)
        and not cur[0].is_reduce
        and cur[0].role not in (Role.REGISTER, Role.SERIAL_OUTER, Role.STAGE_INNER, Role.PIPELINE)
    ):
        axes_with_bind.append((cur[0].axis, _bind_for_role(cur[0].role)))
        if cur[0].role is Role.SPLITK_BLOCK:
            splitk_name = cur[0].axis.name
        cur = cur[0].body
    return tuple(axes_with_bind), splitk_name, cur


# ---------------------------------------------------------------------------
# Atomic-Write rewrite for SPLITK_BLOCK lifting
# ---------------------------------------------------------------------------


def _rewrite_writes_for_splitk(body: tuple[LoopStmt, ...], k_s_name: str) -> tuple[LoopStmt, ...]:
    """Descend through wrapper Loops (REGISTER / etc.) until we reach
    the level where ``Write`` stmts live, then rewrite each Write so
    multiple CTAs can safely commit their partials via atomic-add.

    The rewrite is driven by a def-DAG classification of each Write's
    value into ``K_s-dep`` (reaches an Accum inside the K_s scope) vs
    ``K_s-indep`` components:

    - Pure ``K_s-dep`` value → mark Write atomic.
    - ``add(K_s-indep, K_s-dep)`` immediate producer → emit
      ``atomic(dep) + Cond(K_s == 0, atomic(indep))``.
    - Otherwise → leave the Write alone (the kernel may produce wrong
      output if SPLITK > 1; the planner enumerates SPLITK = 1 too so a
      safe variant exists in the cartesian).
    """
    wrappers: list[Loop] = []
    cur: tuple[LoopStmt, ...] = tuple(body)
    while not any(isinstance(s, Write) for s in cur):
        if len(cur) != 1 or not isinstance(cur[0], Loop):
            return tuple(body)  # can't locate the kernel level
        wrappers.append(cur[0])
        cur = tuple(cur[0].body)

    new_kernel_stmts = _rewrite_kernel_writes(cur, k_s_name)

    current: tuple[LoopStmt, ...] = new_kernel_stmts
    for w in reversed(wrappers):
        current = (dc_replace(w, body=current),)
    return current


def _rewrite_kernel_writes(stmts: tuple[LoopStmt, ...], k_s_name: str) -> tuple[LoopStmt, ...]:
    """At the kernel level (where Write and any Accum / K_o / Assign
    siblings live), classify each Write's value via def-DAG and
    rewrite for split-K correctness."""
    defs: dict[str, LoopStmt] = {s.name: s for s in stmts if hasattr(s, "name")}
    k_dep = _compute_k_dep_set(stmts)

    out: list[LoopStmt] = []
    for s in stmts:
        if isinstance(s, Write):
            rewritten = _rewrite_write_for_splitk(s, defs, k_dep, k_s_name)
            if rewritten is not None:
                out.extend(rewritten)
                continue
        out.append(s)
    return tuple(out)


def _compute_k_dep_set(stmts: tuple[LoopStmt, ...]) -> set[str]:
    """SSA names whose value transitively touches an ``Accum`` inside
    these kernel-level stmts (Accums live inside the K_s scope, so
    "touches Accum" ≡ "K_s-dependent")."""
    k_dep: set[str] = set()
    for s in stmts:
        if isinstance(s, Accum):
            k_dep.add(s.name)
    changed = True
    while changed:
        changed = False
        for s in stmts:
            if isinstance(s, Assign) and s.name not in k_dep:
                if any(arg in k_dep for arg in s.args):
                    k_dep.add(s.name)
                    changed = True
    return k_dep


def _rewrite_write_for_splitk(
    write: Write,
    defs: dict[str, LoopStmt],
    k_dep: set[str],
    k_s_name: str,
) -> tuple[LoopStmt, ...] | None:
    """Classify ``write.value`` and emit the appropriate atomic-Write
    rewrite. Returns ``None`` to leave the Write alone."""
    if write.reduce_op is not None:
        return None  # already atomic (idempotence)

    value = write.value

    # Splittable: immediate producer is ``add(K_s-indep, K_s-dep)``.
    # Check this BEFORE the pure-dep path so ``v = add(r, acc)`` (which
    # has v ∈ k_dep) gets decomposed instead of naively atomic-added
    # (which would double-count the K_s-independent term).
    producer = defs.get(value)
    if isinstance(producer, Assign) and producer.op.name == "add" and len(producer.args) == 2:
        a, b = producer.args
        a_dep = a in k_dep
        b_dep = b in k_dep
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
                cond=BinaryExpr("==", Var(k_s_name), Literal(0, "int")),
                body=Body((atomic_indep,)),
                else_body=Body(()),
            )
            return (atomic_dep, cond)

    # Pure K_s-dep (plain matmul ``Write(acc)`` and mult-chain ``Write(c·acc)``
    # both end up here): mark atomic. ``sum_k (c·a_k) = c·sum_k a_k`` so
    # marking the final Write atomic-add gives the correct result.
    if value in k_dep:
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
