"""Linear-residual epilogue analysis + gate rewrite for SPLITK > 1 matmul.

Under SPLITK > 1 the K-reduce splits across ``K_s`` CTAs that each atomic-
add their partial Accum into the output. For a ``matmul_add``-shape post-
reduce epilogue ``Load(r) + Assign(v=acc+r) + Write(out, v)`` this would
atomic-add the residual ``K_s`` times instead of once. The fix is the
``K_s == 0`` gate: rewrite the epilogue under ``Cond(K_s == 0, body,
else=[Write(out, acc)])`` so only the K_s == 0 CTA contributes the
residual while every other CTA atomic-adds the bare Accum —
``sum_i acc_i + r = c · sum_k a_k + r``.

Two consumers:

- ``010_partition_loops`` calls :func:`has_nonlinear_post_reduce_epilogue`
  in ``_plan_kernel`` to compute ``force_splitk_one`` — non-linear /
  multi-Write / vector-Write epilogues force SPLITK = 1 at enumeration
  time, so the gate never sees them.
- ``015_gate_splitk_residual`` calls :func:`gate_linear_epilogue_on_k_s_zero`
  as a post-planner pass to apply the rewrite on the wrapped TileOp body
  for the variants where SPLITK > 1 actually fires.

Kept as a non-pass sibling module (``_`` prefix → Pass loader skips) so
both files import from one source of truth without going through
``importlib.import_module`` cross-pass dances.
"""

from __future__ import annotations

from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Cond, Stmt, Write
from deplodock.compiler.ir.tile.ir import RegisterTile


def stmt_contains_accum(stmt: Stmt) -> bool:
    """True if ``stmt`` is or transitively contains an ``Accum``."""
    if isinstance(stmt, Accum):
        return True
    for body in stmt.nested():
        for s in body:
            if stmt_contains_accum(s):
                return True
    return False


def find_first_accum_name(stmts: tuple[Stmt, ...]) -> str | None:
    """Return the first Accum's ``name`` found anywhere in ``stmts``, or
    ``None`` if no Accum exists. Used by the SPLITK matmul-add lowering to
    spell the else-branch ``Write(out, acc_name)``."""
    for s in stmts:
        if isinstance(s, Accum):
            return s.name
        for body in s.nested():
            n = find_first_accum_name(tuple(body))
            if n is not None:
                return n
    return None


def is_linear_in_accum(value_name: str, acc_name: str, assigns_by_name: dict[str, Assign]) -> bool:
    """True iff ``value_name`` is computed from ``acc_name`` via a chain of
    ``add`` Assigns (any number of external Load arguments allowed).

    Recurses through the SSA chain of Assigns that produces ``value_name``.
    At each ``Assign``, requires ``op == "add"`` and that at least one arg
    transitively traces back to ``acc_name``. Bare Var args that are not in
    ``assigns_by_name`` and not ``acc_name`` are treated as external loads
    (residuals) — allowed under linearity. A non-``add`` Assign or a chain
    that never reaches ``acc_name`` returns False."""
    if value_name == acc_name:
        return True
    a = assigns_by_name.get(value_name)
    if a is None:
        return False
    if a.op.name != "add":
        return False
    return any(arg == acc_name or is_linear_in_accum(arg, acc_name, assigns_by_name) for arg in a.args)


def has_nonlinear_post_reduce_epilogue(stmts: tuple[Stmt, ...]) -> bool:
    """True iff ``stmts`` has any post-reduce epilogue that is NOT a linear
    add chain over the Accum.

    Used by the partition planner to decide ``force_splitk_one`` for
    matmul-shape kernels: linear epilogues (``matmul_add``) are handled by
    :func:`gate_linear_epilogue_on_k_s_zero` post-planner; non-linear ones
    (e.g. ``v = acc * r``) must run at SPLITK = 1."""
    epilogue = tuple(s for s in stmts if not stmt_contains_accum(s))
    if not epilogue:
        return False
    writes = [s for s in epilogue if isinstance(s, Write)]
    if len(writes) != 1:
        return True
    write = writes[0]
    if write.is_vector:
        return True
    acc_name = find_first_accum_name(stmts)
    if acc_name is None:
        return True
    assigns_by_name = {s.name: s for s in epilogue if isinstance(s, Assign)}
    return not is_linear_in_accum(write.value, acc_name, assigns_by_name)


def gate_linear_epilogue_on_k_s_zero(stmts: tuple[Stmt, ...], k_s_name: str) -> tuple[Stmt, ...]:
    """Rewrite ``stmts`` so the linear post-reduce epilogue runs only on the
    ``K_s == 0`` CTA, with the other CTAs atomic-adding the bare Accum.

    Input shape (per-cell, matmul_add after K-loop replacement)::

        [Load(r), <reduce tower>, Assign(v=acc+r), Write(out, v)]

    Output shape::

        [<reduce tower>,
         Cond(K_s == 0,
              body=[Load(r), Assign(v=acc+r), Write(out, v)],
              else_body=[Write(out, acc)])]

    Blocked-builder input shape — the matmul_add epilogue lives inside the
    Write-tower ``RegisterTile(N_r)``, separated from the reduce tower by
    the planner's three-tower layout::

        [RegisterTile(N_r, [Init]),
         <reduce tower with Accum nested in RegisterTile(N_r)>,
         RegisterTile(N_r, [Load(r), Assign(v=acc+r), Write(out, v)])]

    Output shape (the Cond moves INSIDE the Write tower)::

        [RegisterTile(N_r, [Init]),
         <reduce tower>,
         RegisterTile(N_r,
             [Cond(K_s == 0,
                   body=[Load(r), Assign(v=acc+r), Write(out, v)],
                   else_body=[Write(out, acc)])])]

    The kernel-IR replicator (``010_split_register_axes``) then unrolls
    each ``RegisterTile(N_r)`` per cell — the Cond's predicate is K_s-only
    and replicates verbatim, the body/else_body Loads / Assigns / Writes
    pick up the per-cell SSA suffix as usual.

    Both Writes lower to ``atomicAdd`` under SPLITK > 1, so the final
    output is ``sum_i acc_i + r`` — the residual added exactly once.
    Returns ``stmts`` unchanged when no linear epilogue is present (also:
    the rewrite is idempotent — after firing once the epilogue is inside
    a Cond, so re-applying finds no epilogue Write at the gating scope).

    **Partition is positional, not set-based.** When the K-loop is fully
    unrolled (BK = 1 + K_o_ext = 1, i.e. ``_wrap_tower`` drops both K_o and
    K_i as size-1) the Loads / Assigns that *feed* the Accum end up as
    siblings of the Accum at this level. Those are reduce-body stmts, not
    epilogue — they must stay with the Accum, not get moved into the Cond
    (which would leave the Accum referencing values defined inside a scope
    it no longer dominates). Split at the position of the last Accum-
    bearing stmt: ``[:cut+1]`` is the reduce body that the Cond skips
    over; ``[cut+1:]`` is the true post-reduce epilogue.
    """
    last_accum_idx = -1
    for i, s in enumerate(stmts):
        if stmt_contains_accum(s):
            last_accum_idx = i
    if last_accum_idx < 0:
        return stmts
    reduce_part = stmts[: last_accum_idx + 1]
    epilogue = stmts[last_accum_idx + 1 :]
    if not epilogue:
        return stmts
    acc_name = find_first_accum_name(stmts)
    if acc_name is None:
        return stmts

    # Blocked path: the epilogue is a single ``RegisterTile(N_r, [Load,
    # Assign, Write])`` (the planner's Write tower). Recurse into the
    # tower body and emit the Cond there; the surrounding RegisterTile
    # stays untouched so the replicator can still unroll it per cell.
    if len(epilogue) == 1 and isinstance(epilogue[0], RegisterTile):
        rt = epilogue[0]
        gated = _gate_linear_chain(tuple(rt.body), acc_name, k_s_name)
        if gated is None:
            return stmts
        return reduce_part + (RegisterTile(axes=rt.axes, body=Body(gated)),)

    # Per-cell flat epilogue.
    gated = _gate_linear_chain(epilogue, acc_name, k_s_name)
    if gated is None:
        return stmts
    return reduce_part + gated


def _gate_linear_chain(stmts: tuple[Stmt, ...], acc_name: str, k_s_name: str) -> tuple[Stmt, ...] | None:
    """Apply the K_s == 0 ``Cond`` wrap to a flat ``[Load*, Assign*, Write]``
    chain that's structurally linear in ``acc_name``.

    Returns ``(cond,)`` — a single-element tuple holding the wrapped
    chain — or ``None`` if the chain doesn't match the linear-residual
    shape (no Write, vector Write, multi-Write, non-linear Assign chain).
    The caller splices the result into the surrounding sibling list."""
    writes = [s for s in stmts if isinstance(s, Write)]
    if len(writes) != 1:
        return None
    write = writes[0]
    if write.is_vector:
        return None
    assigns_by_name = {s.name: s for s in stmts if isinstance(s, Assign)}
    if not is_linear_in_accum(write.value, acc_name, assigns_by_name):
        return None
    write_acc = Write(
        output=write.output,
        index=write.index,
        value=acc_name,
        value_dtype=write.value_dtype,
    )
    cond = Cond(
        cond=BinaryExpr("==", Var(k_s_name), Literal(0, "int")),
        body=Body(stmts),
        else_body=Body((write_acc,)),
    )
    return (cond,)
