"""Materialize the body rewrite for cooperative-reduce kernels.

Runs after ``004_launch_geometry`` has decided the launch shape: a
Tile with a synthetic ``t=THREAD`` axis (cooperative threads) and
every output axis bound to ``BLOCK``. This pass performs the body-side
transformation:

- Each reduce ``Loop`` becomes a ``StridedLoop(axis, start=Var("t"),
  step=BLOCK_SIZE, body=...)`` — threads of the block stride through
  the axis. The original axis Var stays in body indices; the strided
  iteration shape is encoded by the loop construct.
- After each reduce loop, one ``Combine(name, op)`` per Accum is
  inserted; materialization emits the cross-thread tree-halve.
- A *naked* output axis (one referenced only at Tile-level, outside any
  loop, with extent divisible by the cooperative thread count) is
  consumed by a StridedLoop wrapping the post-reduce epilogue. The
  axis is removed from ``Tile.axes`` since its iteration now lives in
  the wrapper. LICM hoists epilogue stmts whose dependencies don't
  involve the wrap axis out of the wrapper.

Trigger:

- Exactly one ``Tile`` in the body.
- A ``t=THREAD`` axis is present (signal from 004).
- The body has at least one reduce ``Loop`` (otherwise nothing to
  rewrite — matmul / pointwise tiles get here as no-ops).
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Expr, Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Loop, StridedLoop
from deplodock.compiler.ir.tile.ir import Combine, Stmt, Tile, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

PATTERN = [Pattern("root", TileOp)]

_WARP_SIZE = 32


def rewrite(root: Node) -> Graph | None:
    body = root.op.body
    idx, tile = single_tile(body)

    # The cooperative launch from 004 leaves exactly one THREAD axis
    # (the synthetic ``t``) and BLOCK-binds the output. Matmul tiles
    # have ≥2 THREAD axes; pointwise tiles have no reduce Loop in body.
    thread_axes = tile.thread_axes
    if len(thread_axes) != 1:
        raise RuleSkipped(f"need exactly 1 THREAD axis (cooperative ``t``); have {len(thread_axes)}")
    reduce_loops = [s for s in tile.body if isinstance(s, Loop) and s.is_reduce]
    if not reduce_loops:
        raise RuleSkipped("Tile body has no reduce Loop")
    # Mirror 004's viability gate: extent ≥ WARP_SIZE means 004 elected
    # cooperative launch and added ``t``; otherwise the lone THREAD axis
    # is just a pointwise output dim and we mustn't rewrite the body.
    if int(reduce_loops[0].axis.extent) < _WARP_SIZE:
        raise RuleSkipped(f"first reduce-axis extent < WARP_SIZE={_WARP_SIZE} (004 didn't elect cooperative)")
    # Idempotence: post-rewrite the reduces have become StridedLoops.
    if any(isinstance(s, StridedLoop) and s.is_reduce for s in tile.body):
        raise RuleSkipped("Tile body already cooperative (reduce StridedLoop present)")

    new_tile = _rewrite_block(tile, thread_axes[0])
    return TileOp(body=body[:idx] + (new_tile,) + body[idx + 1 :], name=root.op.name)


def _rewrite_block(blk: Tile, t_axis: Axis) -> Tile:
    eff_block = int(t_axis.extent)
    t_start: Expr = Var(t_axis.name)
    step: Expr = Literal(eff_block, "int")

    # Phase 1: each body Loop → StridedLoop driven by t. Reduce Loops
    # get a Combine sibling for the cross-thread tree-halve.
    body_phase1: list[Stmt] = []
    for s in blk.body:
        if isinstance(s, Loop):
            body_phase1.append(StridedLoop(axis=s.axis, start=t_start, step=step, body=s.body))
            if s.is_reduce:
                for a in s.body:
                    if isinstance(a, Accum):
                        body_phase1.append(Combine(name=a.name, op=a.op))
        else:
            body_phase1.append(s)

    # Phase 2: pick a naked output axis — referenced only at Tile-level
    # (outside any loop) with extent divisible by the cooperative thread
    # count. Wrap the post-reduce epilogue in a StridedLoop over it.
    in_loop_vars: set[str] = set()
    tile_level_vars: set[str] = set()
    for s in body_phase1:
        if isinstance(s, (Loop, StridedLoop)):
            in_loop_vars |= _stmt_free_vars(s)
        elif not isinstance(s, Combine):
            tile_level_vars |= _stmt_free_vars(s)
    naked_only_tile = tile_level_vars - in_loop_vars
    naked_axis: Axis | None = None
    for ba in reversed(list(blk.axes)):
        if ba.axis.name not in naked_only_tile:
            continue
        ext = int(ba.axis.extent)
        if ext >= eff_block and ext % eff_block == 0:
            naked_axis = ba.axis
            break

    # Phase 3: if we found a naked axis, wrap the post-reduce tail
    # in a StridedLoop. Alias the wrap axis name to a preceding reduce
    # axis with matching extent so post-reduce Loads match in-reduce
    # Loads textually (007_stage_inputs can then stage them together).
    reduce_axes = [s.axis for s in body_phase1 if isinstance(s, StridedLoop) and s.is_reduce]
    if naked_axis is not None:
        split_idx = 0
        for i, s in enumerate(body_phase1):
            if isinstance(s, (Loop, StridedLoop, Combine)):
                split_idx = i + 1
        prefix = body_phase1[:split_idx]
        suffix: list[Stmt] = body_phase1[split_idx:]
        wrap_axis = naked_axis
        for ra in reduce_axes:
            if int(ra.extent) == int(naked_axis.extent) and ra.name != naked_axis.name:
                wrap_axis = ra
                break
        if wrap_axis is not naked_axis:
            sigma = Sigma({naked_axis.name: Var(wrap_axis.name)})
            suffix = [s.rewrite(lambda n: n, sigma) for s in suffix]
        # LICM the suffix: stmts whose deps don't transitively involve the
        # wrap axis hoist out of the StridedLoop. Bespoke walker because
        # generic ``normalize_body`` treats post-Combine Accums as still
        # axis-dependent, missing the cross-thread broadcast semantics.
        hoisted, inner = _licm_split(suffix, wrap_axis.name)
        new_body: list[Stmt] = [*prefix, *hoisted, StridedLoop(axis=wrap_axis, start=t_start, step=step, body=inner)]
    else:
        new_body = body_phase1

    # Phase 4: drop the wrapped axis from Tile.axes — its iteration now
    # lives in the StridedLoop wrapper.
    naked_name = naked_axis.name if naked_axis is not None else None
    new_axes = tuple(ba for ba in blk.axes if ba.axis.name != naked_name)
    return Tile(axes=new_axes, body=new_body)


def _licm_split(suffix: list[Stmt], wrap_axis_name: str) -> tuple[list[Stmt], list[Stmt]]:
    """Split ``suffix`` into ``(hoisted, inner)``: stmts whose deps don't
    involve ``wrap_axis_name`` hoist out of the wrapper; the rest stay
    inside. Body-bearing stmts conservatively stay inside."""
    tainted: set[str] = {wrap_axis_name}
    hoisted: list[Stmt] = []
    inner: list[Stmt] = []
    for s in suffix:
        if s.nested():
            inner.append(s)
            for d in s.defines():
                tainted.add(d)
            continue
        free = _stmt_free_vars(s)
        deps = set(s.deps())
        if free & tainted or deps & tainted:
            inner.append(s)
            for d in s.defines():
                tainted.add(d)
        else:
            hoisted.append(s)
    return hoisted, inner


def _stmt_free_vars(s: Stmt) -> set[str]:
    """All Var names referenced anywhere in ``s`` (recursive over nested bodies)."""
    out: set[str] = set()
    for e in s.exprs():
        out |= e.free_vars()
    for child_body in s.nested():
        for c in child_body:
            out |= _stmt_free_vars(c)
    return out


# Re-export for tests that imported the symbol from the old module path.
def _has_matmul_reduce(body) -> bool:  # noqa: ARG001 — compat shim removed; see tuning._has_matmul_reduce
    raise RuntimeError("_has_matmul_reduce moved — import from deplodock.compiler.tuning instead")
