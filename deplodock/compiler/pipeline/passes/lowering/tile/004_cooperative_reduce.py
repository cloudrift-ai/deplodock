"""Cooperative-reduce strategy — converts each cooperative axis from a
serial ``Loop`` to a ``StridedLoop`` driven by the cooperative thread
axis ``t``.

Reads the logical ``Tile`` produced by ``tileify`` (default:
``thread_axes=output_axes`` / ``block_axes=()``, inner Loops all
serial) and rewrites:

- The Tile's output axis (``thread_axes``) flips to ``BIND_BLOCK`` —
  one CUDA block per output slot.
- A synthetic cooperative thread axis ``t`` (``BLOCK_SIZE`` threads)
  is added to ``Tile.axes`` as ``BIND_THREAD``.
- Each inner ``Loop`` becomes a ``StridedLoop(axis, start=Var("t"),
  step=BLOCK_SIZE)`` — threads of the block stride through the axis.
  Body indices stay as ``Var(axis.name)`` (no rewriting); the strided
  iteration is encoded by the loop construct itself.
- After each reduce loop, a ``Combine(name, op)`` sibling is inserted;
  materialization emits the cross-thread tree-halve over smem indexed
  by ``t``.

Post-rewrite example (softmax)::

    Tile(axes=(t=THREAD, i=BLOCK), body=(
      StridedLoop(k1, start=t, step=256, body=(
        Load input[i, k1],
        Accum("acc_max", max),
      )),
      Combine("acc_max", max),
      StridedLoop(k2, start=t, step=256, body=(
        Load input[i, k2],
        Assign, Assign,
        Accum("acc_sum", add),
      )),
      Combine("acc_sum", add),
      StridedLoop(k3, start=t, step=256, body=(
        Load input[i, k3],
        Assign, Assign, Assign,
        Write merged[i, k3],
      )),
    ))

Trigger conditions:

- ``TileOp.body`` contains exactly one ``Tile``.
- ``Tile.thread_axes`` is non-empty and ``block_axes`` is empty
  (idempotence). Multi-axis output tiles (e.g. softmax over the last
  dim of a 4D tensor) are handled by promoting all existing thread
  axes to ``BIND_BLOCK`` alongside the synthetic cooperative ``t``.
- ``Tile.body`` contains at least one reduce ``Loop`` whose immediate
  body has one or more ``Accum`` stmts. Multiple Accums are permitted
  as long as they are independent (no Accum's value transitively reads
  another Accum's running value); online algorithms (online softmax,
  Welford) are still punted.
- The first reduce Loop's axis extent ≥ ``WARP_SIZE`` (32). When the
  extent is smaller than the configured ``BLOCK_SIZE`` we still
  cooperate, but use ``next_pow2(extent)`` threads instead — softmax
  over a 128-wide attention row gets a 128-thread block-reduce rather
  than each of 128 threads redundantly walking the row sequentially.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Body, Load, Loop, StridedLoop
from deplodock.compiler.ir.tile.ir import (
    BLOCK_SIZE,
    Combine,
    Stmt,
    Tile,
    TileOp,
)
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

_WARP_SIZE = 32


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _effective_block_size(reduce_extent: int) -> int:
    """Threads per CTA for the cooperative reduce. Capped at the
    configured ``BLOCK_SIZE``; floored at ``WARP_SIZE`` so the
    cross-thread tree-halve still has a full warp to combine."""
    return max(_WARP_SIZE, min(BLOCK_SIZE, _next_pow2(reduce_extent)))


def _has_matmul_reduce(body) -> bool:
    """True iff ``body`` contains a reduce ``Loop`` whose immediate
    Loads touch ≥2 distinct buffers — the structural signature of a
    matmul (or P @ V style) reduction."""
    return any(
        isinstance(s, Loop) and s.is_reduce and len({ld.input for ld in s.body.of_type(Load)}) >= 2 for s in Body.coerce(body).iter()
    )


PATTERN = [Pattern("root", TileOp)]


def _accums_independent(body: Body) -> bool:
    """True iff no Accum's value transitively depends on another
    Accum's running value. Permits multiple independent Accums in one
    reduce loop (e.g. ``sum`` + ``sum_of_squares``); rejects online
    algorithms (online softmax, Welford) where one accumulator's
    update reads another's running value.
    """
    body = Body.coerce(body)
    accum_names = {s.name for s in body if isinstance(s, Accum)}
    return not any(body.depends_on(s.value, accum_names - {s.name}) for s in body if isinstance(s, Accum))


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: tuple) -> tuple | None:
    idx, blk = single_tile(body)
    if blk.block_axes:
        raise RuleSkipped("Tile already cooperative (block_axes non-empty)")

    new_blk = _rewrite_block(blk)
    if new_blk is None:
        return None
    return body[:idx] + (new_blk,) + body[idx + 1 :]


def _rewrite_block(blk: Tile) -> Tile | None:
    if not blk.thread_axes:
        raise RuleSkipped("Tile has no thread_axes to convert to BLOCK")

    reduce_loops = [loop for loop in blk.body.of_type(Loop) if loop.is_reduce]
    if not reduce_loops:
        raise RuleSkipped("Tile body has no reduce Loop")
    reduce_extent = int(reduce_loops[0].axis.extent)
    if reduce_extent < _WARP_SIZE:
        raise RuleSkipped(f"first reduce-axis extent {reduce_extent} < WARP_SIZE={_WARP_SIZE} (too small to cooperate)")
    eff_block = _effective_block_size(reduce_extent)
    for rl in reduce_loops:
        if not any(isinstance(s, Accum) for s in rl.body):
            raise RuleSkipped(f"reduce Loop {rl.axis.name!r} has no Accum")
        if not _accums_independent(rl.body):
            raise RuleSkipped(f"reduce Loop {rl.axis.name!r} has dependent Accums (online algorithm — punted)")

    # Skip cooperative when the body has a matmul-reduce signature (a
    # reduce Loop with ≥2 distinct buffer Loads). Such kernels have
    # plenty of output-dim parallelism: blockify's matmul-aware path
    # will thread-tile (PAT × PAT) on the outputs, register_tile will
    # F²-replicate, and the per-thread arithmetic-amortization wins
    # over what cooperative would buy. SDPA's P @ V kernel is the
    # poster child: cooperative makes it 1M CTAs with redundant softmax,
    # while the matmul path lands ~16K CTAs with per-thread tile reuse.
    if any(_has_matmul_reduce(s.body) for s in reduce_loops if isinstance(s, Loop)) or _has_matmul_reduce(blk.body):
        raise RuleSkipped("body has matmul-reduce signature — defer to register_tile / blockify matmul path")

    t_axis = Axis("t", eff_block)
    t_start = Var(t_axis.name)
    step = Literal(eff_block, "int")

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

    # Phase 2: identify the innermost "naked" output axis — one
    # referenced *only* at Tile level (outside any Loop / StridedLoop)
    # with extent ≥ BLOCK_SIZE divisible. Axes referenced *both* at Tile
    # level and inside an inner Loop can't be wrapped this way: the
    # wrap would only bring the axis into scope at the tail, leaving
    # in-loop references unbound. Without wrapping, these stay BLOCK.
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

    # Phase 3: wrap the post-reduce tail (everything after the last
    # block-structured stmt) in a StridedLoop over the chosen naked axis.
    #
    # If a preceding reduce loop has the same extent as the naked axis,
    # alias the naked axis name to that reduce axis name in the suffix.
    # This makes the post-reduce Loads textually equal to the in-reduce
    # Loads (e.g. softmax: ``x[..., a3]`` vs ``x[..., a4]`` collapse to
    # ``x[..., a3]`` everywhere), so ``007_stage_inputs`` partitions
    # them into a single Stage instead of allocating two identical
    # smem buffers and double-loading the row from DRAM.
    if naked_axis is not None:
        split_idx = 0
        for i, s in enumerate(body_phase1):
            if isinstance(s, (Loop, StridedLoop, Combine)):
                split_idx = i + 1
        prefix = body_phase1[:split_idx]
        suffix: list[Stmt] = body_phase1[split_idx:]
        wrap_axis = naked_axis
        for rl in reduce_loops:
            if int(rl.axis.extent) == int(naked_axis.extent) and rl.axis.name != naked_axis.name:
                wrap_axis = rl.axis
                break
        if wrap_axis is not naked_axis:
            sigma = Sigma({naked_axis.name: Var(wrap_axis.name)})
            suffix = [s.rewrite(lambda n: n, sigma) for s in suffix]
        new_body: list[Stmt] = [*prefix, StridedLoop(axis=wrap_axis, start=t_start, step=step, body=suffix)]
    else:
        new_body = body_phase1

    # Phase 4: bind axes. t → THREAD; output axes → BLOCK except the
    # wrapped one (its iteration lives in the StridedLoop wrapper, so
    # remove it from Tile.axes).
    naked_name = naked_axis.name if naked_axis is not None else None
    new_axes = (
        BoundAxis(axis=t_axis, bind=BIND_THREAD),
        *(BoundAxis(axis=ba.axis, bind=BIND_BLOCK) for ba in blk.axes if ba.axis.name != naked_name),
    )
    return Tile(axes=new_axes, body=new_body)


def _stmt_free_vars(s: Stmt) -> set[str]:
    """All Var names referenced anywhere in ``s`` (recursive over nested
    bodies). Used to detect output-axis references at Tile level."""
    out: set[str] = set()
    for e in s.exprs():
        out |= e.free_vars()
    for child_body in s.nested():
        for c in child_body:
            out |= _stmt_free_vars(c)
    return out
