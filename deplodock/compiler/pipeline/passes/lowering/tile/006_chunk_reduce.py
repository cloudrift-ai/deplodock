"""Chunk non-matmul reduce Loops to make their Loads stage-eligible.

``007_stage_inputs`` requires per-slab smem ≤ 16 KB. For
non-matmul reduces (softmax, SDPA-reduce, RMSNorm) the K extent equals
the full reduce dimension (e.g. seq_len=512). With a 16-wide thread
axis appearing in the Load index, the candidate slab is
``16 × 512 × 4 B = 32 KB`` — over budget, so staging emits nothing and
the kernel runs with no shared-memory reuse.

Matmul reduces are already chunked by ``002_chunk_matmul_k`` (which runs
before launch geometry). This rule covers the remaining reduce shapes.

Runs *after* ``005_blockify_launch`` so the trigger can inspect
``Tile.thread_axes`` and only fire when at least one thread axis is a
fan-in axis for the reduce body's Loads (otherwise chunking adds no
staging benefit).

Trigger — for a reduce ``Loop(K, body=B)`` inside a ``TileOp``'s ``Tile``:

- ``B`` contains at least one ``Load`` whose index references ``K``.
- *Some* thread axis in the enclosing Tile does **not** appear in any
  K-indexed Load's index — that axis would supply staging fan-in.
- For at least one K-indexed Load, the candidate slab
  ``K × prod(in_axis_extents) × 4 B`` exceeds
  ``007_stage_inputs._MAX_SLAB_BYTES`` (16 KB). Chunking when the slab
  already fits would just bloat IR without enabling new staging.
- The Loop is not matmul-shaped (deferred to ``002_chunk_matmul_k``).
- The Loop isn't already chunked (immediate body has no nested reduce).

Rewrite::

    Loop(K, body=B)
    →
    Loop(K_o, body=Loop(K_i, body=B[K → K_o*BK + K_i]))

``K_o`` is a free Loop (no immediate Accum). ``kernel/000_place_inits``
hoists the ``Init`` above ``K_o`` so the accumulator persists across
chunks. Picks the largest ``BK`` from ``_BK_CANDIDATES`` that divides
``K`` and leaves enough budget headroom: ``thread_max × BK × 4 ≤ 8 KB``
(half of ``007_stage_inputs._MAX_SLAB_BYTES``, so multiple slabs and
downstream pad / double-buffer all fit).

Idempotence: a reduce whose immediate body already contains a
reduce-Loop is left alone.
"""

from __future__ import annotations

import os
from dataclasses import replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Cond, Load, Loop, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import BYTES_PER_ELEM, TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce, single_tile

PATTERN = [Pattern("root", TileOp)]

# Match ``007_stage_inputs._MAX_SLAB_BYTES`` exactly — that's the cap
# we're trying to bring slabs under. Chunked slabs target half this so
# downstream pad + double-buffer expansion doesn't push them back over.
_STAGE_SLAB_CAP_BYTES = 16 * 1024
_MAX_SLAB_BYTES_HEADROOM = 8 * 1024
_BK_CANDIDATES = (128, 64, 32, 16, 8)


def rewrite(graph: Graph, root: Node) -> Graph | None:
    # Diagnostic gate for visualizing pipeline behavior with this pass
    # disabled (e.g. smem-layout before/after diffs).
    if os.environ.get("DEPLODOCK_DISABLE_CHUNK_REDUCE") == "1":
        raise RuleSkipped("disabled via DEPLODOCK_DISABLE_CHUNK_REDUCE=1")
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        raise RuleSkipped("rewrite helper returned no change")
    return TileOp(body=new_body, name=root.op.name)


def _maybe_rewrite(body):
    idx, tile = single_tile(body)
    if not tile.thread_axes:
        raise RuleSkipped("Tile has no thread_axes — fan-in undefined")
    new_body, changed = _chunk_in_body(tile.body, tile.thread_axes)
    if not changed:
        raise RuleSkipped("no non-matmul reduce Loop with stage-eligible fan-in needs chunking")
    return body[:idx] + (Tile(axes=tile.axes, body=new_body),) + body[idx + 1 :]


def _chunk_in_body(stmts: tuple, thread_axes: tuple[Axis, ...]) -> tuple[tuple, bool]:
    """Walk a body, chunking every qualifying non-matmul reduce Loop.

    Recurses through wrapper Loops / StridedLoops / Conds so reduces
    nested under output-position free loops are reachable. Unlike
    ``002_chunk_matmul_k`` (which fires on the *first* match per body),
    this pass chunks every qualifying reduce so sibling reductions
    (softmax max, sum, output) all become stage-eligible together.
    """
    out: list[Stmt] = []
    changed = False
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce and _qualifies(s, thread_axes):
            chunked = _chunk_loop(s, thread_axes)
            if chunked is not None:
                out.append(chunked)
                changed = True
                continue
        if isinstance(s, (Loop, StridedLoop)):
            inner, inner_changed = _chunk_in_body(s.body, thread_axes)
            if inner_changed:
                out.append(replace(s, body=inner))
                changed = True
                continue
        if isinstance(s, Cond):
            inner_b, cb = _chunk_in_body(s.body, thread_axes)
            inner_e, ce = _chunk_in_body(s.else_body, thread_axes)
            if cb or ce:
                out.append(Cond(cond=s.cond, body=inner_b, else_body=inner_e))
                changed = True
                continue
        out.append(s)
    return tuple(out), changed


def _qualifies(loop: Loop, thread_axes: tuple[Axis, ...]) -> bool:
    """True iff ``loop`` is a non-matmul reduce with thread-axis fan-in
    over at least one K-indexed Load.

    ``007_stage_inputs`` only stages a Load if at least one cache-axis
    candidate (thread axis or the reduce axis) is *absent* from its
    index — that axis's threads then all read the same staged value.
    Chunking only helps when such fan-in exists; otherwise the reduce
    has no reuse to exploit and we'd just bloat the IR.
    """
    if is_matmul_reduce(loop):
        return False
    # Idempotence: reduce immediately wrapping another reduce was
    # already chunked.
    if any(inner.is_reduce for inner in loop.body.of_type(Loop)):
        return False
    has_fanin, max_in_ext, k_indexed_count = _slab_geometry(loop, thread_axes)
    if k_indexed_count == 0:
        return False
    # Need at least one Load with thread-axis fan-in AND a candidate
    # slab that exceeds ``007_stage_inputs``'s 16 KB per-slab cap.
    # Slab geometry per Load: cache axes = (thread axes appearing in
    # the index) + reduce axis ``K``; slab_bytes = K_ext × prod(ext) ×
    # elem_bytes. If every eligible Load's slab already fits, staging
    # would admit them without chunking — firing here would bloat the
    # IR (and may enable downstream paths like TMA / double-buffer
    # before they're ready).
    K_extent = int(loop.axis.extent)
    needs_chunk = K_extent * max_in_ext * BYTES_PER_ELEM > _STAGE_SLAB_CAP_BYTES
    return has_fanin and needs_chunk


def _slab_geometry(loop: Loop, thread_axes: tuple[Axis, ...]) -> tuple[bool, int, int]:
    """Inspect every K-indexed Load in ``loop.body``.

    Returns ``(has_fanin, max_in_extent_product, k_indexed_count)``:

    - ``has_fanin``: at least one Load has a thread axis *absent* from its
      index (that axis would supply staging fan-in).
    - ``max_in_extent_product``: max over Loads of the product of extents
      of thread axes that *do* appear in the index. The chunked slab is
      ``max_in_extent_product × K_inner × elem_bytes``; the picker uses
      this to pick BK that keeps the slab under the headroom budget.
    - ``k_indexed_count``: number of Loads referencing ``K``. ``0`` lets
      the caller short-circuit (chunking has no staging benefit).
    """
    K_name = loop.axis.name
    has_fanin = False
    max_in_ext = 1
    count = 0
    for ld in loop.body.iter_of_type(Load):
        ld_vars = {v for e in ld.index for v in e.free_vars()}
        if K_name not in ld_vars:
            continue
        count += 1
        in_ext = 1
        for ax in thread_axes:
            if ax.name in ld_vars:
                in_ext *= int(ax.extent)
            else:
                has_fanin = True
        if in_ext > max_in_ext:
            max_in_ext = in_ext
    return has_fanin, max_in_ext, count


def _chunk_loop(loop: Loop, thread_axes: tuple[Axis, ...]) -> Loop | None:
    K = int(loop.axis.extent)
    _, max_in_ext, _ = _slab_geometry(loop, thread_axes)
    BK = _pick_bk(K, max_in_ext)
    if BK is None:
        return None
    K_name = loop.axis.name
    K_o = Axis(f"{K_name}_o", K // BK)
    K_i = Axis(f"{K_name}_i", BK)
    sigma = Sigma({K_name: Var(K_o.name) * Literal(BK, "int") + Var(K_i.name)})
    inner_body = tuple(s.rewrite(_id, sigma) for s in loop.body)
    return Loop(axis=K_o, body=(Loop(axis=K_i, body=inner_body),))


def _pick_bk(K: int, in_extent_product: int) -> int | None:
    """Largest BK from ``_BK_CANDIDATES`` that divides K (with K > BK) and
    keeps the per-Load slab within the headroom budget.

    The chunked slab is ``in_extent_product × BK × 4 B``, where
    ``in_extent_product`` is the product of thread-axis extents
    appearing in any K-indexed Load. Using the actual extent (not a
    hardcoded thread-tile width) matters for kernels whose pre-
    register-tile thread axes are 64-wide rather than 16 — picking
    BK against a 16-wide assumption would still bust the cap there.
    """
    if in_extent_product < 1:
        in_extent_product = 1
    for c in _BK_CANDIDATES:
        if K % c != 0 or K <= c:
            continue
        if in_extent_product * c * BYTES_PER_ELEM > _MAX_SLAB_BYTES_HEADROOM:
            continue
        return c
    return None


def _id(name: str) -> str:
    return name
