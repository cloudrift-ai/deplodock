"""Cross-thread combine helpers for ``008_materialize_tile``.

A cooperative ``Accum`` whose reduction axis is split across the CTA's
threads needs a cross-thread reduce after the per-thread partials land.
``emit_combine`` picks warp-shuffle / hierarchical / block-wide smem
tree-halve by thread count; ``find_nested_reduce_accums`` and
``single_thread_var`` are the small queries the materializer uses to
locate the Accums and the single THREAD axis the combine runs over.

Pure functions — no shared materializer state. The leading-underscore
module name keeps the pass loader (globs ``*.py``, skips ``_``-prefixed)
from mistaking this for a rule.
"""

from __future__ import annotations

from deplodock.compiler.dtype import F32, DataType
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel.ir import Smem, Sync, TreeHalve, WarpShuffle
from deplodock.compiler.ir.stmt import Accum, Cond, Load, Stmt, Write
from deplodock.compiler.ir.tile.ir import RegisterTile, SerialTile, StridedTile


def find_nested_reduce_accums(stmts) -> dict[str, Accum]:
    """All ``Accum``s sitting at the immediate-body level of the first
    nested reduce ``SerialTile`` / ``StridedTile`` subtree, keyed by
    Accum name. Used by the materializer when a non-reduce outer tile
    wraps a deeper reduce — e.g. the cooperative-K shape
    ``SerialTile(K_o, "serial_outer", body=[SerialTile(K_i, "stage_inner",
    reduce, [Accum, ...])])`` produced by the partition planner's σ-split,
    possibly with F-replicated sibling Accums from
    ``006a_register_tile_planned``. Returns ``{}`` when no reduce-with-
    immediate-Accum is found, preserving the existing "stray Combine
    raises" safety net."""
    for s in stmts:
        if isinstance(s, (SerialTile, StridedTile)) and s.is_reduce:
            accums = {a.name: a for a in s.body if isinstance(a, Accum)}
            if accums:
                return accums
        if isinstance(s, (SerialTile, StridedTile, RegisterTile)):
            found = find_nested_reduce_accums(s.body)
            if found:
                return found
    return {}


def single_thread_var(thread_axes: tuple[Axis, ...]) -> str:
    """Combine + TreeHalve emit a single ``tid_var`` string. Only valid
    when there's exactly one THREAD axis — softmax-style cooperation
    (matmul has multi-axis THREAD set but doesn't emit Combine)."""
    if len(thread_axes) != 1:
        raise ValueError(f"Combine requires a single THREAD axis; got {len(thread_axes)}")
    return thread_axes[0].name


def emit_combine(name: str, op, t: str, n_threads: int, dtype: DataType = F32, *, warp_size: int) -> list[Stmt]:
    """Emit the cross-thread combine producing ``<name>_b``.

    Three paths, picked by ``n_threads``:

    - **Warp** (``n_threads ≤ WARP_SIZE`` and power of two): a single
      ``WarpShuffle`` butterfly via ``__shfl_xor_sync``. No smem, no
      syncthreads.
    - **Hierarchical** (``n_threads`` a power-of-two multiple of
      ``WARP_SIZE``): each warp first shuffle-reduces its lanes into
      register-resident ``<acc>_w`` (broadcast within the warp); lane 0
      of each warp writes ``<acc>_w`` to a tiny ``smem[n_warps]`` slab;
      one ``Sync`` + ``TreeHalve(length=n_warps)`` collapses across
      warps; broadcast load delivers ``<acc>_b``. The ``TreeHalve``
      runs on the ``warp`` index — sized to ``n_warps`` (4 / 8 / etc.)
      rather than ``n_threads`` (128 / 256 / etc.), so the cross-warp
      reduce is one round of compare-sync instead of five.
    - **Block** (otherwise — n_threads not a clean multiple of 32):
      legacy path. Each thread writes its partial to a smem buffer
      indexed by ``t``, a single ``TreeHalve`` over ``n_threads``
      reduces in place, broadcast load.

    ``dtype`` flows from the parent ``Accum.dtype`` (set by the
    Init-placement pass) so the per-warp register, the inter-warp smem
    slab, and the TreeHalve combine all render in the accumulator's
    element type — fp16 reductions stay fp16 across the inter-warp
    step instead of promoting back to fp32 in the broadcast.

    The Tile renderer emits ``int lane = threadIdx.x & 31;`` and
    ``int warp = threadIdx.x >> 5;`` for any cooperative Tile with
    ``n_threads > WARP_SIZE`` so the hierarchical path's ``Var("lane")``
    / ``Var("warp")`` references resolve.
    """
    from deplodock.compiler.backend.cuda.dtype import cuda_name as _cuda_name  # noqa: PLC0415

    smem_c_name = _cuda_name(dtype)
    broadcast_name = f"{name}_b"
    if n_threads <= warp_size and (n_threads & (n_threads - 1)) == 0:
        return [WarpShuffle(name=broadcast_name, value=name, op=op, length=n_threads, dtype=dtype)]
    if n_threads % warp_size == 0 and (n_threads & (n_threads - 1)) == 0:
        n_warps = n_threads // warp_size
        smem_name = f"{name}_smem"
        warp_w = f"{name}_w"
        return [
            WarpShuffle(name=warp_w, value=name, op=op, length=warp_size, dtype=dtype),
            Smem(name=smem_name, extents=(n_warps,), dtype=smem_c_name),
            Cond(
                cond=BinaryExpr("==", Var("lane"), Literal(0, "int")), body=(Write(output=smem_name, index=(Var("warp"),), value=warp_w),)
            ),
            Sync(),
            TreeHalve(buf=smem_name, op=op, length=n_warps, tid_var="warp", dtype=dtype),
            # TreeHalve's render ends each loop iter with __syncthreads(), so a
            # trailing Sync here would be a no-op pair with the loop's last sync.
            Load(name=broadcast_name, input=smem_name, index=(Literal(0, "int"),)),
        ]
    smem_name = f"{name}_smem"
    return [
        Smem(name=smem_name, extents=(n_threads,), dtype=smem_c_name),
        Write(output=smem_name, index=(Var(t),), value=name),
        Sync(),
        TreeHalve(buf=smem_name, op=op, length=n_threads, tid_var=t, dtype=dtype),
        # See note above on TreeHalve's trailing sync.
        Load(name=broadcast_name, input=smem_name, index=(Literal(0, "int"),)),
    ]
