"""Cross-thread combine helpers for ``100_materialize_tile``.

A cooperative reduce carrier — a scalar ``Accum`` or the general tuple ``Monoid``
(flash online-softmax) — whose reduction axis is split across the CTA's threads
needs a cross-thread reduce after the per-thread partials land. ``emit_combine``
picks warp-shuffle / hierarchical / block-wide smem tree-halve by thread count,
driven uniformly off the carrier's algebra surface (``carried_names`` /
``combine_operands`` / ``combine_partials``): a scalar ``Accum`` is the degenerate
1-component monoid (``state=("acc",)``, ``combine_states=(Assign("acc", op,
("acc","acc__o")),)``), so ONE emitter and ONE pair of stmts (``WarpShuffle`` /
``TreeHalve``) cover both. The combine reassigns the carried state **in place** — no
``_b`` rename, since the butterfly / tree leaves every thread holding the full
reduction in the carried SSA names.

``find_nested_reduce_carriers`` and ``cooperative_combine_geometry`` are the small
queries the materializer uses to locate the carriers and the combine's tid var +
thread count (the whole CTA in the BN=BM=1 form, or each row's BR-lane segment when
free-axis threads ride alongside — strided-cooperative rows).

Pure functions — no shared materializer state. The leading-underscore
module name keeps the pass loader (globs ``*.py``, skips ``_``-prefixed)
from mistaking this for a rule.
"""

from __future__ import annotations

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel.ir import Smem, Sync, TreeHalve, WarpShuffle
from deplodock.compiler.ir.stmt import Accum, Cond, Monoid, Stmt, Write
from deplodock.compiler.ir.tile.ir import RegisterTile, SerialTile, StridedTile


def find_nested_reduce_carriers(stmts) -> dict[str, Stmt]:
    """All reduce carriers (``Accum`` / ``Monoid``) at the immediate body level of
    the first nested reduce ``SerialTile`` / ``StridedTile`` subtree, keyed by the
    carrier's first carried name (``carried_names()[0]`` — the ``Accum`` name or the
    ``Monoid``'s first state component).

    Used by the materializer when a non-reduce outer tile wraps a deeper reduce —
    e.g. the cooperative-K shape ``SerialTile(K_o, "serial_outer",
    body=[SerialTile(K_i, "stage_inner", reduce, [Accum, ...])])`` produced by the
    partition planner's σ-split, possibly with F-replicated sibling carriers from
    ``010_split_register_axes``. ``Mma`` carriers are excluded — the tensor-core
    fragment fold has its own combine path.

    Returns ``{}`` when no reduce-with-carrier subtree is found."""
    for s in stmts:
        if isinstance(s, (SerialTile, StridedTile)) and s.is_reduce:
            carriers = {c.carried_names()[0]: c for c in s.body if isinstance(c, (Accum, Monoid))}
            if carriers:
                return carriers
        if isinstance(s, (SerialTile, StridedTile, RegisterTile)):
            found = find_nested_reduce_carriers(s.body)
            if found:
                return found
    return {}


def cooperative_combine_geometry(thread_axes: tuple[Axis, ...], coop_names: frozenset[str], *, warp_size: int) -> tuple[str, int]:
    """``(tid_var, n_threads)`` for one Accum's cross-thread combine.

    ``coop_names`` is the Accum's cooperative axis set
    (``Body.coordination.accum_cooperative_axes`` — reduce axes that are
    also THREAD axes; exactly one ``K_c`` by planner construction).

    - **Whole-CTA** (every THREAD axis cooperative — the BN=BM=1 form):
      the combine spans the CTA, any size; ``emit_combine`` picks warp
      shuffle / hierarchical / smem tree-halve.
    - **Strided-cooperative rows** (free-axis THREAD axes alongside):
      the combine must be a SEGMENTED warp shuffle over each row's BR
      lanes, valid only when the cooperative axis is the innermost
      (fastest-varying) THREAD axis — its lanes then form a contiguous
      BR-aligned intra-warp group — and BR is a power of two ≤
      warp_size (a segment must not straddle a warp). The planner /
      enumerator guarantee both; violations raise here rather than
      emit a mis-combining kernel.
    """
    coop = [ax for ax in thread_axes if ax.name in coop_names]
    if len(coop) != 1:
        raise ValueError(f"Monoid requires exactly one cooperative THREAD axis; got {[ax.name for ax in coop]}")
    n_coop = coop[0].extent.as_static()
    if len(coop) == len(thread_axes):
        return coop[0].name, n_coop
    if thread_axes[-1].name != coop[0].name:
        raise ValueError(f"cooperative axis {coop[0].name!r} must be the innermost THREAD axis for the segmented-shuffle combine")
    if n_coop > warp_size or n_coop & (n_coop - 1):
        raise ValueError(f"segmented-shuffle combine needs power-of-two BR <= {warp_size}; got {n_coop}")
    return coop[0].name, n_coop


def emit_combine(
    carrier,
    t: str,
    n_threads: int,
    *,
    warp_size: int,
    barrier_id: int = 0,
    barrier_count: int | None = None,
) -> list[Stmt]:
    """Emit the cross-thread combine for a reduce carrier (``Accum`` / ``Monoid``),
    reassigning the carried state **in place**.

    Driven off the carrier's algebra surface — ``carried_names()`` (the state),
    ``combine_operands()`` (the second-operand state names), ``combine_partials()``
    (the state-merges-state program). A scalar ``Accum`` is the degenerate
    1-component case; the tuple ``Monoid`` (flash online-softmax) is the general one.
    Three paths, picked by ``n_threads`` (all require ``commutative`` — the butterfly
    / tree reorders — checked at the carrier):

    - **Warp** (``n_threads ≤ warp_size``, power of two): a single ``WarpShuffle``
      register butterfly. No smem, no syncthreads. The XOR butterfly never crosses
      an aligned ``n_threads``-lane group, so this is also the SEGMENTED per-row
      combine for strided-cooperative rows (caller passes the segment size as
      ``n_threads`` — see :func:`cooperative_combine_geometry`).
    - **Hierarchical** (``n_threads`` a power-of-two multiple of ``warp_size``): each
      warp first shuffle-reduces its lanes in place (broadcast within the warp);
      lane 0 of each warp writes the per-warp state to a tiny ``smem[n_warps]`` slab
      per component; one ``Sync`` + ``TreeHalve(length=n_warps, tid_var="warp")``
      collapses across warps and broadcasts. The cross-warp reduce is one round of
      compare-sync (``n_warps`` ≪ ``n_threads``) instead of ``log2(n_threads)``.
    - **Block** (otherwise — power of two, not a clean warp multiple): each thread
      writes its partial to a ``smem[n_threads]`` slab per component, one ``Sync``,
      a single ``TreeHalve(length=n_threads)`` reduces + broadcasts in place.

    ``dtype`` flows from the carrier (``Accum.dtype``; fp32 for a ``Monoid``) so the
    smem slabs + the combine render in the accumulator's element type.

    The Tile renderer emits ``int lane = threadIdx.x & 31;`` and ``int warp =
    threadIdx.x >> 5;`` for any cooperative Tile with ``n_threads > warp_size`` so
    the hierarchical path's ``Var("lane")`` / ``Var("warp")`` references resolve.

    ``barrier_id`` / ``barrier_count`` route the emitted ``Sync`` + ``TreeHalve``'s
    per-iter sync to a named barrier when non-zero (warp-specialized consumer branch
    — sync only the consumer threads, not the whole CTA). The warp-shuffle path uses
    ``__shfl_xor_sync`` (intra-warp), so it's unaffected.
    """
    state = carrier.carried_names()
    state_b = carrier.combine_operands()
    prog = carrier.combine_partials()
    dtype = getattr(carrier, "dtype", None) or F32

    if n_threads <= warp_size and (n_threads & (n_threads - 1)) == 0:
        return [WarpShuffle(state=state, state_b=state_b, combine_states=prog, length=n_threads, dtype=dtype)]
    if n_threads & (n_threads - 1):
        raise ValueError(f"cross-thread combine needs a power-of-two thread count, got {n_threads}")

    from deplodock.compiler.backend.cuda.dtype import cuda_name as _cuda_name  # noqa: PLC0415

    smem_c = _cuda_name(dtype)
    bufs = tuple(f"{st}_smem" for st in state)

    if n_threads % warp_size == 0:
        # Hierarchical: per-warp in-place WarpShuffle, lane-0 of each warp stages its
        # warp's state into a tiny n_warps slab, a TreeHalve collapses + broadcasts.
        n_warps = n_threads // warp_size
        out: list[Stmt] = [WarpShuffle(state=state, state_b=state_b, combine_states=prog, length=warp_size, dtype=dtype)]
        out += [Smem(name=b, extents=(n_warps,), dtype=smem_c) for b in bufs]
        out.append(
            Cond(
                cond=BinaryExpr("==", Var("lane"), Literal(0, "int")),
                body=tuple(Write(output=b, index=(Var("warp"),), value=st) for b, st in zip(bufs, state, strict=True)),
            )
        )
        out.append(Sync(barrier_id=barrier_id, count=barrier_count))
        out.append(
            TreeHalve(
                bufs=bufs,
                state=state,
                state_b=state_b,
                combine_states=prog,
                length=n_warps,
                tid_var="warp",
                dtype=dtype,
                barrier_id=barrier_id,
                barrier_count=barrier_count,
            )
        )
        return out

    # Block: every thread stages its partial into a full-width slab, TreeHalve folds.
    out = [Smem(name=b, extents=(n_threads,), dtype=smem_c) for b in bufs]
    out += [Write(output=b, index=(Var(t),), value=st) for b, st in zip(bufs, state, strict=True)]
    out.append(Sync(barrier_id=barrier_id, count=barrier_count))
    out.append(
        TreeHalve(
            bufs=bufs,
            state=state,
            state_b=state_b,
            combine_states=prog,
            length=n_threads,
            tid_var=t,
            dtype=dtype,
            barrier_id=barrier_id,
            barrier_count=barrier_count,
        )
    )
    return out
