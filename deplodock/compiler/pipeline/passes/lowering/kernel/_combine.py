"""Cross-thread combine emission for the cooperative materializer (``010_materialize``).

A cooperative reduce folds the reduce axis across the CTA's threads: each thread lands a
per-lane partial in the carrier's state, then a cross-thread combine merges them. This
module is the **emit** half — it constructs the ``WarpShuffle`` / ``Smem`` + ``Sync`` +
``TreeHalve`` stmts. The combine-plan *branching* (which fold mechanism at each level)
lives on :meth:`ReduceStage.combine` (derived from the level); here we just realize the
folds it returns.

The combine is pure **geometry** carrying the carrier's combine ALGEBRA straight onto the
nodes: it reads the carrier's combine surface (``state.names`` / ``twist.state_b`` /
``twist.combine_states``) and the nodes realize the merge via ``render_merge_program`` (the
SAME realizer the streaming merge uses). So one emitter covers any carrier — a degenerate
``sum`` / ``max`` (the auto-derived ``name = op(name, name__o)`` combine) and a future
twisted carrier alike. The combine reassigns the carried state **in place** (the butterfly
/ tree leaves every thread holding the full reduction in the carried SSA names — no ``_b``
rename for the post-reduce epilogue).

Leading ``_`` so the pass loader (globs ``*.py``, skips ``_``-prefixed) skips this module.
"""

from __future__ import annotations

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel.ir import Smem, Sync, TreeHalve, WarpShuffle
from deplodock.compiler.ir.stmt import Cond, Write
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.ir.tile import Fold, Level, ReduceStage


def emit_combine(carrier, t: str, n_threads: int, *, warp_size: int = 32, segmented: bool = False) -> list[Stmt]:
    """Build the cross-thread combine of a cooperative reduce ``carrier`` (a ``Monoid``)
    over ``n_threads`` cooperating threads, reassigning the carried state in place.

    The mechanism per level is derived by :meth:`ReduceStage.combine`:

    - a ``SHFL`` fold → one ``WarpShuffle`` register butterfly. The XOR butterfly never
      crosses an aligned ``width``-lane group, so a lone ``SHFL`` is also the SEGMENTED
      per-row combine for strided-cooperative rows (caller passes ``segmented=True``).
    - a ``SMEM`` fold **after** a ``SHFL`` → the hierarchical cross-warp slab: lane-0 of each
      warp stages its broadcast state to a ``smem[n_warps]`` slab per component; one ``Sync``
      + ``TreeHalve(tid_var="warp")`` collapses across warps and broadcasts.
    - a standalone ``SMEM`` → the block slab: every thread stages its partial, one ``Sync``,
      a single ``TreeHalve`` reduces + broadcasts in place.

    The carrier's combine surface (``state.names`` / ``twist.state_b`` /
    ``twist.combine_states``) drives the nodes; the combine renders at the accumulator dtype
    (fp32 for a reduction, with the carrier's own dtype honored when set)."""
    state = carrier.state.names
    state_b = carrier.state_b
    prog = carrier.combine_states
    dtype = next((a.dtype for a in prog if a.dtype is not None), None) or F32

    folds = ReduceStage(Level.BLOCK, n_threads).combine(warp_size=warp_size, segmented=segmented)

    from deplodock.compiler.backend.cuda.dtype import cuda_name as _cuda_name  # noqa: PLC0415

    smem_c = _cuda_name(dtype)
    bufs = tuple(f"{st}_smem" for st in state)
    out: list[Stmt] = []
    for i, fold in enumerate(folds):
        if fold is Fold.SHFL:
            # The lane-level butterfly: warp-wide when followed by a cross-warp SMEM stage
            # (hierarchical), else the full ``n_threads`` (one warp / segment).
            width = warp_size if (len(folds) == 2 and folds[1] is Fold.SMEM) else n_threads
            out.append(WarpShuffle(state=state, state_b=state_b, combine_states=prog, length=width, dtype=dtype))
        elif fold is Fold.SMEM:
            hierarchical = i > 0 and folds[i - 1] is Fold.SHFL
            width = n_threads // warp_size if hierarchical else n_threads
            tid_var = "warp" if hierarchical else t
            out += [Smem(name=b, extents=(width,), dtype=smem_c) for b in bufs]
            if hierarchical:
                # Lane-0 of each warp stages that warp's broadcast state, indexed by ``warp``.
                out.append(
                    Cond(
                        cond=BinaryExpr("==", Var("lane"), Literal(0, "int")),
                        body=tuple(Write(output=b, index=(Var("warp"),), value=st) for b, st in zip(bufs, state, strict=True)),
                    )
                )
            else:
                out += [Write(output=b, index=(Var(tid_var),), value=st) for b, st in zip(bufs, state, strict=True)]
            out.append(Sync())
            out.append(TreeHalve(bufs=bufs, state=state, state_b=state_b, combine_states=prog, length=width, tid_var=tid_var, dtype=dtype))
        else:  # Fold.ATOMIC / Fold.REG — cross-CTA / register tiers, not emitted by the intra-CTA walk.
            raise NotImplementedError(f"intra-CTA combine cannot emit {fold} (cta/reg tiers are future work)")
    return out
