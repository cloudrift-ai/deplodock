"""Cross-thread combine helpers for ``100_materialize_tile``.

A cooperative reduce carrier — a scalar ``Accum`` or the general tuple ``Monoid``
(flash online-softmax) — whose reduction axis is split across the CTA's threads
needs a cross-thread reduce after the per-thread partials land. ``emit_combine`` builds the
cross-thread **distribution** — pure geometry (``WarpShuffle`` / ``Smem`` / ``TreeHalve`` by
thread count), NOT a combiner: it carries the carrier's ``combine_states`` straight onto those
nodes, which realize the combine ALGEBRA via ``render_merge_program`` (the SAME realizer the
scalar ``ScalarCombiner`` / ``Monoid.render`` use). Driven off the carrier's algebra surface
(``carried_names`` / ``combine_operands`` / ``combine_partials``): a scalar ``Accum`` is the
degenerate 1-component monoid (``state=("acc",)``, ``combine_states=(Assign("acc", op,
("acc","acc__o")),)``), so ONE emitter and ONE pair of stmts (``WarpShuffle`` / ``TreeHalve``)
cover both. The combine reassigns the carried state **in place** — no ``_b`` rename, since the
butterfly / tree leaves every thread holding the full reduction in the carried SSA names.

``find_nested_reduce_carriers`` and ``cooperative_combine_geometry`` are the small
queries the materializer uses to locate the carriers and the combine's tid var +
thread count (the whole CTA in the BN=BM=1 form, or each row's BR-lane segment when
free-axis threads ride alongside — strided-cooperative rows).

Pure functions — no shared materializer state. The leading-underscore
module name keeps the pass loader (globs ``*.py``, skips ``_``-prefixed)
from mistaking this for a rule.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel.ir import Smem, Sync, TreeHalve, WarpShuffle
from deplodock.compiler.ir.stmt import Accum, Cond, Monoid, Stmt, Write
from deplodock.compiler.ir.tile.ir import RegisterTile, SerialTile, StridedTile


class Fold(enum.Enum):
    """A combine stage's fold primitive — the hardware mechanism that merges the
    partials at one level of the cross-execution-unit reduction hierarchy.

    - ``SHFL`` — a lane-level register butterfly (``WarpShuffle`` / ``__shfl_xor_sync``).
      The dominant intra-warp scheme (smem-within-a-warp is strictly dominated).
    - ``SMEM`` — a cross-warp / block-wide smem tree-halve (``Smem`` slab + ``Sync`` +
      ``TreeHalve``). Warps cannot shfl across each other, so the across-warps level is
      always SMEM.
    - ``ATOMIC`` — a cross-CTA in-place ``atomicAdd`` (the split-K finalize). Reserved
      for the cross-CTA stage's finalize policy (Milestone 3); ``emit_combine`` (intra-CTA
      only) never emits it.

    The fold is **DERIVED from the execution level**, not tuned: lane → ``SHFL``,
    across-warps → ``SMEM``, cross-CTA → ``ATOMIC``. The level is implied by the stage's
    place in the plan."""

    SHFL = "shfl"
    SMEM = "smem"
    ATOMIC = "atomic"


@dataclass(frozen=True)
class CombineStage:
    """One level of the cross-execution-unit reduction hierarchy: a ``width`` of partials
    folded by a ``fold`` primitive.

    - ``width`` — DERIVED from the partition (the already-tuned cooperative / split-K
      degree); the number of partials this stage merges.
    - ``fold`` — DERIVED from the execution level (see :class:`Fold`).
    - ``kernel_boundary`` — the cross-CTA stage's POLICY knob (Milestone 3): a deferred
      fold in a fresh kernel vs an in-place atomic. Always ``False`` for the intra-CTA
      stages :func:`derive_combine_plan` produces."""

    width: int
    fold: Fold
    kernel_boundary: bool = False


#: The combine as an ordered array of stages, outer→inner, applied after the in-thread
#: ``(serial/fold)`` accumulation. A **derived** structure: assembled from the partition
#: widths + the level→fold derivation, not stored as free per-cell knobs. The product of
#: the intra-CTA widths equals the old ``coop`` thread count.
CombinePlan = tuple[CombineStage, ...]


def derive_combine_plan(n_threads: int, warp_size: int) -> CombinePlan:
    """The intra-CTA :data:`CombinePlan` for an ``n_threads``-wide cooperative partition —
    the typed representation of the geometry ``emit_combine`` used to pick by raw count.

    Reproduces the three historical paths exactly:

    - ``n_threads ≤ warp_size`` → one ``SHFL`` stage (the warp butterfly).
    - ``n_threads`` a multiple of ``warp_size`` → ``SHFL`` (lanes) then ``SMEM`` (the
      ``n_warps`` cross-warp tree) — the hierarchical scheme.
    - otherwise (pow-of-two, not a clean warp multiple) → one ``SMEM`` stage (the
      block-wide tree).

    Requires a power-of-two ``n_threads`` (the butterfly / tree reorders)."""
    if n_threads & (n_threads - 1):
        raise ValueError(f"cross-thread combine needs a power-of-two thread count, got {n_threads}")
    if n_threads <= warp_size:
        return (CombineStage(n_threads, Fold.SHFL),)
    if n_threads % warp_size == 0:
        return (CombineStage(warp_size, Fold.SHFL), CombineStage(n_threads // warp_size, Fold.SMEM))
    return (CombineStage(n_threads, Fold.SMEM),)


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


def emit_combine(carrier, t: str, n_threads: int, *, warp_size: int, barrier_id: int = 0, barrier_count: int | None = None) -> list[Stmt]:
    """Build the **cross-thread distribution** of a reduce carrier's combine across the CTA's threads
    — the butterfly / smem tree (``WarpShuffle`` / ``TreeHalve``). This is pure *geometry*: it carries
    the carrier's ``combine_states`` straight onto the nodes, which realize the combine ALGEBRA
    themselves via ``render_merge_program`` (the SAME realizer the scalar ``ScalarCombiner`` /
    ``Monoid.render`` use). So there is no per-tier "combiner" here — the carrier's
    ``carried_names`` / ``combine_operands`` / ``combine_partials`` surface IS the combine, and
    ``Accum`` is the degenerate 1-component case (``combine_partials`` = the additive ``⊙``).

    The geometry is the derived intra-CTA :data:`CombinePlan` (:func:`derive_combine_plan`):

    - a ``SHFL`` stage → a single ``WarpShuffle`` register butterfly. The XOR butterfly never crosses
      an aligned ``width``-lane group, so a lone ``SHFL`` stage is also the SEGMENTED per-row combine
      for strided-cooperative rows (caller passes the segment size as ``n_threads``).
    - a ``SMEM`` stage **after** a ``SHFL`` stage → the *hierarchical* cross-warp slab: lane-0 of each
      warp writes its broadcast state to a ``smem[width]`` slab per component; one ``Sync`` +
      ``TreeHalve(tid_var="warp")`` collapses across warps and broadcasts.
    - a standalone ``SMEM`` stage → the *block* slab: every thread writes its partial, one ``Sync``,
      a single ``TreeHalve`` reduces + broadcasts in place.

    ``barrier_id`` / ``barrier_count`` route the ``Sync`` + ``TreeHalve`` per-iter sync to a named
    barrier (warp-specialized consumer branch). The warp-shuffle path uses ``__shfl_xor_sync``."""
    state = carrier.carried_names()
    state_b = carrier.combine_operands()
    prog = carrier.combine_partials()
    dtype = getattr(carrier, "dtype", None) or F32

    plan = derive_combine_plan(n_threads, warp_size)

    from deplodock.compiler.backend.cuda.dtype import cuda_name as _cuda_name  # noqa: PLC0415

    smem_c = _cuda_name(dtype)
    bufs = tuple(f"{st}_smem" for st in state)
    out: list[Stmt] = []
    for i, stage in enumerate(plan):
        if stage.fold is Fold.SHFL:
            out.append(WarpShuffle(state=state, state_b=state_b, combine_states=prog, length=stage.width, dtype=dtype))
        elif stage.fold is Fold.SMEM:
            # Two SMEM forms, distinguished by the preceding stage: a cross-warp SMEM after a per-warp
            # SHFL is the *hierarchical* slab — lane-0 of each warp stages its warp's broadcast state,
            # indexed by ``warp``; a standalone SMEM is the *block* slab — every thread stages its own.
            hierarchical = i > 0 and plan[i - 1].fold is Fold.SHFL
            tid_var = "warp" if hierarchical else t
            out += [Smem(name=b, extents=(stage.width,), dtype=smem_c) for b in bufs]
            if hierarchical:
                out.append(
                    Cond(
                        cond=BinaryExpr("==", Var("lane"), Literal(0, "int")),
                        body=tuple(Write(output=b, index=(Var("warp"),), value=st) for b, st in zip(bufs, state, strict=True)),
                    )
                )
            else:
                out += [Write(output=b, index=(Var(tid_var),), value=st) for b, st in zip(bufs, state, strict=True)]
            out.append(Sync(barrier_id=barrier_id, count=barrier_count))
            out.append(
                TreeHalve(
                    bufs=bufs,
                    state=state,
                    state_b=state_b,
                    combine_states=prog,
                    length=stage.width,
                    tid_var=tid_var,
                    dtype=dtype,
                    barrier_id=barrier_id,
                    barrier_count=barrier_count,
                )
            )
    return out
