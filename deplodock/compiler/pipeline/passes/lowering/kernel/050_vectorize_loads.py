"""Widen runs of consecutive scalar ``Load`` Stmts into one vector ``Load``.

Until this pass, the body of every Kernel-IR Tile carries scalar
``Load`` Stmts (``extra_names=()``). Some sequences of those Loads have
a "vector" shape: N consecutive Loads from the same source buffer whose
last-dim indices differ by 0, 1, ..., N-1. The CUDA backend can emit
those as a single ``float<N>`` / ``__half2`` reinterpret-cast read
followed by N ``.x/.y/.z/.w`` unpacks. Folding the run into one
``Load(name=n0, extra_names=(n1..n_{N-1}), input, index)`` makes the
optimization visible in the IR (``--ir kernel`` shows one Load with
multiple LHS names) while keeping the renderer simple — ``Load.render``
branches on ``extra_names`` to emit either the scalar or the vector
form.

## What the pass does

For each ``Body`` (Tile body and every nested Loop / StridedLoop / Cond /
Tile body, post-order):

1. Walk the stmts. At each position, try widths 8 then 4 then 2.
2. If ``[body[i], ..., body[i+n-1]]`` are all scalar ``Load``s from the
   same input buffer, with matching outer indices, and last-dim indices
   that affinely decompose to ``anchor, anchor+1, ..., anchor+n-1``
   (same coefficients on free vars), AND the target supports
   ``vector_type(elem_dtype, n)`` for the source-buffer dtype, replace
   the run with one widened ``Load``.
3. Otherwise advance one stmt.

## Why this lives at the Kernel-IR boundary

The decision needs the source-buffer dtype (graph node dtypes for
graph dtypes via ``KernelOp.inputs`` keys + ``Smem.dtype`` for smem buffers). Body alone doesn't carry that
info, so ``normalize_body`` (which runs on every Body construction)
can't make the call without external context. Running the pass here —
after ``100_materialize_tile`` has placed the Smem decls and before
the CUDA-source rendering in ``lowering/cuda`` — keeps both pieces of
context available in one place.

## Composition with the demote pass

Runs after ``040_demote_to_write_dtype`` so the demote pass sees the
original scalar Loads (the demote analysis is on Assigns, not Loads,
so order is mostly independent; this is the conservative ordering).

## Observed impact (RTX 5090, nvcc/ptxas 13.0)

Like ``095_interleave_loads``, this is an IR-legibility pass, not a perf
lever — and ``cuobjdump`` says exactly why. The vectorized and scalar source
forms compile to **identical SASS** at every deployable opt level: the
scalar-source kernel shows the same two ``LDS.128`` as the ``float4`` source.
ptxas does its own load coalescing once it knows the static smem layout and
16-byte alignment, so a manual ``float4`` only re-states an alignment it can
already prove. Measured latency is therefore **0 %** across the whole
tile-intensity spectrum (load-bound ``FM=1, FN=4`` through compute-bound
``FM=FN=8``).

What flag makes the pass change the SASS? Only ``-Xptxas -O0`` — the one level
that disables the coalescer. There the scalar source stays scalar (0
``LDS.128``) while the ``float4`` source keeps its 2; from ``-O1`` up both
coalesce to the same SASS:

    -Xptxas -O   scalar-source LDS.128   float4-source LDS.128
    -O0          0                       2
    -O1/-O2/-O3  2                       2

``-O0`` is never deployable, and even there the load width is not this matmul's
bottleneck, so the latency is unchanged regardless. Manual coalescing is still
a real win on hand-written kernels where ptxas *cannot* prove alignment, where
the access chain has gaps, or where pointers may alias without
``const __restrict__`` — none of which a statically-shaped tile hits.
(Mechanism: ptxas auto-vectorizes scalar ``ld.shared`` runs once alignment is
known — https://github.com/JuliaGPU/CUDA.jl/issues/68 ; the cast is purely an
alignment promise — https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/ .)

The pass folds the runs anyway so ``--ir kernel`` / ``--ir cuda`` show one
wide ``Load`` matching the hand-written SGEMM shape. ``VECTORIZE_LOADS`` is
*not* a search dimension — only ``True`` is enumerated, so the autotuner never
forks on it. ``DEPLODOCK_VECTORIZE_LOADS=0`` is a manual override for the
scalar-load form.
"""

from __future__ import annotations

from collections.abc import Iterable

from deplodock.compiler.backend.cuda.render_target import CudaRenderTarget
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, affine_form
from deplodock.compiler.ir.stmt import Body, Load, Stmt
from deplodock.compiler.ir.tile.ir import Source, Stage, StageBundle, TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType

PATTERN = [Pattern("root", TileOp)]

_TARGET = CudaRenderTarget()

VECTORIZE_LOADS = Knob(
    "VECTORIZE_LOADS",
    KnobType.BOOL,
    hints=(True,),  # on by default; not a search dimension — manual override only via the env var
    help="Fold runs of consecutive scalar Loads into one wide vector Load (float4 / __half2).",
)


def rewrite(match: Match, root: Node) -> Graph | None:  # noqa: ARG001 — match required by rule dispatch signature
    # Only ``True`` is enumerated, so the autotuner never forks on this knob;
    # ``DEPLODOCK_VECTORIZE_LOADS=0`` still pins ``False`` (``narrow`` honours an env
    # pin authoritatively, even when it is not in the candidate set).
    if not VECTORIZE_LOADS.narrow((True,))[0]:
        raise RuleSkipped("VECTORIZE_LOADS=0 pinned")
    top: TileOp = root.op
    new_body = _vectorize_body(top, top.body)
    if new_body == top.body:
        raise RuleSkipped("no vectorizable Load runs found")
    return TileOp(body=new_body, name=top.name)


def _vectorize_body(top: TileOp, body: Body) -> Body:
    """Post-order body transform: recurse into nested bodies first, then
    scan this scope for consecutive-Load runs. Threads ``top`` through so
    constant-input filtering can resolve against the surrounding TileOp."""
    descended: list[Stmt] = []
    for s in body:
        nested = s.nested()
        if nested:
            descended.append(s.with_bodies(tuple(_vectorize_body(top, b) for b in nested)))
        else:
            descended.append(s)

    out: list[Stmt] = []
    i = 0
    while i < len(descended):
        replaced = False
        for run_n in (8, 4, 2):
            vec = _try_vec_load(descended, i, run_n, top)
            if vec is not None:
                out.append(vec)
                i += run_n
                replaced = True
                break
        if not replaced:
            out.append(descended[i])
            i += 1
    return Body(tuple(out))


def _try_vec_load(stmts: Iterable[Stmt], start: int, n: int, top: TileOp) -> Load | None:
    """If ``stmts[start:start+n]`` matches the consecutive-Load pattern
    and the target supports ``vector_type(elem_dtype, n)`` for the
    source buffer's dtype, return the widened :class:`Load`. Otherwise
    return ``None``."""
    stmts_list = list(stmts)
    if start + n > len(stmts_list):
        return None
    loads = stmts_list[start : start + n]
    if not all(isinstance(s, Load) for s in loads):
        return None
    # Already-widened Loads in the run aren't safe to re-merge — bail.
    if any(s.is_vector for s in loads):
        return None
    # No literal-constant loads (those render as embedded scalar floats).
    if any(getattr(s, "input", None) is None for s in loads):
        return None
    # Every Load in the run must carry a stamped dtype (set by
    # ``030_stamp_types``). If not, bail — the source dtype is the
    # decision point for picking a vector type, and falling back to f32
    # would silently mis-vectorize fp16 chains.
    if any(s.dtype is None for s in loads):
        return None

    inputs = {s.input for s in loads}
    if len(inputs) != 1:
        return None
    (input_name,) = inputs
    src_tensor = top.inputs.get(input_name)
    if src_tensor is not None and src_tensor.constant and src_tensor.value is not None:
        # Scalar-constant inputs get inlined at CUDA lowering — the
        # surrounding kernel doesn't take that buffer as a parameter,
        # so a vectorized reinterpret_cast would reference an undefined
        # symbol.
        return None
    src_dt = loads[0].dtype.name
    if _TARGET.vector_type(src_dt, n) is None:
        return None

    # Same rank, same outer indices.
    rank = len(loads[0].index)
    if rank == 0 or any(len(s.index) != rank for s in loads[1:]):
        return None
    outer = loads[0].index[:-1]
    for s in loads[1:]:
        if s.index[:-1] != outer:
            return None

    # Last-dim indices: same free-var coefficients, anchor differs by
    # exactly k for the k-th load. Same affine-form check that
    # ``_vec_load_run`` used in the previous rendering-side fast path.
    inner_0 = loads[0].index[-1]
    free = inner_0.free_vars()
    for s in loads[1:]:
        free = free | s.index[-1].free_vars()
    af0 = affine_form(inner_0, free)
    if af0 is None:
        return None
    anchor_0, coeffs_0 = af0
    for k, s in enumerate(loads):
        if k == 0:
            continue
        af = affine_form(s.index[-1], free)
        if af is None:
            return None
        anchor_k, coeffs_k = af
        if coeffs_k != coeffs_0:
            return None
        diff = BinaryExpr("-", anchor_k, anchor_0).simplify(SimplifyCtx.empty())
        if not (isinstance(diff, Literal) and isinstance(diff.value, int) and diff.value == k):
            return None

    # The reinterpret-cast destination must be aligned to ``n * elem_bytes``.
    # Prove statically from the affine form: every free-var coefficient on
    # the last dim must be a multiple of n, and the literal anchor must also
    # be a multiple of n.
    #
    # Earlier this check was skipped for n=2 fp16 with the comment "__half2
    # is 4-byte aligned in cuda_fp16.h" — the TYPE is, but reinterpreting a
    # fp16 pointer at an odd-element offset still misses the alignment.
    # Per-cell matmul shapes with FN > 1 expose this: an N-stride of 3
    # (lm_head-style vocab=3 test case) gives a3 a stride of 3 elements =
    # 6 bytes — not a multiple of 4 — and the half2 read faults with
    # CUDA_ERROR_MISALIGNED_ADDRESS.
    if n >= 2:
        if not all(c % n == 0 for c in coeffs_0.values()):
            return None
        anchor_simplified = anchor_0.simplify(SimplifyCtx.empty())
        if not isinstance(anchor_simplified, Literal) or anchor_simplified.value % n != 0:
            return None

        # Multi-dim staged smem case: the last logical dim may be a constant
        # offset (cell index 0/1/...) while the BASE address ``Σ outer_dim ×
        # outer_stride`` carries the per-cell stride packed into preceding
        # dims (e.g. blocked-N smem with layout ``[K, N_t, N_r]`` and FN=3
        # has stride ``a3 * 3`` from the N_t dim; packing two cells across
        # the inner N_r yields byte offsets ``base + 0`` and ``base + 4``,
        # but ``base = a3 * 12`` lands at 12 bytes for a3=1 — not 8-byte
        # aligned for float2). The last-dim coeff check passes vacuously
        # (no free vars there), so the broader check has to walk back into
        # the preceding logical dims via the Source's cache_dims. If any
        # preceding dim's cache-axis extent is not a multiple of n at the
        # innermost slot (= the byte stride from the cell-offset Load
        # group's base to its neighbour group isn't n-aligned), refuse to
        # vectorize.
        source = _find_source(top.body, input_name)
        if source is not None and len(source.cache_dims) >= 2:
            inner_extent = source.cache_axes[-1].extent.as_static()
            if inner_extent % n != 0:
                return None

    return Load(
        names=tuple(s.name for s in loads),
        input=input_name,
        index=loads[0].index,
        dtype=loads[0].dtype,
    )


def _find_source(body: Body, name: str) -> Source | None:
    """Walk ``body`` for a ``Source`` whose ``name`` matches — used by
    multi-dim staged-smem alignment checks. Returns the first match
    (sources are unique by name within a TileOp by construction) or
    ``None`` if the buffer isn't a Stage-backed smem (e.g. a gmem input)."""
    for stmt in body.iter():
        if isinstance(stmt, Stage):
            for src in stmt.sources:
                if src.name == name:
                    return src
        elif isinstance(stmt, StageBundle):
            for stage in stmt.stages:
                for src in stage.sources:
                    if src.name == name:
                        return src
    return None
