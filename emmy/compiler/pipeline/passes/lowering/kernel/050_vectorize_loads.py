"""Widen runs of consecutive scalar ``Load`` Stmts into one vector ``Load``.

Until this pass, the body of every materialized kernel carries scalar
``Load`` Stmts. Some sequences of those Loads have a "vector" shape: N
consecutive Loads from the same source buffer whose last-dim indices differ
by 0, 1, ..., N-1. The CUDA backend can emit those as a single
``float<N>`` / ``__half2`` reinterpret-cast read followed by N ``.x/.y/.z/.w``
unpacks. Folding the run into one ``Load(names=(n0, n1, ...), input, index)``
makes the optimization visible in the IR (``--ir kernel`` shows one Load with
multiple LHS names) while keeping the renderer simple — ``Load.render``
branches on the vector form.

## What the pass does

For each ``Body`` (every nested Tile / Loop / StridedLoop / Cond body,
post-order):

1. Walk the stmts. At each position, try widths 8 then 4 then 2.
2. If ``[body[i], ..., body[i+n-1]]`` are all scalar ``Load``s from the
   same input buffer, with matching outer indices, and last-dim indices
   that affinely decompose to ``anchor, anchor+1, ..., anchor+n-1``
   (same coefficients on free vars), AND the target supports
   ``vector_type(elem_dtype, n)`` for the source-buffer dtype, replace
   the run with one widened ``Load``.
3. Otherwise advance one stmt.

## Why this needs the source-buffer dtype

The decision needs the source-buffer dtype, read off the stamped
``Load.dtype`` (``030_stamp_types``). Runs after ``040_demote_to_write_dtype``
so the demote pass sees the original scalar Loads.

## Observed impact

This is an IR-legibility pass, not a perf lever: ptxas coalesces scalar
``ld.shared`` runs once alignment is known, so the vectorized and scalar
source forms compile to identical SASS at every deployable opt level.
``VECTORIZE_LOADS`` is therefore *not* a search dimension — only ``True`` is
enumerated. ``EMMY_VECTORIZE_LOADS=0`` is a manual override.
"""

from __future__ import annotations

from collections.abc import Iterable

from emmy.compiler.backend.cuda.render_target import CudaRenderTarget
from emmy.compiler.graph import Node
from emmy.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, affine_form
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body, Load, Stmt
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.search.space import VECTORIZE_LOADS

PATTERN = [Pattern("root", KernelOp)]

_TARGET = CudaRenderTarget()


def rewrite(root: Node) -> KernelOp | None:
    top: KernelOp = root.op
    # Idempotence: the policy is recorded as the VECTORIZE_LOADS knob, so a
    # re-scan of the rebound op skips here.
    if VECTORIZE_LOADS.name in top.knobs:
        raise RuleSkipped("VECTORIZE_LOADS already decided (idempotence via knob)")
    # Only ``True`` is enumerated, so the autotuner never forks on this knob;
    # ``EMMY_VECTORIZE_LOADS=0`` still pins ``False``.
    if not VECTORIZE_LOADS.narrow((True,))[0]:
        return KernelOp(body=top.body, name=top.name, knobs={**top.knobs, VECTORIZE_LOADS.name: False})
    # Stamp the policy (True) even when no run is foldable — the realized config
    # records that vectorization was enabled, keeping a uniform knob set.
    new_body = _vectorize_body(top, top.body)
    return KernelOp(body=new_body, name=top.name, knobs={**top.knobs, VECTORIZE_LOADS.name: True})


def _vectorize_body(top: KernelOp, body: Body) -> Body:
    """Post-order body transform: recurse into nested bodies first, then
    scan this scope for consecutive-Load runs. Threads ``top`` through so
    constant-input filtering can resolve against the surrounding op."""
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


def _try_vec_load(stmts: Iterable[Stmt], start: int, n: int, top: KernelOp) -> Load | None:
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
    # exactly k for the k-th load.
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
    # be a multiple of n. n=2 fp16 is NOT a freebie despite __half2's 4-byte
    # type alignment — an odd-element offset still misses the alignment and
    # faults with CUDA_ERROR_MISALIGNED_ADDRESS.
    if n >= 2:
        if not all(c % n == 0 for c in coeffs_0.values()):
            return None
        anchor_simplified = anchor_0.simplify(SimplifyCtx.empty())
        if not isinstance(anchor_simplified, Literal) or anchor_simplified.value % n != 0:
            return None

    return Load(
        names=tuple(s.name for s in loads),
        input=input_name,
        index=loads[0].index,
        dtype=loads[0].dtype,
    )
