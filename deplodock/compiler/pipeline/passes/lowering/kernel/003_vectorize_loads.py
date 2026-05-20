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
after ``001_materialize_tile`` has placed the Smem decls and before
the CUDA-source rendering in ``lowering/cuda`` — keeps both pieces of
context available in one place.

## Composition with the demote pass

Runs after ``002_demote_to_write_dtype`` so the demote pass sees the
original scalar Loads (the demote analysis is on Assigns, not Loads,
so order is mostly independent; this is the conservative ordering).
"""

from __future__ import annotations

from collections.abc import Iterable

from deplodock.compiler.backend.cuda.dtype import canonical_from_cuda_name
from deplodock.compiler.backend.cuda.render_target import CudaRenderTarget
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, affine_form
from deplodock.compiler.ir.kernel import KernelOp
from deplodock.compiler.ir.stmt import Body, Load, Stmt
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]

_TARGET = CudaRenderTarget()


def rewrite(match: Match, root: Node) -> Graph | None:  # noqa: ARG001 — match required by rule dispatch signature
    kop: KernelOp = root.op
    new_body = _vectorize_body(kop, kop.body)
    if new_body == kop.body:
        raise RuleSkipped("no vectorizable Load runs found")
    return KernelOp(body=new_body, name=kop.name)


def _buf_dtype(kop: KernelOp, name: str) -> str:
    """Resolve a buffer's canonical dtype: matcher-populated ``kop.inputs``
    / ``kop.outputs`` for graph buffers, then ``kop.smem_buffers`` for
    smem-local names, then ``f32``."""
    t = kop.inputs.get(name) or kop.outputs.get(name)
    if t is not None:
        return t.dtype.name
    smem = kop.smem_buffers.get(name)
    if smem is not None:
        canonical = canonical_from_cuda_name(smem.dtype)
        if canonical is not None:
            return canonical
    return "f32"


def _vectorize_body(kop: KernelOp, body: Body) -> Body:
    """Post-order body transform: recurse into nested bodies first, then
    scan this scope for consecutive-Load runs. Threads ``kop`` through so
    ``_buf_dtype`` can resolve per-buffer dtypes against the same op."""
    descended: list[Stmt] = []
    for s in body:
        nested = s.nested()
        if nested:
            descended.append(s.with_bodies(tuple(_vectorize_body(kop, b) for b in nested)))
        else:
            descended.append(s)

    out: list[Stmt] = []
    i = 0
    while i < len(descended):
        replaced = False
        for run_n in (8, 4, 2):
            vec = _try_vec_load(descended, i, run_n, kop)
            if vec is not None:
                out.append(vec)
                i += run_n
                replaced = True
                break
        if not replaced:
            out.append(descended[i])
            i += 1
    return Body(tuple(out))


def _try_vec_load(stmts: Iterable[Stmt], start: int, n: int, kop: KernelOp) -> Load | None:
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

    inputs = {s.input for s in loads}
    if len(inputs) != 1:
        return None
    (input_name,) = inputs
    src_tensor = kop.inputs.get(input_name)
    if src_tensor is not None and src_tensor.constant and src_tensor.value is not None:
        # Scalar-constant inputs get inlined at CUDA lowering — the
        # surrounding kernel doesn't take that buffer as a parameter,
        # so a vectorized reinterpret_cast would reference an undefined
        # symbol.
        return None
    src_dt = _buf_dtype(kop, input_name)
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

    # For widths above the natural 4-byte (one __half2 / one float)
    # boundary, the reinterpret-cast destination must be aligned to
    # ``n * elem_bytes``. Prove statically from the affine form: every
    # free-var coefficient on the last dim must be a multiple of n,
    # and the literal anchor must also be a multiple of n. (n=2 fp16
    # = 4 bytes always works since __half2 is 4-byte aligned in cuda_fp16.h.)
    if n > 2 or (n == 2 and src_dt == "f32"):
        if not all(c % n == 0 for c in coeffs_0.values()):
            return None
        anchor_simplified = anchor_0.simplify(SimplifyCtx.empty())
        if not isinstance(anchor_simplified, Literal) or anchor_simplified.value % n != 0:
            return None

    return Load(
        names=tuple(s.name for s in loads),
        input=input_name,
        index=loads[0].index,
    )
