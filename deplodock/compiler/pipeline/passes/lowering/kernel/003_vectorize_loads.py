"""Replace runs of consecutive ``Load`` Stmts with one :class:`VecLoad`.

Until this pass, the body of every Kernel-IR Tile carries scalar
``Load`` Stmts. Some sequences of those Loads have a "vector" shape: N
consecutive Loads from the same source buffer whose last-dim indices
differ by 0, 1, ..., N-1. The CUDA backend can emit those as a single
``float<N>`` / ``__half2`` reinterpret-cast read followed by N .x/.y/.z/.w
unpacks. The fold used to live inside ``render_body`` as a string-level
fast path; lifting it here makes the optimization visible in the IR
(``--ir kernel`` shows a single ``VecLoad`` line where there used to be
N Loads), keeps the renderer simple, and gives us a home for the
upcoming ``__half2`` accumulator packing.

## What the pass does

For each ``Body`` (Tile body and every nested Loop / StridedLoop / Cond /
Tile body, post-order):

1. Walk the stmts. At each position, try widths 4 then 2.
2. If ``[body[i], ..., body[i+n-1]]`` are all ``Load``s from the same
   input buffer, with matching outer indices, and last-dim indices
   that affinely decompose to ``anchor, anchor+1, ..., anchor+n-1``
   (same coefficients on free vars), AND the target supports
   ``vector_type(elem_dtype, n)`` for the source-buffer dtype, replace
   the run with one ``VecLoad(names=(n0..n_{n-1}), input, base_index,
   elem_dtype)``.
3. Otherwise advance one stmt.

## Why this lives at the Kernel-IR boundary

The decision needs the source-buffer dtype (``KernelOp.input_tensors``
+ ``Smem.dtype`` for smem buffers). Body alone doesn't carry that
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
from dataclasses import replace as _replace

from deplodock.compiler.backend.cuda.dtype import canonical_from_cuda_name
from deplodock.compiler.backend.cuda.render_target import CudaRenderTarget
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.base import ConstantOp
from deplodock.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, affine_form
from deplodock.compiler.ir.kernel import KernelOp
from deplodock.compiler.ir.kernel.ir import Smem
from deplodock.compiler.ir.stmt import Body, Cond, Load, Loop, Stmt, StridedLoop, Tile, VecLoad
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]

_TARGET = CudaRenderTarget()


def rewrite(match: Match, root: Node) -> Graph | None:
    kop: KernelOp = root.op

    # Per-buffer dtype map: graph buffers from input_tensors /
    # output_tensors (set by 001_lower_kernelop downstream — empty
    # here), then graph fallback, then Smem dtypes from the body.
    buf_dtypes: dict[str, str] = {}
    for name, t in {**kop.input_tensors, **kop.output_tensors}.items():
        buf_dtypes[name] = t.dtype.name
    for nid in kop.inputs:
        node = match.graph.nodes.get(nid)
        if node is not None:
            buf_dtypes.setdefault(nid, node.output.dtype.name)
    for out in kop.outputs:
        node = match.graph.nodes.get(out)
        if node is not None:
            buf_dtypes.setdefault(out, node.output.dtype.name)
    for s in kop.body.iter():
        if isinstance(s, Smem):
            canonical = canonical_from_cuda_name(s.dtype)
            if canonical is not None:
                buf_dtypes[s.name] = canonical

    # Loads from scalar ``ConstantOp`` inputs get inlined as float
    # literals at CUDA lowering (``001_lower_kernelop`` populates
    # ``literal_constants`` and the surrounding kernel doesn't even
    # take that buffer as a parameter). We must not vectorize Loads
    # from such buffers — the reinterpret_cast would reference a name
    # that doesn't exist in the rendered kernel.
    literal_const_bufs: set[str] = set()
    for bid in kop.inputs:
        node = match.graph.nodes.get(bid)
        if node is not None and isinstance(node.op, ConstantOp) and node.op.value is not None:
            literal_const_bufs.add(bid)

    new_body = _vectorize_body(kop.body, buf_dtypes, literal_const_bufs)
    if new_body == kop.body:
        raise RuleSkipped("no vectorizable Load runs found")

    return KernelOp(
        body=new_body,
        name=kop.name,
        input_tensors=dict(kop.input_tensors),
        output_tensors=dict(kop.output_tensors),
    )


def _vectorize_body(body: Body, buf_dtypes: dict[str, str], literal_const_bufs: set[str]) -> Body:
    """Post-order body transform: recurse into nested bodies first, then
    scan this scope for consecutive-Load runs."""
    descended: list[Stmt] = []
    for s in body:
        if isinstance(s, Loop):
            descended.append(_replace(s, body=_vectorize_body(s.body, buf_dtypes, literal_const_bufs)))
        elif isinstance(s, StridedLoop):
            descended.append(_replace(s, body=_vectorize_body(s.body, buf_dtypes, literal_const_bufs)))
        elif isinstance(s, Cond):
            descended.append(
                Cond(
                    cond=s.cond,
                    body=_vectorize_body(s.body, buf_dtypes, literal_const_bufs),
                    else_body=_vectorize_body(s.else_body, buf_dtypes, literal_const_bufs),
                )
            )
        elif isinstance(s, Tile):
            descended.append(Tile(axes=s.axes, body=_vectorize_body(s.body, buf_dtypes, literal_const_bufs)))
        else:
            descended.append(s)

    out: list[Stmt] = []
    i = 0
    while i < len(descended):
        replaced = False
        for run_n in (8, 4, 2):
            vec = _try_vec_load(descended, i, run_n, buf_dtypes, literal_const_bufs)
            if vec is not None:
                out.append(vec)
                i += run_n
                replaced = True
                break
        if not replaced:
            out.append(descended[i])
            i += 1
    return Body(tuple(out))


def _try_vec_load(stmts: Iterable[Stmt], start: int, n: int, buf_dtypes: dict[str, str], literal_const_bufs: set[str]) -> VecLoad | None:
    """If ``stmts[start:start+n]`` matches the consecutive-Load pattern
    and the target supports ``vector_type(elem_dtype, n)`` for the
    source buffer's dtype, return the replacement :class:`VecLoad`.
    Otherwise return ``None``."""
    stmts_list = list(stmts)
    if start + n > len(stmts_list):
        return None
    loads = stmts_list[start : start + n]
    if not all(isinstance(s, Load) for s in loads):
        return None
    # No literal-constant loads (those render as embedded scalar floats).
    if any(getattr(s, "input", None) is None for s in loads):
        return None

    inputs = {s.input for s in loads}
    if len(inputs) != 1:
        return None
    (input_name,) = inputs
    if input_name in literal_const_bufs:
        # Scalar-constant inputs get inlined at CUDA lowering — the
        # surrounding kernel doesn't take that buffer as a parameter,
        # so a vectorized reinterpret_cast would reference an undefined
        # symbol.
        return None
    src_dt = buf_dtypes.get(input_name, "f32")
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

    return VecLoad(
        names=tuple(s.name for s in loads),
        input=input_name,
        base_index=loads[0].index,
        elem_dtype=src_dt,
    )
