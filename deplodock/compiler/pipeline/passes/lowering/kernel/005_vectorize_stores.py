"""Widen runs of consecutive scalar ``Write`` Stmts into one vector ``Write``.

Symmetric to ``003_vectorize_loads``. Until this pass, the body of every
Kernel-IR Tile carries scalar ``Write`` Stmts (``extra_values=()``).
Some sequences have a "vector" shape: N consecutive Writes to the same
output buffer whose last-dim indices differ by 0, 1, ..., N-1. The CUDA
backend can emit those as one ``make_<vec_type>(...)`` + one
``*reinterpret_cast<<vec_type>*>(&buf[base]) = packed;`` transaction.

## What the pass does

For each ``Body`` (Tile body and every nested Loop / StridedLoop / Cond /
Tile body, post-order):

1. Walk the stmts. At each position, try widths 8 then 4 then 2.
2. If ``[body[i], ..., body[i+n-1]]`` are all scalar ``Write``s to the
   same output buffer with ``reduce_op is None`` (atomic reduce-writes
   do NOT vectorize), matching outer indices, and last-dim indices that
   affinely decompose to ``anchor, anchor+1, ..., anchor+n-1`` (same
   coefficients on free vars), AND the target supports
   ``vector_type(elem_dtype, n)`` for the destination-buffer dtype,
   replace the run with one widened ``Write``.
3. Otherwise advance one stmt.

## Why this lives at the Kernel-IR boundary

Same reason as ``003_vectorize_loads``: the decision needs the
destination-buffer dtype, which comes from ``KernelOp.output_tensors``
or the graph node's dtype. Running here keeps both pieces of context in
one place and ensures the IR-visible form (``--ir kernel``) reflects
the optimization.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace as _replace

from deplodock.compiler.backend.cuda.dtype import canonical_from_cuda_name
from deplodock.compiler.backend.cuda.render_target import CudaRenderTarget
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, affine_form
from deplodock.compiler.ir.kernel import KernelOp
from deplodock.compiler.ir.kernel.ir import Smem
from deplodock.compiler.ir.stmt import Body, Cond, Loop, Stmt, StridedLoop, Tile, Write
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]

_TARGET = CudaRenderTarget()


def rewrite(match: Match, root: Node) -> Graph | None:
    kop: KernelOp = root.op

    # Per-buffer dtype map: graph buffers from input_tensors /
    # output_tensors, then graph fallback, then Smem dtypes from the
    # body. Mirror ``003_vectorize_loads`` so we cover smem writebacks
    # too even though the common case is the kernel's output buffer.
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

    new_body = _vectorize_body(kop.body, buf_dtypes)
    if new_body == kop.body:
        raise RuleSkipped("no vectorizable Write runs found")

    return KernelOp(
        body=new_body,
        name=kop.name,
        input_tensors=dict(kop.input_tensors),
        output_tensors=dict(kop.output_tensors),
    )


def _vectorize_body(body: Body, buf_dtypes: dict[str, str]) -> Body:
    """Post-order body transform: recurse into nested bodies first, then
    scan this scope for consecutive-Write runs."""
    descended: list[Stmt] = []
    for s in body:
        if isinstance(s, Loop):
            descended.append(_replace(s, body=_vectorize_body(s.body, buf_dtypes)))
        elif isinstance(s, StridedLoop):
            descended.append(_replace(s, body=_vectorize_body(s.body, buf_dtypes)))
        elif isinstance(s, Cond):
            descended.append(
                Cond(
                    cond=s.cond,
                    body=_vectorize_body(s.body, buf_dtypes),
                    else_body=_vectorize_body(s.else_body, buf_dtypes),
                )
            )
        elif isinstance(s, Tile):
            descended.append(Tile(axes=s.axes, body=_vectorize_body(s.body, buf_dtypes)))
        else:
            descended.append(s)

    out: list[Stmt] = []
    i = 0
    while i < len(descended):
        replaced = False
        for run_n in (8, 4, 2):
            vec = _try_vec_store(descended, i, run_n, buf_dtypes)
            if vec is not None:
                out.append(vec)
                i += run_n
                replaced = True
                break
        if not replaced:
            out.append(descended[i])
            i += 1
    return Body(tuple(out))


def _try_vec_store(stmts: Iterable[Stmt], start: int, n: int, buf_dtypes: dict[str, str]) -> Write | None:
    """If ``stmts[start:start+n]`` matches the consecutive-Write pattern
    and the target supports ``vector_type(elem_dtype, n)`` for the
    destination buffer's dtype, return the widened :class:`Write`.
    Otherwise return ``None``."""
    stmts_list = list(stmts)
    if start + n > len(stmts_list):
        return None
    writes = stmts_list[start : start + n]
    if not all(isinstance(s, Write) for s in writes):
        return None
    # Already-widened Writes in the run aren't safe to re-merge — bail.
    if any(s.is_vector for s in writes):
        return None

    outputs = {s.output for s in writes}
    if len(outputs) != 1:
        return None
    (output_name,) = outputs

    # Atomic reduce-writes cannot be vectorized (each lane needs its
    # own atomicAdd). Require ``reduce_op is None`` for the whole run.
    if any(s.reduce_op is not None for s in writes):
        return None

    dst_dt = buf_dtypes.get(output_name, "f32")
    if _TARGET.vector_type(dst_dt, n) is None:
        return None

    # Same rank, same outer indices.
    rank = len(writes[0].index)
    if rank == 0 or any(len(s.index) != rank for s in writes[1:]):
        return None
    outer = writes[0].index[:-1]
    for s in writes[1:]:
        if s.index[:-1] != outer:
            return None

    # Last-dim indices: same free-var coefficients, anchor differs by
    # exactly k for the k-th write. Mirrors the affine-form check in
    # ``003_vectorize_loads._try_vec_load``.
    inner_0 = writes[0].index[-1]
    free = inner_0.free_vars()
    for s in writes[1:]:
        free = free | s.index[-1].free_vars()
    af0 = affine_form(inner_0, free)
    if af0 is None:
        return None
    anchor_0, coeffs_0 = af0
    for k, s in enumerate(writes):
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

    # Alignment proof: for widths above the natural 4-byte boundary,
    # the reinterpret-cast destination must be aligned to
    # ``n * elem_bytes``. Same as ``003_vectorize_loads``: every
    # free-var coefficient on the last dim must be a multiple of n,
    # and the literal anchor must also be a multiple of n.
    if n > 2 or (n == 2 and dst_dt == "f32"):
        if not all(c % n == 0 for c in coeffs_0.values()):
            return None
        anchor_simplified = anchor_0.simplify(SimplifyCtx.empty())
        if not isinstance(anchor_simplified, Literal) or anchor_simplified.value % n != 0:
            return None

    return Write(
        output=output_name,
        index=writes[0].index,
        values=tuple(s.value for s in writes),
        reduce_op=None,
    )
