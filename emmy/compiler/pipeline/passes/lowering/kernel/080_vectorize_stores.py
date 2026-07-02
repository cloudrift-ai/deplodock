"""Widen runs of consecutive scalar ``Write`` Stmts into one vector ``Write``.

Symmetric to ``050_vectorize_loads``. Until this pass, the body of every
materialized kernel carries scalar ``Write`` Stmts. Some sequences have a
"vector" shape: N consecutive Writes to the same output buffer whose last-dim
indices differ by 0, 1, ..., N-1. The CUDA backend can emit those as one
``make_<vec_type>(...)`` + one
``*reinterpret_cast<<vec_type>*>(&buf[base]) = packed;`` transaction.

## What the pass does

For each ``Body`` (every nested Tile / Loop / StridedLoop / Cond body,
post-order):

1. Walk the stmts. At each position, try widths 8 then 4 then 2.
2. If ``[body[i], ..., body[i+n-1]]`` are all scalar ``Write``s to the same
   output buffer, matching outer indices, and last-dim indices that affinely
   decompose to ``anchor, anchor+1, ..., anchor+n-1`` (same coefficients on
   free vars), AND the target supports ``vector_type(elem_dtype, n)`` for the
   destination-buffer dtype, replace the run with one widened ``Write``.
3. Otherwise advance one stmt.

The destination-buffer dtype comes from the op's ``outputs`` / ``inputs``
(matcher-populated graph Tensors).

NOTE: atomic reduce-writes must NOT vectorize (each lane needs its own
``atomicAdd``). The scalar tier emits no atomic writes — cross-CTA reduction
is future work — so the atomic guard is currently a no-op (``atomic_write_ids``
empty); it needs rebuilding when split-K / split-reduce returns.
"""

from __future__ import annotations

from collections.abc import Iterable

from emmy.compiler.backend.cuda.render_target import CudaRenderTarget
from emmy.compiler.graph import Node
from emmy.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, affine_form
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body, Stmt, Write
from emmy.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", KernelOp)]

_TARGET = CudaRenderTarget()


def rewrite(root: Node) -> KernelOp | None:
    top: KernelOp = root.op
    new_body = _vectorize_body(top, top.body)
    if new_body == top.body:
        raise RuleSkipped("no vectorizable Write runs found")
    return KernelOp(body=new_body, name=top.name, knobs=dict(top.knobs))


def _buf_dtype(top: KernelOp, name: str) -> str:
    """Resolve a destination buffer's canonical dtype. Writes target either a
    kernel output or, in rare cases, a kernel input (when an optimization
    folds a copy). Falls back to f32 when not found."""
    t = top.outputs.get(name) or top.inputs.get(name)
    if t is not None:
        return t.dtype.name
    return "f32"


def _vectorize_body(top: KernelOp, body: Body) -> Body:
    """Post-order body transform: recurse into nested bodies first, then
    scan this scope for consecutive-Write runs. Threads ``top`` through so
    ``_buf_dtype`` can resolve per-buffer dtypes against the same op."""
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
            vec = _try_vec_store(descended, i, run_n, top)
            if vec is not None:
                out.append(vec)
                i += run_n
                replaced = True
                break
        if not replaced:
            out.append(descended[i])
            i += 1
    return Body(tuple(out))


def _try_vec_store(stmts: Iterable[Stmt], start: int, n: int, top: KernelOp) -> Write | None:
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
    # Already-widened Writes in the run aren't safe to re-merge — bail. Atomic reduce-writes
    # never vectorize (each contributing lane needs its own ``atomicAdd``).
    if any(s.is_vector or s.atomic for s in writes):
        return None

    outputs = {s.output for s in writes}
    if len(outputs) != 1:
        return None
    (output_name,) = outputs

    dst_dt = _buf_dtype(top, output_name)
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
    # exactly k for the k-th write. Mirrors ``050_vectorize_loads``.
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

    # Alignment proof: the reinterpret-cast destination must be aligned to
    # ``n * elem_bytes``. Same as ``050_vectorize_loads``: every free-var
    # coefficient on the last dim must be a multiple of n, and the literal
    # anchor must also be a multiple of n.
    if n >= 2:
        if not all(c % n == 0 for c in coeffs_0.values()):
            return None
        anchor_simplified = anchor_0.simplify(SimplifyCtx.empty())
        if not isinstance(anchor_simplified, Literal) or anchor_simplified.value % n != 0:
            return None

    return Write(
        output=output_name,
        index=writes[0].index,
        values=tuple(s.value for s in writes),
        value_dtype=writes[0].value_dtype,
    )
