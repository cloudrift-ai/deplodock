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
destination-buffer dtype, which comes from graph node dtypes for ``KernelOp.outputs`` keys
or the graph node's dtype. Running here keeps both pieces of context in
one place and ensures the IR-visible form (``--ir kernel``) reflects
the optimization.
"""

from __future__ import annotations

from collections.abc import Iterable

from deplodock.compiler.backend.cuda.render_target import CudaRenderTarget
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, affine_form
from deplodock.compiler.ir.stmt import Body, Cond, Stmt, Write
from deplodock.compiler.ir.tile.ir import GridTile, RegisterTile, SerialTile, Stage, StridedTile, ThreadTile, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]

_TARGET = CudaRenderTarget()


def rewrite(root: Node) -> Graph | None:
    top: TileOp = root.op
    new_body = _vectorize_body(top, top.body)
    if new_body == top.body:
        raise RuleSkipped("no vectorizable Write runs found")
    return TileOp(body=new_body, name=top.name, knobs=dict(top.knobs))


def _buf_dtype(top: TileOp, name: str) -> str:
    """Resolve a destination buffer's canonical dtype. TileOp-stage
    Writes target either a kernel output (Tensor in ``top.outputs``)
    or, in rare cases, a kernel input (when downstream optimizations
    fold a copy). Falls back to f32 when not found."""
    t = top.outputs.get(name) or top.inputs.get(name)
    if t is not None:
        return t.dtype.name
    return "f32"


def _vectorize_body(top: TileOp, body: Body) -> Body:
    """Post-order body transform: recurse into nested bodies first, then
    scan this scope for consecutive-Write runs. Threads ``top`` through
    so ``_buf_dtype`` can resolve per-buffer dtypes against the same op.

    Stage Writes belong to the *producer* side, synthesized at materialize
    time — they aren't in the TileOp body yet, so the Stage block stmts
    are walked as opaque (their consumer subtree's Writes target the
    kernel output, not the smem slab)."""
    descended: list[Stmt] = []
    for s in body:
        if isinstance(s, (SerialTile, StridedTile, RegisterTile, Stage)):
            new_nested = tuple(_vectorize_body(top, b) for b in s.nested())
            descended.append(s.with_bodies(new_nested))
        elif isinstance(s, Cond):
            descended.append(
                Cond(
                    cond=s.cond,
                    body=_vectorize_body(top, s.body),
                    else_body=_vectorize_body(top, s.else_body),
                )
            )
        elif isinstance(s, (GridTile, ThreadTile)):
            descended.append(s.with_bodies((_vectorize_body(top, s.body),)))
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


def _try_vec_store(stmts: Iterable[Stmt], start: int, n: int, top: TileOp) -> Write | None:
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
        value_dtype=writes[0].value_dtype,
    )
