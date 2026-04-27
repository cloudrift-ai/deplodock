"""K-chunk every matmul-shaped reduce Loop in a Tile body into outer
(``K_o``) + inner (``K_i``) loops.

Runs *before* launch-geometry / blockify, so the trigger can't depend on
``Tile.axes`` partitioning. Detection is structural on the reduce body::

    Loop(K, reduce, body containing
        Load(A) and Load(B) on distinct buffers,
        a multiply combining them,
        an Accum that consumes the multiply
    )

This catches every matmul-shaped reduction regardless of where it sits
in the kernel — top of Tile body (plain matmul), nested inside a free
output-position loop (fused SDPA's V-projection), etc. Soft-max / sum /
RMSNorm reductions use a single buffer in their reduce body and are
correctly skipped.

Rewrite::

    Loop(K, reduce, body=B)
    →
    Loop(K_o, body=Loop(K_i, reduce, body=B[K → K_o*BK + K_i]))

The outer ``K_o`` is a serial chunk loop (``Loop`` with no immediate
``Accum`` — its ``is_reduce`` is False). The recursive Init-scoping rule
in ``kernel/000_place_inits`` recognises it as a reduce-passthrough and
keeps the Init at the surrounding scope so ``acc`` accumulates across
``K_o`` iterations.

BK = largest divisor of K from ``_BK_CANDIDATES`` (with K > BK so the
split actually fires). Idempotence: a reduce whose immediate body
already contains a reduce-Loop is left alone.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Cond, Load, Loop, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", TileOp)]

_BK_CANDIDATES = (64, 32, 16, 8, 4, 2)


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body):
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        return None
    idx, tile = tiles[0]
    new_body, changed = _chunk_in_body(tile.body)
    if not changed:
        return None
    return body[:idx] + (Tile(axes=tile.axes, body=new_body),) + body[idx + 1 :]


def _chunk_in_body(stmts: tuple) -> tuple[tuple, bool]:
    """Walk a body, chunking the first matmul-shaped reduce Loop found.
    Recurses through wrapper Loops / StridedLoops / Conds so a matmul
    nested inside an output-position free loop (fused SDPA V-projection)
    is reachable. Returns ``(new_body, changed)``."""
    out: list[Stmt] = []
    changed = False
    for s in stmts:
        if changed:
            out.append(s)
            continue
        if isinstance(s, Loop) and s.is_reduce and _is_matmul_reduce(s):
            chunked = _chunk_loop(s)
            if chunked is not None:
                out.append(chunked)
                changed = True
                continue
        if isinstance(s, (Loop, StridedLoop)):
            inner, inner_changed = _chunk_in_body(s.body)
            if inner_changed:
                out.append(_clone_loop(s, inner))
                changed = True
                continue
        if isinstance(s, Cond):
            inner_b, cb = _chunk_in_body(s.body)
            inner_e, ce = _chunk_in_body(s.else_body)
            if cb or ce:
                out.append(Cond(cond=s.cond, body=inner_b, else_body=inner_e))
                changed = True
                continue
        out.append(s)
    return tuple(out), changed


def _is_matmul_reduce(loop: Loop) -> bool:
    """A reduce Loop is matmul-shaped iff its immediate body has Loads
    of two distinct buffers whose indices both reference the loop's
    K axis. The multiply / Accum check is implicit — the only way two
    K-indexed Loads end up in a reduce body and contribute is through a
    fused-multiply-accumulate chain produced by lifting + fusion."""
    K_name = loop.axis.name
    bufs_with_K: set[str] = set()
    for s in loop.body:
        if isinstance(s, Load):
            free = set()
            for e in s.index:
                free |= e.free_vars()
            if K_name in free:
                bufs_with_K.add(s.input)
    if len(bufs_with_K) < 2:
        return False
    # Must contain an Accum (otherwise the reduce loop's compute is
    # pointwise — extremely unusual but guard against it).
    return any(isinstance(s, Accum) for s in loop.body)


def _chunk_loop(loop: Loop) -> Loop | None:
    K = int(loop.axis.extent)
    BK = next((c for c in _BK_CANDIDATES if K % c == 0 and K > c), None)
    if BK is None:
        return None

    # Idempotence: already chunked if body has a nested reduce-Loop.
    for s in loop.body:
        if isinstance(s, Loop) and s.is_reduce:
            return None

    K_name = loop.axis.name
    K_o = Axis(f"{K_name}_o", K // BK)
    K_i = Axis(f"{K_name}_i", BK)
    sigma = Sigma({K_name: Var(K_o.name) * Literal(BK, "int") + Var(K_i.name)})
    inner_body = tuple(s.rewrite(_id, sigma) for s in loop.body)
    return Loop(axis=K_o, body=(Loop(axis=K_i, body=inner_body),))


def _clone_loop(loop, body: tuple):
    if isinstance(loop, StridedLoop):
        return StridedLoop(axis=loop.axis, start=loop.start, step=loop.step, body=body)
    return Loop(axis=loop.axis, body=body)


def _id(name: str) -> str:
    return name
