"""Split-K every matmul-shaped reduce Loop in a Tile body into outer
(``K_o``) + inner (``K_i``) loops.

Matmul-only by design — the trigger is structural for dot-product
reductions (≥2 distinct K-indexed buffer Loads + an Accum). Single-
buffer reductions (softmax, sum, RMSNorm) are skipped.

Runs *before* launch-geometry / blockify, so the trigger can't depend on
``Tile.axes`` partitioning. Detection is structural on the reduce body::

    Loop(K, reduce, body containing
        Load(A) and Load(B) on distinct buffers,
        both Load index expressions referencing K,
        an Accum somewhere in the body
    )

The multiply between the two Loads is implicit — by construction it's
the only way two K-indexed Loads of distinct buffers contribute to an
Accum in this IR (lifted + fused from a frontend matmul). The rule
doesn't pattern-match the multiply Assign explicitly.

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

from dataclasses import replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Cond, Loop, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce, single_tile

PATTERN = [Pattern("root", TileOp)]

_BK_CANDIDATES = (64, 32, 16, 8, 4, 2)


def _bk_candidates(tile) -> tuple[int, ...]:
    from deplodock.compiler.tuning import forced_bk

    f = forced_bk(tile)
    if f is None:
        return _BK_CANDIDATES
    # Forced value tried first; fall back to the built-in list if it
    # doesn't divide K (the caller still applies ``K % BK == 0`` filter).
    return (f, *(c for c in _BK_CANDIDATES if c != f))


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body):
    idx, tile = single_tile(body)
    new_body, changed = _chunk_in_body(tile.body, tile)
    if not changed:
        raise RuleSkipped("no matmul-shaped reduce Loop with K-divisor in candidates")
    return body[:idx] + (Tile(axes=tile.axes, body=new_body),) + body[idx + 1 :]


def _chunk_in_body(stmts: tuple, tile) -> tuple[tuple, bool]:
    """Walk a body, chunking the first matmul-shaped reduce Loop found.
    Recurses through wrapper Loops / StridedLoops / Conds so a matmul
    nested inside an output-position free loop (fused SDPA V-projection)
    is reachable. ``tile`` provides the enclosing-tile context the
    BK-picking heuristic uses to scale the chunk to the output volume.
    Returns ``(new_body, changed)``."""
    out: list[Stmt] = []
    changed = False
    for s in stmts:
        if changed:
            out.append(s)
            continue
        if isinstance(s, Loop) and s.is_reduce and is_matmul_reduce(s):
            chunked = _chunk_loop(s, tile)
            if chunked is not None:
                out.append(chunked)
                changed = True
                continue
        if isinstance(s, (Loop, StridedLoop)):
            inner, inner_changed = _chunk_in_body(s.body, tile)
            if inner_changed:
                out.append(replace(s, body=inner))
                changed = True
                continue
        if isinstance(s, Cond):
            inner_b, cb = _chunk_in_body(s.body, tile)
            inner_e, ce = _chunk_in_body(s.else_body, tile)
            if cb or ce:
                out.append(Cond(cond=s.cond, body=inner_b, else_body=inner_e))
                changed = True
                continue
        out.append(s)
    return tuple(out), changed


def _chunk_loop(loop: Loop, tile) -> Loop | None:
    K = int(loop.axis.extent)
    BK = next((c for c in _bk_candidates(tile) if K % c == 0 and K > c), None)
    if BK is None:
        return None

    # Idempotence: already chunked if body has a nested reduce-Loop.
    if any(inner.is_reduce for inner in loop.loops):
        return None

    K_name = loop.axis.name
    K_o = Axis(f"{K_name}_o", K // BK)
    K_i = Axis(f"{K_name}_i", BK)
    sigma = Sigma({K_name: Var(K_o.name) * Literal(BK, "int") + Var(K_i.name)})
    inner_body = tuple(s.rewrite(_id, sigma) for s in loop.body)
    return Loop(axis=K_o, body=(Loop(axis=K_i, body=inner_body),))


def _id(name: str) -> str:
    return name
