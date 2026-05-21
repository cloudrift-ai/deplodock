"""Tile every matmul-shaped reduce Loop in a Tile body into outer
(``K_o``) + inner (``K_i``) loops.

This is intra-CTA K tiling — the runtime cross-CTA "split-K" strategy
lives in ``003_split_matmul_k``, which runs after this and promotes
``K_o`` to a grid dimension when the natural ``(M, N)`` grid doesn't
fill the device.

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

**Autotune fork.** The rule returns a *list* of TileOp variants —
one per ``BK`` candidate (from ``_BK_CANDIDATES``) that divides ``K``
with ``K > BK``. Option 0 is the largest valid divisor so
deterministic ``run_pipeline`` callers (greedy policy) behave exactly
as before; the rest only get explored under ``deplodock tune``. Each
variant stamps ``knobs={"BK": bk}`` for greedy-replay routing.

To pin BK for a specific test or sweep, set ``DEPLODOCK_BK=<int>``
(handled in ``tuning.forced_bk``) — that value is emitted as option 0
so greedy compiles use it and tune walks variants in the same order.

Idempotence: a reduce whose immediate body already contains a
reduce-Loop is left alone.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis, Role
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Cond, Loop, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce, single_tile

PATTERN = [Pattern("root", TileOp)]

_BK_CANDIDATES = (64, 32, 16, 8, 4, 2)

BK = Knob("BK", KnobType.INT, hints=_BK_CANDIDATES, help="Per-stage K-chunk size (intra-CTA K-loop trip count = K / BK)")


def _bk_candidates(tile, static_smem_cap: int) -> tuple[int, ...]:
    from deplodock.compiler.tuning import forced_bk

    f = forced_bk(tile, static_smem_cap)
    if f is None:
        return _BK_CANDIDATES
    # Forced value tried first; fall back to the built-in list if it
    # doesn't divide K (the caller still applies ``K % BK == 0`` filter).
    return (f, *(c for c in _BK_CANDIDATES if c != f))


def rewrite(ctx: Context, root: Node) -> Graph | None | list[TileOp]:
    """Fork over BK for matmul tiles. Option 0 is the heuristic BK
    (largest valid divisor, preserves deterministic-compile behavior);
    the rest only get explored under ``deplodock tune``."""
    body = root.op.body
    idx, tile = single_tile(body)
    site = _find_first_matmul_reduce(tile.body)
    if site is None:
        raise RuleSkipped("no matmul-shaped reduce Loop in body")
    if site.role is Role.STAGE_INNER:
        raise RuleSkipped("planner already chunked matmul K (Role.STAGE_INNER on reduce)")
    if any(inner.is_reduce for inner in site.body.of_type(Loop)):
        raise RuleSkipped("matmul reduce already chunked (idempotence)")

    K = int(site.axis.extent)
    cands = [c for c in _bk_candidates(tile, ctx.static_smem_cap) if K % c == 0 and K > c]
    if not cands:
        raise RuleSkipped("no BK candidate divides K with K > BK")

    variants: list[TileOp] = []
    for bk in cands:
        new_inner_body, changed = _chunk_in_body(tile.body, bk)
        if not changed:
            continue
        new_body = body[:idx] + (Tile(axes=tile.axes, body=new_inner_body),) + body[idx + 1 :]
        variants.append(TileOp(body=new_body, name=root.op.name, knobs={BK.name: bk}))
    if not variants:
        raise RuleSkipped("no BK variant produced a rewrite")
    if len(variants) == 1:
        return variants[0]
    return variants


def _find_first_matmul_reduce(stmts: tuple) -> Loop | None:
    """Pre-walk to locate the first matmul-shaped reduce Loop — mirrors
    the recursion structure in ``_chunk_in_body`` so the K it returns is
    the K the rewrite will actually target."""
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce and is_matmul_reduce(s):
            return s
        if isinstance(s, (Loop, StridedLoop)):
            found = _find_first_matmul_reduce(s.body)
            if found is not None:
                return found
        if isinstance(s, Cond):
            found = _find_first_matmul_reduce(s.body) or _find_first_matmul_reduce(s.else_body)
            if found is not None:
                return found
    return None


def _chunk_in_body(stmts: tuple, bk: int) -> tuple[tuple, bool]:
    """Walk a body, chunking the first matmul-shaped reduce Loop found
    with the supplied ``bk``. Recurses through wrapper Loops /
    StridedLoops / Conds so a matmul nested inside an output-position
    free loop (fused SDPA V-projection) is reachable. Returns
    ``(new_body, changed)``."""
    out: list[Stmt] = []
    changed = False
    for s in stmts:
        if changed:
            out.append(s)
            continue
        if isinstance(s, Loop) and s.is_reduce and is_matmul_reduce(s):
            chunked = _chunk_loop(s, bk)
            if chunked is not None:
                out.append(chunked)
                changed = True
                continue
        if isinstance(s, (Loop, StridedLoop)):
            inner, inner_changed = _chunk_in_body(s.body, bk)
            if inner_changed:
                out.append(replace(s, body=inner))
                changed = True
                continue
        if isinstance(s, Cond):
            inner_b, cb = _chunk_in_body(s.body, bk)
            inner_e, ce = _chunk_in_body(s.else_body, bk)
            if cb or ce:
                out.append(Cond(cond=s.cond, body=inner_b, else_body=inner_e))
                changed = True
                continue
        out.append(s)
    return tuple(out), changed


def _chunk_loop(loop: Loop, bk: int) -> Loop | None:
    K = int(loop.axis.extent)
    if K % bk != 0 or K <= bk:
        return None

    # Idempotence: already chunked if body has a nested reduce-Loop.
    if any(inner.is_reduce for inner in loop.body.of_type(Loop)):
        return None

    K_name = loop.axis.name
    K_o = Axis(f"{K_name}_o", K // bk)
    K_i = Axis(f"{K_name}_i", bk)
    sigma = Sigma({K_name: Var(K_o.name) * Literal(bk, "int") + Var(K_i.name)})
    inner_body = tuple(s.rewrite(_id, sigma) for s in loop.body)
    return Loop(axis=K_o, body=(Loop(axis=K_i, body=inner_body),))


def _id(name: str) -> str:
    return name
