"""Detect smem bank conflicts in body Loads of staged buffers and apply
``+1`` padding to one cache-axis extent to break the conflict.

Problem
=======

After ``010_stage_inputs`` + ``011_merge_stages`` lay out a slab in
shared memory, body Loads read the slab using thread-decoded coords.
When the slab's per-axis stride is a multiple of 32 floats (the smem
bank count × 4 bytes / 4 bytes per float), every thread in a warp
hits the same bank — a 32-way conflict that serializes 32 LDS
instructions into 32 single-bank reads.

Concretely for a matmul W slab with shape ``(8, 4, 64, 2)`` and
strides ``(512, 128, 2, 1)``: within a warp ``(a3, a4)`` cycles
through ``[0,8)×[0,4)``, addresses span ``a3*512 + a4*128``, mod 32
floats = 0 for every thread. 32-way conflict.

Fix
===

Add ``+1`` padding to the second-fastest cache-axis extent. The smem
allocation grows by one row's worth, every higher-stride row shifts by
one float, and the per-thread address mod 32 changes from a uniform 0
to a sequence that touches all 32 banks.

The pass:

1. For each ``Stage`` in the Tile body, find body ``Load`` stmts that
   read from this stage.
2. Simulate one warp's smem accesses. Decode each thread's
   ``(a1, a3, a4, ...)`` from ``tid``, evaluate the Load index with
   those values, compute ``flat_addr % 32`` per thread → bank
   distribution.
3. If max-way conflict > 1 (and not a pure broadcast), try ``+1``
   padding on each cache-axis dim (innermost-first). Pick the first
   pad that resolves the conflict.
4. If no single ``+1`` resolves it, log a diagnostic with the
   remaining conflict so the user can consider a larger restructure
   (axis permute, layout change in staging).

Side note: padding the *innermost* dim (stride 1) doesn't help because
the body Load still reads stride-1 across cells. Padding *outer* dims
shifts middle-row addresses, which is exactly the broken-alignment we
want.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from deplodock.compiler.ir.stmt import Body, Load, Loop, Stmt, Tile, iter_body
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

logger = logging.getLogger(__name__)

PATTERN = [Pattern("root", TileOp)]

WARP_SIZE = 32
BANKS = 32  # 32 banks of 4 bytes
# Per-Stage padded-slab budget. The static-smem cap on consumer Ada/Hopper
# is 48 KB = 12288 floats. Stages get double-buffered downstream (×2) and
# typical matmul kernels have ≥ 2 sibling Stages, so per-Stage pre-DB
# budget is roughly cap / 4 → 3072. We give a slightly larger headroom
# (5120) so single-large-Stage kernels still benefit, and accept that a
# matmul with two near-equal Stages may miss the perfect-fix candidate
# and fall back to the partial-fix pad.
_MAX_PADDED_SLAB_FLOATS = 5 * 1024


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: Body) -> Body | None:
    idx, tile = single_tile(body)

    thread_axes = tuple(ba.axis for ba in tile.axes if ba.bind == BIND_THREAD)
    if not thread_axes:
        raise RuleSkipped("Tile has no THREAD axes — no bank-conflict layout to pad")

    new_tile_body = _process_body(tile.body, thread_axes)
    if new_tile_body is tile.body or new_tile_body == tile.body:
        raise RuleSkipped("no Stage has a fixable bank conflict within slab budget")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process_body(body: Body, thread_axes: tuple[Axis, ...]) -> Body:
    """Walk body and any free-Loop scopes; pad each Stage that has a
    fixable conflict. Returns the new body (or original if no changes)."""
    new_body: list[Stmt] = list(body)
    changed = False
    for i, s in enumerate(body):
        if isinstance(s, Stage) and not (s.pad and any(s.pad)):
            loads = _loads_reading(body, s.name)
            if not loads:
                continue
            updated = _try_fix(s, loads, thread_axes)
            if updated is not None:
                new_body[i] = updated
                changed = True
        elif isinstance(s, Loop):
            inner = _process_body(s.body, thread_axes)
            if inner is not s.body and inner != s.body:
                new_body[i] = dc_replace(s, body=inner)
                changed = True
    return tuple(new_body) if changed else body


def _loads_reading(body: Body, stage_name: str) -> list[Load]:
    """Collect every Load anywhere in ``body`` reading from ``stage_name``."""
    return [s for s in iter_body(body) if isinstance(s, Load) and s.input == stage_name]


def _try_fix(stage: Stage, loads: list[Load], thread_axes: tuple[Axis, ...]) -> Stage | None:
    n = len(stage.axes)
    base_extents = tuple(int(ax.extent) for ax in stage.axes)

    no_pad = (0,) * n
    base_conflict = _max_conflict(loads, base_extents, thread_axes)
    if base_conflict <= 1:
        return None

    # Try +1 padding combinations, smallest-pad-first. Single-dim pads
    # (n options) before pair-dim pads (n*(n-1)/2 options). Stop at the
    # first conflict-free configuration.
    candidates: list[tuple[int, ...]] = []
    # 1-dim: prefer innermost-first (likely the right granularity).
    for dim in range(n - 1, -1, -1):
        pad = [0] * n
        pad[dim] = 1
        candidates.append(tuple(pad))
    # 2-dim: every unordered pair.
    for d1 in range(n):
        for d2 in range(d1 + 1, n):
            pad = [0] * n
            pad[d1] = 1
            pad[d2] = 1
            candidates.append(tuple(pad))

    best_pad = no_pad
    best_c = base_conflict
    for pad in candidates:
        padded = tuple(e + p for e, p in zip(base_extents, pad, strict=True))
        slab_floats = 1
        for e in padded:
            slab_floats *= e
        if slab_floats > _MAX_PADDED_SLAB_FLOATS:
            continue
        c = _max_conflict(loads, padded, thread_axes)
        if c <= 1:
            n_pad_dims = sum(1 for p in pad if p)
            logger.info(
                "Stage %s: %d-way bank conflict → 1-way after %d-dim pad %s",
                stage.name,
                base_conflict,
                n_pad_dims,
                pad,
            )
            return dc_replace(stage, pad=pad)
        if c < best_c:
            best_c = c
            best_pad = pad

    if best_c < base_conflict:
        logger.warning(
            "Stage %s: %d-way bank conflict; partial fix → %d-way with pad %s (no perfect fix found)",
            stage.name,
            base_conflict,
            best_c,
            best_pad,
        )
        return dc_replace(stage, pad=best_pad)
    logger.warning("Stage %s: %d-way bank conflict; no padding fix found", stage.name, base_conflict)
    return None


def _max_conflict(loads: list[Load], padded_extents: tuple[int, ...], thread_axes: tuple[Axis, ...]) -> int:
    """Worst-case max-way bank conflict across all body Loads of this stage.

    For each Load, simulate one warp (tid 0..31), decode thread-axis Vars
    from tid, evaluate the Load index, compute smem flat address ↦ bank.
    Conflict = max # of *distinct* addresses landing in the same bank
    (broadcast — same address — is not counted)."""
    strides: list[int] = []
    cur = 1
    for e in reversed(padded_extents):
        strides.insert(0, cur)
        cur *= e

    max_way = 1
    for load in loads:
        if len(load.index) != len(padded_extents):
            continue
        bank_to_addrs: dict[int, set[int]] = defaultdict(set)
        for tid in range(WARP_SIZE):
            env = _decode_threads(tid, thread_axes)
            flat = 0
            for d, idx_expr in enumerate(load.index):
                v = _eval_int(idx_expr, env)
                flat += v * strides[d]
            bank_to_addrs[flat % BANKS].add(flat)
        way = max((len(s) for s in bank_to_addrs.values()), default=1)
        max_way = max(max_way, way)
    return max_way


def _decode_threads(tid: int, thread_axes: tuple[Axis, ...]) -> dict[str, int]:
    """Mirror ``materialize_tile._build_linear_tid``: rightmost thread axis
    has stride 1 within a warp; outer axes scale up."""
    env: dict[str, int] = {}
    rem = tid
    for ax in reversed(thread_axes):
        ext = int(ax.extent)
        env[ax.name] = rem % ext
        rem //= ext
    return env


def _eval_int(expr: Expr, env: dict[str, int]) -> int:
    """Integer-eval an index expression. Vars not in ``env`` are treated
    as 0 (they're warp-uniform — reduce-loop vars, block axes, etc — and
    contribute the same offset to every thread, so they don't affect bank
    distribution)."""
    if isinstance(expr, Literal):
        return int(expr.value) if isinstance(expr.value, (int, bool)) else int(expr.value)
    if isinstance(expr, Var):
        return env.get(expr.name, 0)
    if isinstance(expr, BinaryExpr):
        lv = _eval_int(expr.left, env)
        rv = _eval_int(expr.right, env)
        if expr.op == "+":
            return lv + rv
        if expr.op == "-":
            return lv - rv
        if expr.op == "*":
            return lv * rv
        if expr.op in ("/", "//"):
            return lv // rv if rv != 0 else 0
        if expr.op == "%":
            return lv % rv if rv != 0 else 0
    return 0


# Silence ruff F401: ``Counter`` is used implicitly via defaultdict.
_ = Counter
