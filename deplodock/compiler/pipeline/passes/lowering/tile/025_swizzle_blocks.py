"""Default-on CTA swizzle for matmul-shape grids.

For each ``GridTile(axes=(K_s?, M_b, N_b))`` produced by the partition
planner, stamp ``swizzle_group_m = _group_m()`` (default 8, from the ``GROUP_M`` knob).
Renderer-only transform: ``GridTile.axes`` are untouched, only the
field changes. ``GridTile.render`` reads it and emits a Triton-canonical
``blockIdx.x`` decode (``ir/stmt/blocks._render_swizzled_grid_decode``)
so consecutive CTAs walk down M in groups of ``GROUP_M`` before stepping
N. A row-group of CTAs then shares A's row tile in L2 — the headline
optimisation from the CloudRift RTX 5090 SGEMM blog (DRAM throughput
32 % → 8.5 % on a 4096³ SGEMM).

Idempotent (skip if ``swizzle_group_m != 1`` already), matmul-shape
only (skip if the two innermost axes lack the planner's ``_b`` suffix),
and the runtime ``min(GROUP_M, num_m - first_m)`` clamp self-disables
on tiny / tall-skinny matmuls so no extra eligibility plumbing is
needed. ``DEPLODOCK_GROUP_M=1`` is the global escape hatch.

Downstream passes (``020_stage_inputs``, ``040_use_ring_buffers``,
``050_use_tma``, ``080_pipeline_stages``, ``085_warp_specialize``,
``090_mark_unroll``, the kernel-IR materialiser) all key on
``GridTile.axes`` *identity* and on K_o / K_i, never on the rendered
RHS of ``int m_b = …;`` — so the body's σ rewrites pick up the swizzled
values via plain CUDA scoping, with no σ change needed.

Slot rationale: runs after ``020_stage_inputs`` (smem staging reads
axis identities, unaffected by a later swizzle stamp) and before
``025_unify_sibling_stages``. The exact position inside the lowering
chain doesn't matter as long as no later pass rebuilds the GridTile
without propagating ``swizzle_group_m`` — every such rebuilder in the
codebase (``tile/_helpers.replace_parallel_tile_body``,
``tile/085_warp_specialize``, ``kernel/100_materialize_tile``,
``kernel/110_drop_redundant_syncs``, ``ir/tile/passes`` Sigma rewrite)
threads the field through.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock import config
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.ir.tile.ir import GridTile, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType

PATTERN = [Pattern("root", TileOp)]

# CTA-swizzle row-group size. ``1`` is the no-op (row-major decode); ``8`` is
# the Triton/CUTLASS default this pass stamps. Hints double as the allowed-value
# set: an out-of-set or garbage pin raises ``ValueError`` (see ``_group_m``) so
# a typo doesn't silently degrade matmul perf.
GROUP_M = Knob(
    "GROUP_M",
    KnobType.INT,
    hints=(1, 2, 4, 8, 16),
    help="CTA-swizzle row-group size (1 = disable; renderer falls back to row-major decode)",
)
_GROUP_M_DEFAULT = 8


def _group_m() -> int:
    """Resolve ``DEPLODOCK_GROUP_M`` → group size. Unset → ``8``. Garbage or
    out-of-``hints`` values raise ``ValueError``. Reads through the descriptor's
    env path so it shares the ``DEPLODOCK_KNOBS`` splat with the other knobs."""
    raw = config.knob_raw(GROUP_M.name)
    if raw is None or raw == "":
        return _GROUP_M_DEFAULT
    try:
        v = int(raw)
    except ValueError as e:
        raise ValueError(f"DEPLODOCK_GROUP_M must be one of {GROUP_M.hints}, got {raw!r}") from e
    if v not in GROUP_M.hints:
        raise ValueError(f"DEPLODOCK_GROUP_M must be one of {GROUP_M.hints}, got {v}")
    return v


def rewrite(root: Node) -> TileOp | None:
    op: TileOp = root.op
    # Idempotence: the decision is recorded as the GROUP_M knob (every path
    # stamps it now), so a re-scan of the rebound op skips here.
    if GROUP_M.name in op.knobs:
        raise RuleSkipped("GROUP_M already decided (idempotence via knob)")

    def _off() -> TileOp:
        """Record the no-swizzle decision: GROUP_M=1 (row-major decode), body
        unchanged. Stamped on every non-acting path (disabled / not-matmul / no
        eligible grid) so the realized config keeps a uniform knob set."""
        return TileOp(body=op.body, name=op.name, knobs={**op.knobs, GROUP_M.name: 1})

    group_m = _group_m()
    if group_m == 1:
        return _off()
    if not _is_matmul(op):
        return _off()
    new_body, changed = _stamp_top_grid(op.body, group_m)
    if not changed:
        return _off()
    return TileOp(body=new_body, name=op.name, knobs={**op.knobs, GROUP_M.name: group_m})


def _is_matmul(op: TileOp) -> bool:
    """Matmul-priority recognition via ``TileOp.knobs`` stamped by the
    partition planner.

    The planner stamps ``BK`` / ``BR`` for every kernel: matmul kernels
    have ``BK > 1`` (per-stage K-chunk) and ``BR == 1`` (no cooperative
    thread reduction). Pointwise kernels stamp ``BK == 1`` (no K loop);
    cooperative-reduce kernels stamp ``BR > 1``. SDPA's fused-prologue
    matmul matches this signature too — out of scope for the first cut,
    so we additionally require ``SPLITK == 1`` since the planner forces
    SPLITK to 1 there (see ``enumerate_cartesian``'s
    ``force_splitk_one``).
    """
    bk = op.knobs.get("BK", 1)
    br = op.knobs.get("BR", 1)
    return bk > 1 and br == 1


def _stamp_top_grid(body: Body, group_m: int) -> tuple[Body, bool]:
    """Walk the TileOp body, find the (at most one) outer ``GridTile``,
    and stamp ``swizzle_group_m`` if it has ≥ 2 axes and isn't already
    swizzled. The planner emits matmul GridTiles with ``(M_b, N_b)`` or
    ``(K_s, M_b, N_b)`` — len ≥ 2 covers both."""
    out: list[Stmt] = []
    changed = False
    for s in body:
        if isinstance(s, GridTile) and len(s.axes) >= 2 and s.swizzle_group_m == 1:
            out.append(replace(s, swizzle_group_m=group_m))
            changed = True
            continue
        out.append(s)
    return Body(tuple(out)), changed
