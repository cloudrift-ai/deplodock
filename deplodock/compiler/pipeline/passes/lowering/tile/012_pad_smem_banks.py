"""Detect smem bank conflicts in body Loads of staged buffers and apply
``+1`` padding to one cache-axis extent to break the conflict.

Problem
=======

After ``007_stage_inputs`` lays out a slab in shared memory, body
Loads read it using thread-decoded coords. When the slab's per-thread-
axis stride is a multiple of 32 floats (the smem bank count × 4 bytes
/ 4 bytes per float), every thread in a warp hits the same bank — a
32-way conflict that serializes 32 LDS instructions into 32 single-
bank reads.

Concretely for a W slab with shape ``(8, 4, 64, 2)`` and strides
``(512, 128, 2, 1)``: within a warp ``(a3, a4)`` cycles through
``[0,8)×[0,4)``, addresses span ``a3*512 + a4*128``, mod 32 = 0 for
every thread. 32-way conflict.

Fix
===

Add ``+1`` padding to one cache-axis extent. The smem allocation grows
by one row, every higher-stride row shifts by one float, and the per-
thread address mod 32 changes from a uniform 0 to a sequence that
touches all 32 banks.

Analysis (closed-form, no per-thread evaluator)
================================================

For an affine Load index, decompose each dim into
``anchor + sum_v(coeff_v * Var(v))`` over the thread-axis vars (via
:func:`affine_form`). The flat smem address for a Load is
``sum_d (idx[d] * stride[d])``. Substituting the decomposition:

    flat(tid) = warp_const + sum_v (S_v * tid_v(tid))

where ``S_v = sum_d coeff_v_d * stride[d]`` is the per-thread-axis
contribution to the flat address. ``warp_const`` collects the anchor
parts plus everything that doesn't reference a thread-axis Var — it
shifts every thread's address by the same amount and so contributes a
constant offset to *all* banks; the bank distribution it produces is
independent of ``warp_const``.

So bank conflict counting reduces to: enumerate ``tid ∈ [0, 32)``,
decode ``(tid_v)`` per :func:`materialize_tile`'s tid-flatten scheme,
compute ``flat_v = sum_v S_v * tid_v``, distribute by ``flat_v % 32``,
count distinct addresses per bank. Broadcasts (multiple tids → same
``flat_v``) don't count as conflicts.

The pass:

1. For each ``Stage`` in the Tile body, find body Loads reading it.
2. Compute per-axis ``S_v`` once per Load via :func:`affine_form`.
3. If any Load is non-affine in the thread-axis vars, conservatively
   skip the Stage (we can't bound its conflict without the analyzer).
4. Try ``+1`` padding combinations (1-dim, then 2-dim) within
   ``_MAX_PADDED_SLAB_FLOATS``; pick the first that drives every
   Load's max-way conflict to 1. Fall back to the best partial fix.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis
from deplodock.compiler.ir.expr import affine_form
from deplodock.compiler.ir.stmt import Body, Load, Loop, Stmt, Tile
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
    return [s for s in body.iter() if isinstance(s, Load) and s.input == stage_name]


def _try_fix(stage: Stage, loads: list[Load], thread_axes: tuple[Axis, ...]) -> Stage | None:
    n = len(stage.axes)
    base_extents = tuple(int(ax.extent) for ax in stage.axes)

    # Precompute per-Load affine coefficients of the index over thread-axis vars.
    # Bail conservatively if any Load is non-affine in those vars.
    thread_var_set = frozenset(ax.name for ax in thread_axes)
    per_load_coeffs: list[list[dict[str, int]]] = []
    for load in loads:
        if len(load.index) != n:
            return None
        forms = [affine_form(e, thread_var_set) for e in load.index]
        if any(f is None for f in forms):
            return None
        per_load_coeffs.append([coeffs for _, coeffs in forms if (_, coeffs) is not None])  # type: ignore[misc]

    no_pad = (0,) * n
    base_conflict = _max_conflict(per_load_coeffs, base_extents, thread_axes)
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
        c = _max_conflict(per_load_coeffs, padded, thread_axes)
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


def _max_conflict(
    per_load_coeffs: list[list[dict[str, int]]],
    padded_extents: tuple[int, ...],
    thread_axes: tuple[Axis, ...],
) -> int:
    """Worst-case max-way bank conflict across all body Loads.

    For each Load, the flat smem address is an affine function of the
    thread-axis Vars: ``flat = warp_const + sum_v S_v * tid_v`` where
    ``S_v = sum_d coeff_v_d * stride[d]``. The constant warp-uniform
    part shifts every thread's address by the same amount and doesn't
    affect the bank distribution, so we drop it. We then enumerate the
    32 warp lanes, decode each lane's ``tid_v`` per
    ``materialize_tile``'s flatten scheme, compute ``flat_v % 32``, and
    count distinct addresses per bank (broadcasts don't count)."""
    strides = _strides(padded_extents)
    max_way = 1
    for coeffs_per_dim in per_load_coeffs:
        # Per-thread-axis contribution to the flat address.
        contrib: dict[str, int] = defaultdict(int)
        for d, coeffs in enumerate(coeffs_per_dim):
            for ax_name, c in coeffs.items():
                contrib[ax_name] += c * strides[d]

        bank_to_addrs: dict[int, set[int]] = defaultdict(set)
        for tid in range(WARP_SIZE):
            flat = 0
            rem = tid
            for ax in reversed(thread_axes):
                ext = int(ax.extent)
                flat += contrib.get(ax.name, 0) * (rem % ext)
                rem //= ext
            bank_to_addrs[flat % BANKS].add(flat)
        way = max((len(s) for s in bank_to_addrs.values()), default=1)
        max_way = max(max_way, way)
    return max_way


def _strides(padded_extents: tuple[int, ...]) -> list[int]:
    strides: list[int] = []
    cur = 1
    for e in reversed(padded_extents):
        strides.insert(0, cur)
        cur *= e
    return strides
