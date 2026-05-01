"""Split each matmul-shaped inner reduce ``Loop`` (``K_i``) into N
consecutive halves of equal extent, gated by ``DEPLODOCK_BK_SPLIT``.

Motivates the BK=64 + B128 swizzle case: ``cuTensorMapEncodeTiled``
caps inner box bytes at the swizzle width (128 B for B128 fp32 = 32
fp32 elements), so a BK=64 cache axis can't be loaded with a single
TMA box. Splitting ``K_i`` into ``N=2`` halves yields two consecutive
reduce loops, each with ``K_i.extent / 2 = 32``, and N parallel
``Stage`` slabs (each with the narrower inner extent) — the
TMA path then narrows each to a swizzle-eligible box per half.

Runs *after* ``008_register_tile`` so the body is fully register-tiled
when split. Earlier positions don't work because:

- Before ``007_stage_inputs``: stages don't exist yet, so the rule
  would have to predict slab structure.
- Between 007 and 008: register-tile gets confused by two consecutive
  matmul-shape reduces sharing accumulator names and only fully
  replicates one of them.

By 009 the body is::

    Loop(K_o, body=[
        Stage(weight_smem, axes=(M, K_i=64), origin=(.., k_o*64)),
        # input may or may not be staged
        Loop(K_i=64, reduce, body=[ <register-tiled FMA chain> ]),
    ])

After split (N=2)::

    Loop(K_o, body=[
        Stage(weight_smem,   axes=(M, K_i=32), origin=(.., k_o*64 + 0)),
        Loop(K_i=32, reduce, body=[ <register-tiled chain reading weight_smem> ]),
        Stage(weight_smem_1, axes=(M, K_i=32), origin=(.., k_o*64 + 32)),
        Loop(K_i=32, reduce, body=[ <chain reading weight_smem_1>, K_i shifted +32 for any unstaged Loads ]),
    ])

Default ``DEPLODOCK_BK_SPLIT=1`` is a no-op. ``=2`` halves every
matmul-shaped K_i whose extent is divisible by 2 and the resulting
halves stay >= ``_MIN_HALF_EXTENT``. Idempotence: a halved body has
extent below the threshold and the gate fails at re-run.
"""

from __future__ import annotations

import os
from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Cond, Load, Loop, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce, single_tile

PATTERN = [Pattern("root", TileOp)]

# Post-split halves below this width (in fp32) provide no benefit for
# the TMA+swizzle case (B128 swizzle requires inner=32 fp32 = 128 B
# exactly). Refuse to split if any half would be smaller.
_MIN_HALF_EXTENT = 32


def rewrite(graph: Graph, root: Node) -> Graph | None:
    n = _split_count()
    if n <= 1:
        raise RuleSkipped("DEPLODOCK_BK_SPLIT <= 1 — no-op")
    new_body = _maybe_rewrite(root.op.body, n)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _split_count() -> int:
    raw = os.environ.get("DEPLODOCK_BK_SPLIT", "1")
    try:
        n = int(raw)
    except ValueError as e:
        raise ValueError(f"DEPLODOCK_BK_SPLIT must be a positive integer, got {raw!r}") from e
    if n < 1:
        raise ValueError(f"DEPLODOCK_BK_SPLIT must be >= 1, got {n}")
    return n


def _maybe_rewrite(body, n: int):
    idx, tile = single_tile(body)
    new_tile_body, changed = _split_in_body(tile.body, n)
    if not changed:
        raise RuleSkipped(f"no matmul reduce divisible by {n} with halves >= {_MIN_HALF_EXTENT}")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _split_in_body(stmts: tuple[Stmt, ...], n: int) -> tuple[tuple[Stmt, ...], bool]:
    """Walk a body, splitting the first eligible (stages + matmul-reduce)
    pattern found. Mirrors ``002_split_matmul_k._chunk_in_body``:
    recurses through wrapper Loops / StridedLoops / Conds so a matmul
    nested inside a free output-position loop is reachable."""
    out: list[Stmt] = []
    changed = False
    i = 0
    while i < len(stmts):
        if changed:
            out.append(stmts[i])
            i += 1
            continue
        s = stmts[i]
        if isinstance(s, Loop) and s.is_reduce and is_matmul_reduce(s):
            split = _split_pattern(stmts, i, n)
            if split is not None:
                replaced_stmts, consumed_back = split
                # Re-emit any stages that were collected as part of the
                # split: drop them from ``out`` (we already appended them
                # in their original form) before appending the new shape.
                if consumed_back:
                    del out[-consumed_back:]
                out.extend(replaced_stmts)
                changed = True
                i += 1
                continue
        if isinstance(s, (Loop, StridedLoop)):
            inner, ic = _split_in_body(s.body, n)
            if ic:
                out.append(dc_replace(s, body=inner))
                changed = True
                i += 1
                continue
        if isinstance(s, Cond):
            inner_b, cb = _split_in_body(s.body, n)
            inner_e, ce = _split_in_body(s.else_body, n)
            if cb or ce:
                out.append(Cond(cond=s.cond, body=inner_b, else_body=inner_e))
                changed = True
                i += 1
                continue
        out.append(s)
        i += 1
    return tuple(out), changed


def _split_pattern(stmts: tuple[Stmt, ...], reduce_idx: int, n: int) -> tuple[tuple[Stmt, ...], int] | None:
    """Split a (Stage*, Loop[reduce]) pattern into ``n`` halves.

    Returns ``(new_stmts, num_preceding_stages_consumed)`` describing
    the replacement: the caller drops that many already-emitted stmts
    and appends ``new_stmts``. Returns ``None`` if the reduce extent
    isn't cleanly divisible or the halves would be too narrow.

    Stages are detected as the contiguous run of ``Stage``
    immediately preceding the reduce loop. Each is split along the
    cache axis whose name matches the reduce loop's axis name."""
    reduce_loop = stmts[reduce_idx]
    K_name = reduce_loop.axis.name
    K = int(reduce_loop.axis.extent)
    if K % n != 0:
        return None
    half = K // n
    if half < _MIN_HALF_EXTENT:
        return None

    # Walk backwards to collect contiguous Stages.
    stage_start = reduce_idx
    while stage_start > 0 and isinstance(stmts[stage_start - 1], Stage):
        stage_start -= 1
    stages = stmts[stage_start:reduce_idx]

    # Emit all per-half stages first, then all per-half reduce loops.
    # Keeps the body shape ``[Stage*, Loop*]`` so the materializer's
    # shared-mbarrier scheme (count=num_stages, ONE wait covering all
    # arrives) works — interleaving stages and reduces would deadlock
    # because the first wait would need ``num_stages`` arrives but only
    # one stage would have arrived at that point. Smem usage is
    # unchanged (n × half × M = K × M total) and 015_pipeline_async
    # still pipelines the K_outer loop end-to-end.
    # Emit one MERGED reduce loop whose body does the work of all ``n``
    # halves back-to-back per K_i iteration, with K_i ∈ [0, half).
    # Two-loop variants give NVCC two distinct scheduling regions and
    # the compiler keeps each half's locals live across the boundary —
    # observed at 173–255 regs / 17% occ on sm_120 fp32 1024×1024,
    # vs 95 regs / 33% occ for the unsplit BK=64 baseline. With one
    # merged inner loop the compiler interleaves both halves freely,
    # matching the unsplit register footprint.
    #
    # SSA-name strategy for the merged body:
    # - ``Accum`` target names (``acc*``) are SHARED across halves
    #   (Init lives at the parent scope — both halves accumulate into
    #   the same accumulators). NOT renamed.
    # - Every other locally-defined name (``in*``, ``v*``) is suffixed
    #   ``_<half_idx>`` so the halves don't collide.
    out_stages: list[Stage] = []
    acc_names = {s.name for s in reduce_loop.body if isinstance(s, Accum)}
    locally_defined = {n for s in reduce_loop.body for n in s.defines()}
    rename_targets = locally_defined - acc_names

    merged_body: list[Stmt] = []
    for i in range(n):
        offset = i * half
        rename_map: dict[str, str] = {}
        for st in stages:
            split_st, new_name = _split_stage(st, K_name, half, offset, suffix_idx=i)
            out_stages.append(split_st)
            if new_name is not None:
                rename_map[st.name] = new_name
        suffix = "" if i == 0 else f"_{i}"
        sigma = Sigma({K_name: Var(K_name) + Literal(offset, "int")}) if offset else Sigma.IDENTITY

        def rename_ssa(name: str, _suffix=suffix) -> str:
            if not _suffix or name not in rename_targets:
                return name
            return f"{name}{_suffix}"

        for s in reduce_loop.body:
            merged_body.append(_merged_body_stmt(s, rename_ssa, sigma, rename_map))

    merged_loop = Loop(axis=Axis(K_name, half), body=tuple(merged_body), unroll=reduce_loop.unroll)
    return (*out_stages, merged_loop), len(stages)


def _split_stage(stage: Stage, K_name: str, half: int, offset: int, suffix_idx: int) -> tuple[Stage, str | None]:
    """Return (replacement_stage, new_name_or_None).

    If the stage's cache axes contain ``K_name``, halve that axis's
    extent, shift the matching origin entry by ``offset``, and rename
    the buffer for halves > 0 (suffix ``_{suffix_idx}``). Stages
    independent of ``K_name`` pass through unchanged."""
    axis_idx = next((j for j, ax in enumerate(stage.axes) if ax.name == K_name), None)
    if axis_idx is None:
        return stage, None
    new_axes = tuple(Axis(ax.name, half) if j == axis_idx else ax for j, ax in enumerate(stage.axes))
    new_origin = tuple(e + Literal(offset, "int") if j == axis_idx and offset else e for j, e in enumerate(stage.origin))
    new_name = stage.name if suffix_idx == 0 else f"{stage.name}_{suffix_idx}"
    new_stage = dc_replace(stage, name=new_name, axes=new_axes, origin=new_origin)
    return new_stage, (new_name if suffix_idx > 0 else None)


def _merged_body_stmt(stmt: Stmt, rename_ssa, sigma: Sigma, rename_map: dict[str, str]) -> Stmt:
    """Rewrite one stmt of the original reduce body for inclusion in
    the merged loop's body for a particular half.

    - ``Load(input=staged_name, ...)`` gets repointed at the per-half
      cache buffer AND its index stays cache-local (NOT sigma-shifted),
      since the merged loop's K_name is in ``[0, half)`` and the cache
      slab also indexes ``[0, half)``. The output SSA name still goes
      through ``rename_ssa`` so it doesn't collide with the other half.
    - Everything else gets ``rewrite(rename_ssa, sigma)`` — unstaged
      Loads of source tensors pick up the K-axis offset via sigma so
      they reference the right global K-slice, axis arithmetic in
      derived expressions inherits the same shift, and Accum/Assign
      stmts use the suffix-renamed locals while keeping accumulator
      names intact (the rename closure excludes Accum targets).
    """
    if isinstance(stmt, Load) and stmt.input in rename_map:
        return dc_replace(stmt, name=rename_ssa(stmt.name), input=rename_map[stmt.input])
    return stmt.rewrite(rename_ssa, sigma)


def _id(name: str) -> str:
    return name
