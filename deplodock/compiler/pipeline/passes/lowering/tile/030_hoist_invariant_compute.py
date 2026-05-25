"""Hoist invariant compute cones out of a multi-source Stage's K_i body.

For the silu-gated MLP shape ``F.silu(gate) * up @ W``, ``020_stage_inputs``
puts all three gmem operands into one multi-source ``Stage`` whose K_i
reduce body interleaves a silu chain (on ``gate``+``up``) with the matmul
``Accum`` (on ``W``). The silu chain depends only on the M/K_i cache axes
of the cone sources; without hoisting it re-runs per (M, N, K_i) thread-
iter — hundreds of redundant evaluations per K_i element.

**Detection.** A *cone* is identified per multi-source Stage whose K_i
body contains:

1. ``Load`` stmts reading a subset ``G`` of the stage's sources whose
   ``cache_axes`` tuple is identical across all sources in ``G``
   (typically the M/K_i pair on a fused MLP shape).
2. A chain of ``Assign`` stmts whose deps stay inside ``G``'s Loads or
   prior cone Assigns and whose ``free_vars`` are a subset of ``G``'s
   cache axes.
3. Exactly one *boundary* SSA — the cone's frontier Assign output —
   consumed by stmts outside the cone. The boundary must be an
   ``Assign`` output (a raw Load boundary means no compute happens,
   and hoisting just shuffles bytes around for no win).

When (1)-(3) all hold, the cone is "hoistable".

**Autotune fork.** ``FUSED_PIPELINE`` is a BOOL knob; both polarities
are emitted in fixed order whenever a cone is found:

- ``False`` (greedy default) — pass-through. ``020_stage_inputs``
  already produces the inline-fuse shape (multi-source Stage with the
  chain in the K_i body); the chain runs per-thread per-K_i, redundant
  across the N tile.
- ``True`` — split the multi-source Stage into a *transport* Stage for
  the cone sources wrapping a *compute* Stage that cooperatively fills
  a fresh smem slab with the cone frontier; the K_i reduce reads the
  slab instead of re-running the chain. Non-cone sources move to an
  outer Stage so they stay in scope through the rewrite. Materialized
  by ``100_materialize_tile._emit_compute_stage`` — emits a single
  cooperative ``StridedLoop`` over the fused cache axes that runs the
  compute body once per cell.

Idempotence: stamps ``FUSED_PIPELINE`` on every emitted variant so
re-entry self-skips.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.expr import Expr, Var
from deplodock.compiler.ir.stmt import Assign, Body, Load, Stmt, Write
from deplodock.compiler.ir.tile.ir import (
    BufferedStage,
    CacheDim,
    ComputeStage,
    SerialTile,
    Source,
    Stage,
    TileOp,
)
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType

PATTERN = [Pattern("root", TileOp)]

FUSED_PIPELINE = Knob(
    "FUSED_PIPELINE",
    KnobType.BOOL,
    hints=(False, True),
    help=(
        "Hoist an invariant compute cone out of a multi-source Stage's K_i reduce body. "
        "False (default) — pass-through the inline-fuse shape from 020_stage_inputs. "
        "True — split the multi-source Stage into transport + ComputeStage so the cone "
        "chain runs once per cell instead of per (N) thread per K_i. Autotune fork."
    ),
)


def rewrite(ctx: Context, root: Node) -> list[TileOp] | None:
    if FUSED_PIPELINE.name in root.op.knobs:
        raise RuleSkipped("hoist_invariant_compute already applied (idempotence via knob)")

    target = _find_first_cone_target(root.op.body)
    if target is None:
        raise RuleSkipped("no multi-source Stage with a hoistable cone")

    variants: list[TileOp] = []
    for polarity in FUSED_PIPELINE.narrow((False, True)):
        if polarity:
            new_body = _apply_hoist(root.op.body, target)
        else:
            new_body = root.op.body
        variants.append(TileOp(body=new_body, name=root.op.name, knobs={**root.op.knobs, FUSED_PIPELINE.name: polarity}))
    if not variants:
        raise RuleSkipped("FUSED_PIPELINE env pin produced no matching variants")
    return variants


# ---------------------------------------------------------------------------
# Cone detection
# ---------------------------------------------------------------------------


class _ConeTarget:
    """Cached cone-detection result for one Stage."""

    __slots__ = ("stage", "reduce", "cone_sources", "cone_loads", "cone_assigns", "boundary_name", "fused_cache_axes")

    def __init__(self, stage, reduce, cone_sources, cone_loads, cone_assigns, boundary_name, fused_cache_axes):
        self.stage = stage
        self.reduce = reduce
        self.cone_sources = cone_sources
        self.cone_loads = cone_loads
        self.cone_assigns = cone_assigns
        self.boundary_name = boundary_name
        self.fused_cache_axes = fused_cache_axes


def _find_first_cone_target(body: Body) -> _ConeTarget | None:
    for s in body.iter():
        if isinstance(s, Stage) and not isinstance(s, (BufferedStage, ComputeStage)):
            target = _try_find_cone(s)
            if target is not None:
                return target
    return None


def _try_find_cone(stage: Stage) -> _ConeTarget | None:
    if len(stage.sources) < 2:
        return None
    reduce = _find_unique_stage_inner_reduce(stage.body)
    if reduce is None:
        return None

    # Group sources by their cache-axis name tuple. Only groups of ≥ 2
    # produce a fusable cone (1 source means the existing single-stage
    # staging already does the work).
    by_axes: dict[tuple[str, ...], list[Source]] = {}
    for src in stage.sources:
        key = tuple(ax.name for ax in src.cache_axes)
        by_axes.setdefault(key, []).append(src)

    stage_deps, free_vars, defs = _ssa_dataflow(reduce.body)

    for cache_axes_key, group in by_axes.items():
        if len(group) < 2:
            continue
        group_names = frozenset(src.name for src in group)
        cache_axes_set = frozenset(cache_axes_key)

        # Identify cone Loads (against cone sources) and cone Assigns
        # (whose transitive stage-deps lie entirely in the cone group).
        cone_loads = {name for name, ds in stage_deps.items() if ds and ds <= group_names and isinstance(defs[name], Load)}
        if not cone_loads:
            continue
        cone_assigns: set[str] = set()
        for name, ds in stage_deps.items():
            if isinstance(defs[name], Assign) and ds and ds <= group_names and free_vars[name] <= cache_axes_set:
                cone_assigns.add(name)
        if not cone_assigns:
            continue
        boundary = _find_boundary(reduce.body, cone_loads | cone_assigns)
        if len(boundary) != 1:
            continue
        (boundary_name,) = boundary
        if not isinstance(defs[boundary_name], Assign):
            continue

        # The fused slab's cache axes inherit from a representative cone
        # source — they all share the same cache axes by construction.
        fused_cache_axes = group[0].cache_axes
        return _ConeTarget(
            stage=stage,
            reduce=reduce,
            cone_sources=group_names,
            cone_loads=frozenset(cone_loads),
            cone_assigns=frozenset(cone_assigns),
            boundary_name=boundary_name,
            fused_cache_axes=fused_cache_axes,
        )
    return None


def _find_unique_stage_inner_reduce(body: Body) -> SerialTile | None:
    found: list[SerialTile] = []
    for s in body.iter():
        if isinstance(s, SerialTile) and s.kind == "stage_inner" and s.is_reduce:
            found.append(s)
    return found[0] if len(found) == 1 else None


def _ssa_dataflow(body: Body) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]], dict[str, Stmt]]:
    """For each SSA name defined in ``body`` (linear reduce — no nested
    control flow), compute:

    - ``stage_deps``: set of source-smem names transitively read.
    - ``free_vars``: set of axis Vars transitively referenced.
    - ``defs``: name → defining Stmt.
    """
    stage_deps: dict[str, frozenset[str]] = {}
    free_vars: dict[str, frozenset[str]] = {}
    defs: dict[str, Stmt] = {}
    for s in body:
        if isinstance(s, Load):
            defs[s.name] = s
            stage_deps[s.name] = frozenset({s.input})
            fv: frozenset[str] = frozenset()
            for e in s.index:
                fv |= frozenset(e.free_vars())
            free_vars[s.name] = fv
        elif isinstance(s, Assign):
            defs[s.name] = s
            d: frozenset[str] = frozenset()
            fv = frozenset()
            for arg in s.args:
                d |= stage_deps.get(arg, frozenset())
                fv |= free_vars.get(arg, frozenset())
            stage_deps[s.name] = d
            free_vars[s.name] = fv
    return stage_deps, free_vars, defs


def _find_boundary(body: Body, cone: set[str] | frozenset[str]) -> set[str]:
    """Boundary = cone-internal SSAs consumed by stmts outside the cone."""
    boundary: set[str] = set()
    for s in body:
        name = getattr(s, "name", None)
        is_producer = isinstance(s, (Load, Assign)) and name in cone
        if is_producer:
            continue
        for d in s.deps():
            if d in cone:
                boundary.add(d)
    return boundary


# ---------------------------------------------------------------------------
# True-polarity rewrite (Stage split + ComputeStage insertion)
# ---------------------------------------------------------------------------


def _apply_hoist(body: Body, target: _ConeTarget) -> Body:
    """Replace ``target.stage`` with the hoisted shape:

    ``Stage(non_cone_sources, body=Stage(cone_sources, body=ComputeStage(...)))``

    where the ``ComputeStage.compute`` carries the cone Loads+Assigns and
    writes into a fresh smem slab; the K_i reduce body is rewritten to
    Load that slab via the cone's boundary SSA name.

    The walk is by-identity (``id(stmt) == id(target.stage)``) rather
    than via :meth:`Body.map`, which would rebuild every wrapper Stmt
    and break the identity-anchored match.
    """
    target_id = id(target.stage)
    replacement = _build_hoisted_replacement(target)

    def walk(b: Body) -> Body:
        out: list[Stmt] = []
        for s in b:
            if id(s) == target_id:
                out.append(replacement)
                continue
            nested = s.nested()
            if nested:
                new_bodies = tuple(walk(nb) for nb in nested)
                if any(nb is not orig for nb, orig in zip(new_bodies, nested, strict=True)):
                    s = s.with_bodies(new_bodies)
            out.append(s)
        return Body(tuple(out))

    return walk(body)


def _build_hoisted_replacement(target: _ConeTarget) -> Stage:
    cone_sources = tuple(src for src in target.stage.sources if src.name in target.cone_sources)
    non_cone_sources = tuple(src for src in target.stage.sources if src.name not in target.cone_sources)
    fused_name = "_".join(src.name for src in cone_sources) + "_fused"
    fused_index: tuple[Expr, ...] = tuple(Var(ax.name) for ax in target.fused_cache_axes)

    # Compute body: cone Loads (verbatim, against the cone sources' smem
    # at cache-axis indices), cone Assigns (verbatim), final Write into
    # the fused slab.
    compute_stmts: list[Stmt] = []
    for s in target.reduce.body:
        if isinstance(s, Load) and s.name in target.cone_loads:
            compute_stmts.append(s)
    for s in target.reduce.body:
        if isinstance(s, Assign) and s.name in target.cone_assigns:
            compute_stmts.append(s)
    compute_stmts.append(Write(output=fused_name, index=fused_index, value=target.boundary_name))

    # New reduce body: drop cone Loads + Assigns; the boundary SSA now
    # comes from a single Load on the fused smem.
    new_reduce_stmts: list[Stmt] = []
    boundary_emitted = False
    for s in target.reduce.body:
        if isinstance(s, Load) and s.name in target.cone_loads:
            if not boundary_emitted:
                new_reduce_stmts.append(Load(name=target.boundary_name, input=fused_name, index=fused_index))
                boundary_emitted = True
            continue
        if isinstance(s, Assign) and s.name in target.cone_assigns:
            continue
        new_reduce_stmts.append(s)
    if not boundary_emitted:
        new_reduce_stmts.insert(0, Load(name=target.boundary_name, input=fused_name, index=fused_index))
    new_reduce = SerialTile(
        axis=target.reduce.axis,
        body=Body(tuple(new_reduce_stmts)),
        kind=target.reduce.kind,
        unroll=target.reduce.unroll,
    )

    # Output Source for the fused slab. ``buf`` is self-referential —
    # ComputeStage writes into its own smem, not from gmem. Origin is a
    # zero tuple matching the cache rank (smem-only, no gmem offset).
    from deplodock.compiler.ir.expr import Literal  # noqa: PLC0415

    zero_origin = tuple(Literal(0, "int") for _ in target.fused_cache_axes)
    fused_source = Source(
        name=fused_name,
        buf=fused_name,
        cache_dims=tuple(CacheDim(axis=ax, source_dim=i) for i, ax in enumerate(target.fused_cache_axes)),
        origin=zero_origin,
    )
    compute_stage = ComputeStage(sources=(fused_source,), body=Body((new_reduce,)), compute=Body(tuple(compute_stmts)))

    # Replace the cone-Stage stmts inside the original Stage.body — the
    # ComputeStage now sits where the reduce used to live, wrapped by
    # the inner transport Stage (cone sources) and then the outer
    # transport Stage (non-cone sources, if any). The original consumer
    # body's pre-reduce / post-reduce siblings come along for the ride.
    inner_body_stmts: list[Stmt] = []
    placed = False
    for s in target.stage.body:
        if s is target.reduce:
            inner_body_stmts.append(compute_stage)
            placed = True
        else:
            inner_body_stmts.append(s)
    if not placed:  # defensive — reduce should be there
        inner_body_stmts.append(compute_stage)

    inner = Stage(sources=cone_sources, body=Body(tuple(inner_body_stmts)))
    if not non_cone_sources:
        return inner
    return Stage(sources=non_cone_sources, body=Body((inner,)))


__all__ = ["FUSED_PIPELINE", "PATTERN", "rewrite"]
