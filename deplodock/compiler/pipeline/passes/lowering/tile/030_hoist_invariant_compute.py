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

**Autotune fork.** ``HOIST_COMPUTE`` is a BOOL knob; both polarities
are emitted in fixed order whenever a cone is found:

- ``False`` (greedy default) — pass-through. ``020_stage_inputs``
  already produces the inline-fuse shape (multi-source Stage with the
  chain in the K_i body); the chain runs per-thread per-K_i, redundant
  across the N tile.
- ``True`` — keep the single multi-source transport Stage and attach
  the cone frontier as a ``StageBundle.compute`` phase: a self-describing
  cooperative body that reads the cone sibling slabs and writes a fresh
  fused smem slab; the K_i reduce reads that slab instead of re-running
  the chain. Materialized by ``100_materialize_tile`` (``emit_compute_phase``)
  — emits a single cooperative ``StridedLoop`` over the fused cache axes
  (recovered from the compute body's Write + the cone sources) that runs
  the compute body once per cell.

Idempotence: stamps ``HOIST_COMPUTE`` on every emitted variant so
re-entry self-skips.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.expr import Expr, Var
from deplodock.compiler.ir.stmt import Assign, Body, Load, Stmt, Write
from deplodock.compiler.ir.tile.ir import (
    SerialTile,
    Source,
    StageBundle,
    StagePolicy,
    TileOp,
)
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType

PATTERN = [Pattern("root", TileOp)]

HOIST_COMPUTE = Knob(
    "HOIST_COMPUTE",
    KnobType.BOOL,
    hints=(False, True),
    help=(
        "Hoist an invariant compute cone out of a multi-source bundle's K_i reduce body. "
        "False (default) — pass-through the inline-fuse shape from 020_stage_inputs. "
        "True — keep the multi-source transport bundle and attach a StageBundle.compute "
        "phase so the cone chain runs once per cell instead of per (N) thread per K_i. Autotune fork."
    ),
    off=False,
)


def rewrite(ctx: Context, root: Node) -> list[TileOp] | None:
    if HOIST_COMPUTE.name in root.op.knobs:
        raise RuleSkipped("hoist_invariant_compute already applied (idempotence via knob)")

    target = _find_first_cone_target(root.op.body)
    if target is None:
        # No hoistable cone — record the off decision (body unchanged) so the
        # realized config keeps a uniform knob set instead of leaving it absent.
        return [TileOp(body=root.op.body, name=root.op.name, knobs={**root.op.knobs, HOIST_COMPUTE.name: False})]

    variants: list[TileOp] = []
    for polarity in HOIST_COMPUTE.narrow((False, True)):
        if polarity:
            new_body = _apply_hoist(root.op.body, target)
        else:
            new_body = root.op.body
        variants.append(TileOp(body=new_body, name=root.op.name, knobs={**root.op.knobs, HOIST_COMPUTE.name: polarity}))
    if not variants:
        raise RuleSkipped("HOIST_COMPUTE env pin produced no matching variants")
    return variants


# ---------------------------------------------------------------------------
# Cone detection
# ---------------------------------------------------------------------------


class _ConeTarget:
    """Cached cone-detection result for one SYNC bundle."""

    __slots__ = ("bundle", "reduce", "cone_sources", "cone_loads", "cone_assigns", "boundary_name", "fused_cache_axes")

    def __init__(self, bundle, reduce, cone_sources, cone_loads, cone_assigns, boundary_name, fused_cache_axes):
        self.bundle = bundle
        self.reduce = reduce
        self.cone_sources = cone_sources
        self.cone_loads = cone_loads
        self.cone_assigns = cone_assigns
        self.boundary_name = boundary_name
        self.fused_cache_axes = fused_cache_axes


def _find_first_cone_target(body: Body) -> _ConeTarget | None:
    """Scan for a multi-source SYNC bundle (no compute phase) with a
    hoistable cone. Bundles produced by 020_stage_inputs always satisfy
    this shape pre-promotion."""
    for s in body.iter():
        if not isinstance(s, StageBundle):
            continue
        if s.policy != StagePolicy.SYNC or s.compute is not None:
            continue
        target = _try_find_cone(s)
        if target is not None:
            return target
    return None


def _try_find_cone(bundle: StageBundle) -> _ConeTarget | None:
    if len(bundle.sources) < 2:
        return None
    reduce = _find_unique_stage_inner_reduce(bundle.body)
    if reduce is None:
        return None

    # Group sources by their cache-axis name tuple. Only groups of ≥ 2
    # produce a fusable cone (1 source means the existing single-source
    # staging already does the work).
    by_axes: dict[tuple[str, ...], list[Source]] = {}
    for src in bundle.sources:
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
            bundle=bundle,
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
# True-polarity rewrite (bundle stages split + compute member insertion)
# ---------------------------------------------------------------------------


def _apply_hoist(body: Body, target: _ConeTarget) -> Body:
    """Replace ``target.bundle`` with a new SYNC bundle holding a single
    multi-source transport Stage (sources ``non_cone + cone``), a
    ``compute=Body((cone_loads, cone_assigns, fused_write))`` phase, and a
    ``body`` that is the original bundle body with the K_i reduce rewritten
    to read the fused slab via a single Load.

    The compute phase is a per-thread cooperative body that reads sibling
    cone smem and writes into ``fused_name`` smem. Producer ordering at
    materialize is `transport → compute → consumer`.

    The walk is by-identity on the bundle (``id(stmt) == id(target.bundle)``)
    rather than via :meth:`Body.map`, which would rebuild every wrapper
    Stmt and break the identity-anchored match.
    """
    target_id = id(target.bundle)
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


def _build_hoisted_replacement(target: _ConeTarget) -> StageBundle:
    cone_sources = tuple(src for src in target.bundle.sources if src.name in target.cone_sources)
    non_cone_sources = tuple(src for src in target.bundle.sources if src.name not in target.cone_sources)
    fused_name = "_".join(src.name for src in cone_sources) + "_fused"
    fused_index: tuple[Expr, ...] = tuple(Var(ax.name) for ax in target.fused_cache_axes)

    # Compute template body: cone Loads (verbatim, against the cone
    # sources' smem at cache-axis indices), cone Assigns (verbatim),
    # final Write into the fused slab.
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

    # New bundle body: original bundle.body with the reduce stmt replaced
    # by the rewritten new_reduce (which reads the fused slab via the
    # boundary SSA name).
    new_body_stmts: list[Stmt] = []
    placed = False
    for s in target.bundle.body:
        if s is target.reduce:
            new_body_stmts.append(new_reduce)
            placed = True
        else:
            new_body_stmts.append(s)
    if not placed:  # defensive — reduce should be at top level of bundle.body
        new_body_stmts.append(new_reduce)

    # One homogeneous transport bundle holding ALL gmem operands. Source
    # order == producer issue order: ``non_cone_sources + cone_sources``
    # (keeps the cooperative-load emit order byte-identical to the
    # pre-collapse three-stage shape). The hoisted compute is a bundle
    # *phase* (``compute=``) — it is self-describing (its Loads name the
    # cone slabs, its single Write names the fused slab), so no output
    # ``Source`` is needed; the materializer recovers the slab name / loop
    # domain / dtype from the body at emit.
    return StageBundle(
        sources=non_cone_sources + cone_sources,
        body=Body(tuple(new_body_stmts)),
        compute=Body(tuple(compute_stmts)),
        policy=StagePolicy.SYNC,
    )


__all__ = ["HOIST_COMPUTE", "PATTERN", "rewrite"]
