"""Hoist a loop-invariant compute cone out of the K-inner reduce body.

After ``007_stage_inputs`` runs, the Tile body has shape::

    Tile(axes=...):
        for K_outer:                                 # free
            stage_A = Stage(A, axes=(M,K))
            stage_B = Stage(B, axes=(M,K))
            stage_W = Stage(W, axes=(K,N))
            for K_inner:                             # reduce
                a = load stage_A[M, K_inner]
                ... silu/elementwise chain on a ...
                b = load stage_B[M, K_inner]
                v = multiply(b, sig*a)               # final silu(a)*b
                w = load stage_W[K_inner, N]
                acc += multiply(w, v)

For the silu-gated MLP shape (``F.silu(gate)*up @ W``), the chain
``v0..v_silu`` depends only on ``stage_A`` + ``stage_B`` (cache axes
``{M, K_inner}``); it has no dependency on ``N`` (the consumer thread
axis). Today that chain re-runs every ``N``-thread, every ``N``-tile —
~896× redundancy on Qwen-MLP shapes.

This pass identifies such cones and hoists them out of the reduce
body, choosing between two output shapes based on the
``FUSED_PIPELINE`` knob:

**Inline-fuse** (``FUSED_PIPELINE=False`` — default):

    fused = Stage(name=…, axes=(M, K_inner), body=(
        Load(input=A, index=…),                      # gmem
        Load(input=B, index=…),                      # gmem
        … silu chain over A's load …
        Assign(multiply with B's load),
        Write(output=fused, index=(M, K_inner), value=<frontier>),
    ))

Source Stages ``stage_A`` / ``stage_B`` are removed; one ``Stage``
carries both the gmem transport and the silu compute. 1 smem buffer.
Trade-off: the multi-source body bypasses ``010 / 011 / 013`` (those
require ``len(source_loads) == 1``) so A/B can't be double-buffered /
TMA-promoted / async. Best on architectures where smem budget is
tight (sm_120) and silu redundancy is the dominant cost.

**Hoist-compute** (``FUSED_PIPELINE=True``):

    stage_A = Stage(A, axes=(M,K))                   # unchanged
    stage_B = Stage(B, axes=(M,K))                   # unchanged
    fused = ComputeStage(name=…, axes=(M, K_inner), body=(
        Load(input=stage_A, index=(M, K_inner)),     # smem
        Load(input=stage_B, index=(M, K_inner)),     # smem
        … silu chain …
        Write(output=fused, index=(M, K_inner), value=<frontier>),
    ))

Transport Stages stay single-source so ``010/011/013`` can double-
buffer + TMA + cp.async them; ``015_pipeline_k_outer`` can software-
pipeline the transport across K_outer iterations while the
``ComputeStage`` slots between the wait and the K_inner reduce.
1 + N smem buffers. Best on architectures with abundant smem
(sm_90+ with TMA) where the latency hide outweighs the buffer cost.

Both shapes collapse the reduce-body silu chain to a single Load on
the cone's output buffer.

Forks unconditionally — both polarities are always emitted in a fixed
order (inline-fuse first, hoist second). Inline-fuse is the greedy
default because it has a smaller smem footprint and works on every
architecture; hoist is the per-recipe autotune win on sm_90+ with TMA.
Autotune-side variant ordering / pinning is the autotuner's job, not
this pass's.

Scope (conservative for the first cut):

- Single Tile.
- Single reduce Loop inside the Tile body (or inside a free outer Loop).
- Cone is identified per stage-cache-axes group: a group of ≥ 2 Stages
  with identical ``axes`` whose dependent SSAs in the reduce body form
  a coherent dataflow region (only ``Load`` / ``Assign``, no ``Cond`` /
  nested loops) and whose ``free_vars`` are a subset of the cache axes.
- Cone must have exactly one boundary SSA (the value consumed by the
  reduce-body region outside the cone). Multi-output cones are skipped.

Runs once per TileOp (idempotence via the ``FUSED_PIPELINE`` knob
stamp). Runs after ``006a_register_tile_planned`` — the F-axis
replication is already applied, so the cone analysis sees the final
per-cell layout.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.expr import Expr, Var
from deplodock.compiler.ir.stmt import Assign, Body, Load, Loop, Stmt, Write
from deplodock.compiler.ir.tile.ir import ComputeStage, Stage, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

PATTERN = [Pattern("root", TileOp)]

FUSED_PIPELINE = Knob(
    "FUSED_PIPELINE",
    KnobType.BOOL,
    hints=(True, False),
    help=(
        "Hoist invariant compute as a ComputeStage between transport Stages and the "
        "reduce body (True) so 010/011/013 can double-buffer / TMA / async the "
        "transports and 015 can pipeline. False (default) inlines the compute into a "
        "multi-source Stage, saving smem at the cost of A/B async transport."
    ),
)


_VARIANT_ORDER: tuple[bool, ...] = (False, True)
"""Fixed emission order for the FUSED_PIPELINE fork: inline-fuse first
(greedy default — smaller smem, works everywhere), hoist second."""


def rewrite(root: Node) -> list[TileOp] | None:
    """Emit both ``FUSED_PIPELINE`` polarities in a fixed order. Greedy
    picks the first (inline-fuse); autotune searches both."""
    if FUSED_PIPELINE.name in root.op.knobs:
        raise RuleSkipped("hoist already applied (idempotence via knob)")

    variants: list[TileOp] = []
    for fused_pipeline in _VARIANT_ORDER:
        new_body = _maybe_rewrite(root.op.body, fused_pipeline=fused_pipeline)
        if new_body is None:
            continue
        knobs = {**root.op.knobs, FUSED_PIPELINE.name: fused_pipeline}
        variants.append(TileOp(body=new_body, name=root.op.name, knobs=knobs))
    if not variants:
        raise RuleSkipped("no fusable epilogue cone found")
    return variants


def _maybe_rewrite(body: Body, *, fused_pipeline: bool) -> Body | None:
    idx, tile = single_tile(body)
    new_tile_body = _process_scope(tile.body, fused_pipeline=fused_pipeline)
    if new_tile_body is None:
        return None
    new_tile = dc_replace(tile, body=new_tile_body)
    return body[:idx] + (new_tile,) + body[idx + 1 :]


def _process_scope(scope_body: Body, *, fused_pipeline: bool) -> Body | None:
    """Find a reduce-Loop scope with Stages preceding it (the typical
    007 layout) and try to hoist. Recurse into free outer Loops if no
    cone at this level."""
    stages: list[Stage] = []
    reduce_loop: Loop | None = None
    reduce_loop_idx: int | None = None
    for i, s in enumerate(scope_body):
        if isinstance(s, Stage):
            stages.append(s)
        elif isinstance(s, Loop) and s.is_reduce and stages:
            reduce_loop = s
            reduce_loop_idx = i
            break
        elif isinstance(s, Loop) and not s.is_reduce:
            inner = _process_scope(s.body, fused_pipeline=fused_pipeline)
            if inner is not None:
                new_outer = dc_replace(s, body=inner)
                return Body(scope_body[:i] + (new_outer,) + scope_body[i + 1 :])
        else:
            # Anything else (Combine, AsyncWait, etc.) resets the staged
            # group — we only hoist cones whose producers are the
            # immediate prefix of a reduce Loop.
            stages = []

    if reduce_loop is None or reduce_loop_idx is None:
        return None

    cone_info = _try_find_cone(stages, reduce_loop)
    if cone_info is None:
        return None
    group, group_names, cone, boundary_name, _ssa_def = cone_info

    if fused_pipeline:
        new_reduce_body, new_compute = _emit_hoist_compute(group, group_names, reduce_loop.body, cone, boundary_name)
        # Hoist-compute: keep every transport Stage in its original slot;
        # the compute Stage slots between the transports and the reduce.
        surviving_stages: tuple[Stage, ...] = tuple(stages)
    else:
        new_reduce_body, new_compute = _emit_inline_fuse(group, group_names, reduce_loop.body, cone, boundary_name)
        # Inline-fuse: group transports are absorbed by the fused Stage;
        # non-group transports (e.g. w_smem) survive in their original slots.
        surviving_stages = tuple(st for st in stages if st.name not in group_names)

    new_loop = dc_replace(reduce_loop, body=new_reduce_body)
    pre = scope_body[: reduce_loop_idx - len(stages)]
    return Body(pre + surviving_stages + (new_compute, new_loop) + scope_body[reduce_loop_idx + 1 :])


# ---------------------------------------------------------------------------
# Cone analysis (shared between emitters)
# ---------------------------------------------------------------------------


def _try_find_cone(stages: list[Stage], reduce_loop: Loop) -> tuple[list[Stage], frozenset[str], set[str], str, dict[str, Stmt]] | None:
    """Identify the largest fusable cone in ``reduce_loop.body`` whose
    producer set is a group of Stages with identical cache axes.
    Returns ``(group, group_names, cone, boundary_name, ssa_def)`` or
    ``None`` if no fusion is possible."""
    # Group stages by their cache-axis tuple. Only groups with ≥ 2 stages
    # benefit from fusion — one stage means there's already a single
    # cooperative load.
    groups: dict[tuple[str, ...], list[Stage]] = {}
    for st in stages:
        key = tuple(ax.name for ax in st.axes)
        groups.setdefault(key, []).append(st)

    candidates = [(key, grp) for key, grp in groups.items() if len(grp) >= 2]
    if not candidates:
        return None

    ssa_stage_deps, ssa_free_vars, ssa_def = _ssa_dataflow(reduce_loop.body)

    for cache_axes_tuple, group in candidates:
        group_names = frozenset(st.name for st in group)
        cache_axes_set = frozenset(cache_axes_tuple)

        cone: set[str] = set()
        for name, deps in ssa_stage_deps.items():
            if not deps:
                continue
            if deps <= group_names and ssa_free_vars[name] <= cache_axes_set:
                cone.add(name)

        if not cone:
            continue

        boundary = _find_boundary(reduce_loop.body, cone)
        if len(boundary) != 1:
            continue
        boundary_name = next(iter(boundary))

        # The cone's boundary must be an Assign output, not a raw smem
        # Load. A Load boundary means no compute happens between the
        # gmem source and the consumer — hoisting would just write the
        # gmem value into a new smem buffer with no benefit. Defer to
        # the existing single-stage staging path.
        boundary_def = ssa_def[boundary_name]
        if not isinstance(boundary_def, Assign):
            continue

        return group, group_names, cone, boundary_name, ssa_def

    return None


def _ssa_dataflow(
    body: Body,
) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]], dict[str, Stmt]]:
    """Per-SSA-name: (transitive set of source-stage names it depends on,
    free-axis Vars it references, the defining stmt). Operates over a
    linear reduce body (no nested control flow, which is asserted by the
    cone shape check)."""
    stage_deps: dict[str, frozenset[str]] = {}
    free_vars: dict[str, frozenset[str]] = {}
    defs: dict[str, Stmt] = {}
    for s in body:
        if isinstance(s, Load):
            defs[s.name] = s
            stage_deps[s.name] = frozenset({s.input})
            fv: frozenset[str] = frozenset()
            for e in s.index:
                fv = fv | frozenset(e.free_vars())
            free_vars[s.name] = fv
        elif isinstance(s, Assign):
            defs[s.name] = s
            d: frozenset[str] = frozenset()
            fv = frozenset()
            for arg in s.args:
                d = d | stage_deps.get(arg, frozenset())
                fv = fv | free_vars.get(arg, frozenset())
            stage_deps[s.name] = d
            free_vars[s.name] = fv
    return stage_deps, free_vars, defs


def _find_boundary(body: Body, cone: set[str]) -> set[str]:
    """Boundary = cone-internal SSAs consumed by a stmt that isn't in
    the cone. Uses ``Stmt.deps()`` so every consumer type (Assign /
    Accum / Write / Select / Cond / …) is covered uniformly."""
    boundary: set[str] = set()
    for s in body:
        producer = isinstance(s, (Load, Assign)) and getattr(s, "name", None) in cone
        if producer:
            continue
        for d in s.deps():
            if d in cone:
                boundary.add(d)
    return boundary


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------


def _emit_inline_fuse(
    group: list[Stage],
    group_names: frozenset[str],
    reduce_body: Body,
    cone: set[str],
    boundary_name: str,
) -> tuple[Body, Stage]:
    """Build a single multi-source Stage carrying the gmem source Loads
    + cone Assigns + Write. The reduce body's source-stage smem Loads
    collapse to one Load on the fused stage. Returns
    ``(new_reduce_body, fused_stage)`` — the fused stage absorbs all
    source transports."""
    fused_name = "_".join(st.name for st in group) + "_fused"
    fused_axes = group[0].axes  # identical across the group

    fused_body_stmts: list[Stmt] = []

    # Map smem-Load name in reduce body → gmem-Load name in fused body.
    # Pull each source Stage's gmem Load verbatim into the fused body,
    # renaming to ``<stage.name>__gmem`` so the cone Assigns can rewire.
    smem_to_fused_load: dict[str, str] = {}
    for st in group:
        primary = st.primary_load
        gmem_load = Load(name=f"{st.name}__gmem", input=primary.input, index=primary.index)
        fused_body_stmts.append(gmem_load)
    for s in reduce_body:
        if isinstance(s, Load) and s.input in group_names:
            stage = next(st for st in group if st.name == s.input)
            smem_to_fused_load[s.name] = f"{stage.name}__gmem"

    # Copy cone Assigns in body order, σ-rewriting args to the gmem-Load
    # SSA names.
    for s in reduce_body:
        if isinstance(s, Assign) and s.name in cone:
            new_args = tuple(smem_to_fused_load.get(a, a) for a in s.args)
            fused_body_stmts.append(Assign(name=s.name, op=s.op, args=new_args))

    cache_index: tuple[Expr, ...] = tuple(Var(ax.name) for ax in fused_axes)
    fused_body_stmts.append(Write(output=fused_name, index=cache_index, value=boundary_name))
    fused_stage = Stage(name=fused_name, axes=fused_axes, body=Body(tuple(fused_body_stmts)))

    new_reduce_body = _rewrite_reduce_body(reduce_body, group_names, cone, boundary_name, fused_name, cache_index)
    return new_reduce_body, fused_stage


def _emit_hoist_compute(
    group: list[Stage],
    group_names: frozenset[str],
    reduce_body: Body,
    cone: set[str],
    boundary_name: str,
) -> tuple[Body, ComputeStage]:
    """Build a ``ComputeStage`` that reads ``group``'s smem (cache-local
    index), runs the cone Assigns, writes its own smem. Reduce body's
    source-stage Loads collapse to one Load on the compute stage's
    output. Returns ``(new_reduce_body, compute_stage)`` — the source
    transports stay live; caller decides where the compute Stage lands."""
    fused_name = "_".join(st.name for st in group) + "_fused"
    fused_axes = group[0].axes
    cache_index: tuple[Expr, ...] = tuple(Var(ax.name) for ax in fused_axes)

    # Compute body: smem Loads on each source Stage at cache-local
    # indices + cone Assigns + Write into the compute stage's own buffer.
    # The Loads keep their reduce-body SSA names so cone Assigns copy
    # verbatim without arg renaming.
    compute_body_stmts: list[Stmt] = []
    for s in reduce_body:
        if isinstance(s, Load) and s.input in group_names:
            compute_body_stmts.append(Load(name=s.name, input=s.input, index=cache_index))
    for s in reduce_body:
        if isinstance(s, Assign) and s.name in cone:
            compute_body_stmts.append(s)
    compute_body_stmts.append(Write(output=fused_name, index=cache_index, value=boundary_name))

    compute_stage = ComputeStage(name=fused_name, axes=fused_axes, body=Body(tuple(compute_body_stmts)))

    new_reduce_body = _rewrite_reduce_body(reduce_body, group_names, cone, boundary_name, fused_name, cache_index)
    return new_reduce_body, compute_stage


def _rewrite_reduce_body(
    reduce_body: Body,
    group_names: frozenset[str],
    cone: set[str],
    boundary_name: str,
    fused_name: str,
    cache_index: tuple[Expr, ...],
) -> Body:
    """Drop cone stmts and source-stage smem Loads from the reduce body;
    replace them with a single Load on ``fused_name`` (the cone's output
    buffer) using the boundary SSA name so downstream consumers don't
    need rewiring. Shared between both emitters since the reduce body
    looks the same either way."""
    new_stmts: list[Stmt] = []
    boundary_emitted = False
    for s in reduce_body:
        if isinstance(s, Load) and s.input in group_names:
            if not boundary_emitted:
                new_stmts.append(Load(name=boundary_name, input=fused_name, index=cache_index))
                boundary_emitted = True
            continue
        if isinstance(s, Assign) and s.name in cone:
            continue
        new_stmts.append(s)
    if not boundary_emitted:
        new_stmts.insert(0, Load(name=boundary_name, input=fused_name, index=cache_index))
    return Body(tuple(new_stmts))
