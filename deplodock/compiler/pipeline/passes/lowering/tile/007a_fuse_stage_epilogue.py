"""Fuse loop-invariant producer chains into the upstream Stage body.

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

In the silu-gated MLP shape (``F.silu(gate)*up @ W``), the chain
``v0..v_silu`` depends only on ``stage_A`` + ``stage_B`` (cache axes
``{M, K_inner}``); it has no dependency on ``N`` (the consumer thread
axis). Today that chain re-runs every ``N``-thread, every ``N``-tile —
~896× redundancy on Qwen-MLP shapes.

This pass detects such cones and folds them into the producer Stage.
The fused Stage's body becomes a multi-source program::

    fused = Stage(name=…, axes=(M, K_inner), body=(
        Load(input=A, index=…),
        Load(input=B, index=…),
        Assign(silu chain over A's load),
        Assign(multiply with B's load),                 # the cone's frontier
        Write(output=fused, index=(M, K_inner), value=<frontier>),
    ))

``stage_A`` and ``stage_B`` are removed; the consumer reduce body's cone
collapses to a single Load from ``fused``.

Scope (conservative for the first cut):

- Single Tile.
- Single reduce Loop inside the Tile body (or inside a free outer Loop).
- Cone is identified per stage-cache-axes group: a group of ≥ 2 Stages
  with identical ``axes`` whose dependent SSAs in the reduce body form
  a coherent dataflow region (only ``Load`` / ``Assign``, no ``Cond`` /
  nested loops) and whose ``free_vars`` are a subset of the cache axes.
- Cone must have exactly one boundary SSA (the value consumed by the
  reduce-body region outside the cone). Multi-output cones are skipped.

Runs once per TileOp (the rewriter returns ``RuleSkipped`` if no cone
was found). Must run **before** ``008_register_tile`` — that pass
introduces F-axis split and would tangle cones across register tiles.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import Expr, Var
from deplodock.compiler.ir.stmt import Assign, Body, Load, Loop, Stmt, Write
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

PATTERN = [Pattern("root", TileOp)]


def rewrite(root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        raise RuleSkipped("no fusable epilogue cone found")
    return TileOp(body=new_body, name=root.op.name)


def _maybe_rewrite(body: Body) -> Body | None:
    idx, tile = single_tile(body)
    new_tile_body = _process_scope(tile.body)
    if new_tile_body is None:
        return None
    new_tile = dc_replace(tile, body=new_tile_body)
    return body[:idx] + (new_tile,) + body[idx + 1 :]


def _process_scope(scope_body: Body) -> Body | None:
    """Find a reduce-Loop scope that has Stages preceding it (the
    typical 007 layout) and try to fuse. If none found at this level,
    recurse into free outer Loops."""
    # Pattern: a sequence of Stages immediately followed by a reduce Loop.
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
            # Free outer loop — recurse into its body.
            inner = _process_scope(s.body)
            if inner is not None:
                new_outer = dc_replace(s, body=inner)
                return Body(scope_body[:i] + (new_outer,) + scope_body[i + 1 :])
        else:
            # Anything else (Combine, AsyncWait, etc.) resets the staged
            # group — we only fuse stages that are immediate prefix of a
            # reduce Loop.
            stages = []

    if reduce_loop is None or reduce_loop_idx is None:
        return None

    # Try to identify a fusable cone within this reduce body.
    fused = _try_fuse(stages, reduce_loop)
    if fused is None:
        return None
    new_stages, new_reduce_body, fused_stage = fused
    new_loop = dc_replace(reduce_loop, body=new_reduce_body)
    # Reassemble: stages before the fused stage's slot, fused stage, then
    # the remaining (non-source) stages, then the rewritten reduce loop.
    pre = scope_body[: reduce_loop_idx - len(stages)]
    return Body(pre + tuple(new_stages) + (new_loop,) + scope_body[reduce_loop_idx + 1 :])


def _try_fuse(stages: list[Stage], reduce_loop: Loop) -> tuple[list[Stage], Body, Stage] | None:
    """Identify the largest fusable cone in ``reduce_loop.body`` whose
    producer set is a group of Stages with identical cache axes.
    Returns the rewritten ``(stages_list, reduce_body, fused_stage)`` or
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

    # SSA dataflow on the reduce body.
    ssa_stage_deps, ssa_free_vars, ssa_def = _ssa_dataflow(reduce_loop.body)

    for cache_axes_tuple, group in candidates:
        group_names = frozenset(st.name for st in group)
        cache_axes_set = frozenset(cache_axes_tuple)

        # The cone: SSAs whose Loads (transitive) all come from this group
        # AND whose free vars are within the cache-axis set. A name with
        # no stage deps (e.g. an SSA computed entirely from cache-axis
        # Vars without a Load) is not part of any cone.
        cone: set[str] = set()
        for name, deps in ssa_stage_deps.items():
            if not deps:
                continue
            if deps <= group_names and ssa_free_vars[name] <= cache_axes_set:
                cone.add(name)

        if not cone:
            continue

        # Boundary: cone SSAs consumed by any stmt outside the cone.
        boundary = _find_boundary(reduce_loop.body, cone)
        if len(boundary) != 1:
            # Multi-output cones (e.g. two independent silu chains feeding
            # different consumers) need more bookkeeping — skip for now.
            continue
        boundary_name = next(iter(boundary))

        # The cone's *boundary* must be an Assign output, not a raw
        # smem Load. A Load boundary means no compute happens between
        # the gmem source and the consumer — fusion would just write
        # the gmem value into a new smem buffer with no benefit. Defer
        # to the existing single-stage staging path.
        boundary_def = ssa_def[boundary_name]
        if not isinstance(boundary_def, Assign):
            continue

        # Build the fused stage and rewrite the reduce body.
        fused_stage, new_reduce_body = _build_fusion(group, group_names, reduce_loop.body, cone, boundary_name, ssa_def)
        # Replace the source stages in the outer list with the fused
        # stage; keep any other stages (from other groups) as-is.
        new_stages: list[Stage] = []
        emitted_fused = False
        for st in stages:
            if st.name in group_names:
                if not emitted_fused:
                    new_stages.append(fused_stage)
                    emitted_fused = True
            else:
                new_stages.append(st)
        return new_stages, new_reduce_body, fused_stage

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
    Accum / Write / Select / Cond / …) is covered uniformly — easy to
    miss one if we enumerate by hand."""
    boundary: set[str] = set()
    for s in body:
        producer = isinstance(s, (Load, Assign)) and getattr(s, "name", None) in cone
        if producer:
            continue
        for d in s.deps():
            if d in cone:
                boundary.add(d)
    return boundary


def _build_fusion(
    group: list[Stage],
    group_names: frozenset[str],
    reduce_body: Body,
    cone: set[str],
    boundary_name: str,
    ssa_def: dict[str, Stmt],
) -> tuple[Stage, Body]:
    """Construct the fused Stage and the rewritten reduce body.

    Cone SSAs come from the reduce body; their backward closure includes
    Loads on stage smem buffers. In the fused stage's body we replace
    those smem Loads with the corresponding gmem Loads (lifted from
    each source stage's own body).
    """
    # Map smem_name → its source stage's body Load (the gmem load that
    # filled it). Lift the source stage's Load(stage.name=…) verbatim;
    # σ-rewrite its index from per-stage cache vars to the fused stage's
    # cache vars (here the names match since the group shares ``axes``,
    # so no rewrite needed beyond unique SSA names).
    fused_name = "_".join(st.name for st in group) + "_fused"
    fused_axes = group[0].axes  # identical across the group

    # Body: gmem Loads from each source stage's primary Load, then the
    # cone's Assigns in original order, then a terminal Write from the
    # boundary SSA into the fused smem buffer.
    fused_body_stmts: list[Stmt] = []

    # SSA rename map: smem-Load name in reduce body → fresh name we use
    # in the fused body (to avoid colliding with cone Assigns we'll copy
    # over).
    smem_to_fused_load: dict[str, str] = {}

    # For each source Stage, the body has shape Load(gmem)→Write(smem).
    # We pull the Load verbatim, renaming its SSA to <stage.name>__gmem.
    for st in group:
        primary = st.primary_load
        gmem_load = Load(name=f"{st.name}__gmem", input=primary.input, index=primary.index)
        fused_body_stmts.append(gmem_load)

    # Map: each reduce-body Load(input=stage.name) → the gmem-load name
    # we just emitted. Multiple smem Loads on the same stage map to the
    # same gmem name (the gmem value is what they all read).
    for s in reduce_body:
        if isinstance(s, Load) and s.input in group_names:
            stage = next(st for st in group if st.name == s.input)
            smem_to_fused_load[s.name] = f"{stage.name}__gmem"

    # Copy cone Assigns in body order, σ-rewriting any arg names that
    # refer to smem Loads → the gmem-Load SSA names.
    for s in reduce_body:
        if isinstance(s, Assign) and s.name in cone:
            new_args = tuple(smem_to_fused_load.get(a, a) for a in s.args)
            fused_body_stmts.append(Assign(name=s.name, op=s.op, args=new_args))

    # Terminal Write: cache-local index, value = boundary SSA. The
    # boundary SSA is a cone Assign already in the body above.
    cache_index: tuple[Expr, ...] = tuple(Var(ax.name) for ax in fused_axes)
    fused_body_stmts.append(Write(output=fused_name, index=cache_index, value=boundary_name))
    fused_body = Body(tuple(fused_body_stmts))

    fused_stage = Stage(name=fused_name, axes=fused_axes, body=fused_body)

    # Rewrite the reduce body: drop cone stmts (their compute now lives
    # in the fused stage), drop reduce-body Loads whose input is a source
    # stage (they're folded into the fused stage too), and replace the
    # boundary name's value with a fresh Load from the fused stage's
    # smem buffer.
    new_reduce_stmts: list[Stmt] = []
    boundary_emitted = False
    boundary_load_name = boundary_name  # reuse the SSA name for callers
    for s in reduce_body:
        if isinstance(s, Load) and s.input in group_names:
            # All smem Loads from source stages are absorbed; the
            # boundary load (a single fused-stage smem Load) replaces
            # them. Emit it the first time we'd have emitted any source
            # smem Load — keeps original ordering of downstream stmts.
            if not boundary_emitted:
                new_reduce_stmts.append(Load(name=boundary_load_name, input=fused_name, index=cache_index))
                boundary_emitted = True
            continue
        if isinstance(s, Assign) and s.name in cone:
            # Cone Assign moved into fused stage; drop here.
            continue
        new_reduce_stmts.append(s)

    if not boundary_emitted:
        new_reduce_stmts.insert(0, Load(name=boundary_load_name, input=fused_name, index=cache_index))

    return fused_stage, Body(tuple(new_reduce_stmts))
