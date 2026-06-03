"""Hoist staged K-pipeline above the masked-tile boundary ``Cond``.

When ``010_partition_loops`` emits a non-divisor / symbolic output axis it
wraps the parallel-tile body in a boundary ``Cond(decoded_coord < bound,
body=...)`` so masked output cells skip their ``Write``. ``020_stage_inputs``
then runs transparently through the ``Cond`` and produces a body with a
``StageBundle`` sitting INSIDE the Cond:

    Cond(M_decoded < M, body=[
        StageBundle(stages=..., body=[ ... K_o reduce ... ]),
        Write(...),
    ])

That arrangement is incorrect for the cooperative-load transports we
promote to downstream:

- **TMA** elects a single issuer thread per group; if that thread happens
  to be in the if-false branch, ``cp.async.bulk`` never fires and the
  consumer ``mbarrier`` deadlocks.
- **cp.async** needs every thread in the CTA to fetch its lane of the
  slab; threads gated out by the boundary skip the issue.

This pass rearranges the guard so the cooperative load runs for every
thread regardless of the mask. The K-pipeline (``StageBundle`` itself, and
any ``SerialTile`` / ``StridedTile`` whose subtree contains a
``StageBundle``) is hoisted ABOVE the ``Cond``; the ``Write`` and other
output-emitting stmts stay inside, so masked threads still skip their
output. A few extra FMAs run on masked-row accumulators each K iteration
— benign because the ``Write`` that would emit them is still guarded.

Un-staged gmem Loads in the hoisted body whose index references a var the
boundary predicate gates (e.g. the qwen lmhead linear `wl` Load against
the masked-N coord) would now run unconditionally and read OOB. Each such
Load is wrapped in an inner ``Cond(predicate, body=cone)`` covering its
forward SSA cone (``Accum`` targets cross the inner Cond by Cond's
existing SSA rules, matching ``Loop``).

**Pattern preconditions** — both must hold for a Cond to be lifted:

1. The predicate is a ``BinaryExpr`` with op ``<``. This excludes the
   ``==`` SPLITK invariant-compute Cond that ``010_partition_loops``
   emits for the post-reduce linear epilogue (``Cond(K_s == 0, ...)``);
   hoisting inside an ``==`` guard would re-execute the cooperative load
   on the CTAs the guard meant to skip.
2. The Cond's body, after the hoist partition, contains a ``StageBundle``
   somewhere (recursively). Bare Conds with no staged transport are
   left intact.

**Idempotence.** Self-shaped: a second invocation on already-hoisted IR
sees no ``Cond > StageBundle`` and falls through. No knob.

**History.** Lived inline in ``020_stage_inputs._process_scope`` until it
was extracted so 020 became a uniform staging walk and this Cond-shape
rewrite a focused one.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.expr import BinaryExpr
from deplodock.compiler.ir.stmt import Body, Cond, Load, Stmt
from deplodock.compiler.ir.tile.ir import (
    SerialTile,
    Stage,
    StageBundle,
    StridedTile,
    TileOp,
)
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(ctx: Context, root: Node) -> list[TileOp] | None:
    """Rewrite ``root.op`` lifting every masked-tile boundary ``Cond``'s
    K-pipeline above the Cond. Deterministic single-variant pass; raises
    ``RuleSkipped`` when no Cond in the body matches both preconditions.
    """
    input_shapes = _static_input_shapes(root.op)

    def lift(s: Stmt) -> Stmt | tuple[Stmt, ...]:
        return _lift_if_match(s, input_shapes)

    new_body = root.op.body.map(lift)
    if new_body == root.op.body:
        raise RuleSkipped("no masked-tile Cond > StageBundle to lift")
    return [TileOp(body=new_body, name=root.op.name, knobs=root.op.knobs)]


def _static_input_shapes(op) -> dict[str, tuple[int, ...]]:  # noqa: ANN001 — TileOp
    """``{gmem buffer name → static per-dim shape}`` for every kernel input,
    used to stamp ``Source.gmem_extents`` so the hoisted cooperative load can
    clamp its gmem read to the buffer bounds. Buffers with any symbolic
    (non-static) dim are skipped — a runtime-extent clamp is M9 follow-up;
    today only static-shaped weights hit the masked-overhang path."""
    shapes: dict[str, tuple[int, ...]] = {}
    for buf, tensor in op.inputs.items():
        dims = getattr(tensor, "shape", None)
        if dims is None:
            continue
        if all(getattr(d, "is_static", False) for d in dims):
            shapes[buf] = tuple(d.as_static() for d in dims)
    return shapes


def _lift_if_match(s: Stmt, input_shapes: dict[str, tuple[int, ...]]) -> Stmt | tuple[Stmt, ...]:
    """``Body.map`` callback. ``s`` arrives with its nested bodies already
    rewritten (post-order), so any Cond nested inside this one has had its
    own lift applied. Returns ``s`` verbatim when the preconditions don't
    hold, or a tuple ``(*hoisted_with_guards, residual_cond?)`` when they
    do — the tuple is spliced into the parent body in place of the Cond.
    """
    if not isinstance(s, Cond):
        return s
    if not (isinstance(s.cond, BinaryExpr) and s.cond.op == "<"):
        return s
    hoisted: list[Stmt] = []
    inside: list[Stmt] = []
    for sub in s.body:
        if _is_k_pipeline_stmt(sub):
            hoisted.append(sub)
        else:
            inside.append(sub)
    if not _contains_stage_bundle(Body(tuple(hoisted))):
        return s
    gated_vars = frozenset(s.cond.free_vars())
    hoisted = [_guard_unsafe_loads(h, s.cond, gated_vars) for h in hoisted]
    # Stamp ``gmem_extents`` on every cooperative-load Source being hoisted
    # above the boundary: now that the producer runs for all threads
    # (including the overhang past the masked extent), its gmem read must be
    # clamped to the buffer bounds or it reads OOB. ``_stage_expand.emit_stage``
    # applies the clamp at materialize. See the Source.gmem_extents docstring.
    hoisted = [_stamp_gmem_extents(h, input_shapes) for h in hoisted]
    out: list[Stmt] = list(hoisted)
    if inside or s.else_body:
        out.append(Cond(cond=s.cond, body=Body(tuple(inside)), else_body=s.else_body))
    return tuple(out)


def _stamp_gmem_extents(stmt: Stmt, input_shapes: dict[str, tuple[int, ...]]) -> Stmt:
    """Recursively rewrite ``stmt`` so every ``StageBundle`` Source whose
    ``buf`` is a static-shaped kernel input carries ``gmem_extents``. This
    pass runs before ``030_hoist_invariant_compute``, so no bundle carries a
    ``compute`` phase yet — every Source here is a gmem transport operand.
    Both affine and template (reshape) addressings are stamped: a masked
    weight's smem slab
    is often template-addressed (the very case that overruns).
    ``_stage_expand._clamp_source_index`` handles both index shapes — per-dim
    when the ``source_index`` rank matches ``gmem_extents``, and a single
    flat ``< ∏extents`` clamp when the template collapsed the index to one
    dim — so the OOB is caught either way."""
    if isinstance(stmt, StageBundle):
        new_stages: list[Stage] = []
        for stage in stmt.stages:
            new_sources = tuple(
                replace(src, gmem_extents=input_shapes[src.buf]) if src.buf in input_shapes and src.gmem_extents is None else src
                for src in stage.sources
            )
            new_stages.append(replace(stage, sources=new_sources))
        new_body = Body(tuple(_stamp_gmem_extents(s, input_shapes) for s in stmt.body))
        return replace(stmt, stages=tuple(new_stages), body=new_body)
    nested = stmt.nested()
    if not nested:
        return stmt
    new_bodies = tuple(Body(tuple(_stamp_gmem_extents(s, input_shapes) for s in body)) for body in nested)
    return stmt.with_bodies(new_bodies)


def _is_k_pipeline_stmt(stmt: Stmt) -> bool:
    """Identify the stmts the masked-tile Cond hoist should pull above
    the boundary guard. The K-pipeline structure stage_inputs produces
    inside a Cond is a single ``SerialTile`` (K-outer) whose body
    contains a ``StageBundle`` (the cooperative load). After downstream
    pipelining (080) that may also expand to one prologue ``StageBundle``
    + the K-outer + a trailing ``AsyncWait`` / tail ``SerialTile``;
    matching ``StageBundle``-bearing stmts (recursively) plus bare
    ``StageBundle`` siblings covers both shapes. Everything else —
    ``Write`` outputs, ``Cond(a0==0)`` invariant-compute guards from
    ``030_hoist_invariant_compute``, constant init ``Assign``s — stays
    inside the original Cond so the boundary predicate keeps guarding
    output emission."""
    if isinstance(stmt, StageBundle):
        return True
    if isinstance(stmt, (SerialTile, StridedTile)) and _contains_stage_bundle(stmt.body):
        return True
    return False


def _contains_stage_bundle(body: Body) -> bool:
    """Recursive: ``True`` iff any stmt inside ``body`` (at any nesting
    depth) is a ``StageBundle``. Used by the masked-tile Cond hoist to
    decide whether to split the Cond — only worth doing when the inner
    body actually picked up a cooperative-load bundle to hoist."""
    for s in body:
        if isinstance(s, StageBundle):
            return True
        for nb in s.nested():
            if _contains_stage_bundle(nb):
                return True
    return False


def _collect_smem_names(stmt: Stmt) -> set[str]:
    """Names defined as ``Source`` slabs across every ``StageBundle`` in
    ``stmt``'s subtree. Loads on these names are smem reads, dimensioned
    to the per-cell cache extents — they cannot go OOB regardless of the
    cell's position in the output grid, so ``_guard_unsafe_loads`` skips
    them."""
    names: set[str] = set()
    if isinstance(stmt, StageBundle):
        for stage in stmt.stages:
            for src in stage.sources:
                names.add(src.name)
    for body in stmt.nested():
        for s in body:
            names |= _collect_smem_names(s)
    return names


def _guard_unsafe_loads(stmt: Stmt, predicate, gated_vars: frozenset[str], smem_names: frozenset[str] | None = None) -> Stmt:
    """Walk ``stmt`` recursively. For each leaf body containing a direct
    gmem ``Load`` whose index references any var in ``gated_vars`` (i.e.,
    a Load that would read OOB on threads where ``predicate`` is False),
    wrap the Load + its forward SSA cone in ``Cond(predicate, body=cone)``.

    Smem Loads (``Load.input`` ∈ ``smem_names`` collected from every
    StageBundle in the hoist subtree) are skipped — the smem slab is
    sized to per-cell cache extents, so reads at register-tile coords
    are always in-bounds even when the cell is masked.

    ``Accum`` targets cross the inner Cond boundary by Cond's SSA rules
    (matching ``Loop``'s reduce semantics) — so masked threads skip the
    Load + their cell's FMA + their cell's Accum increment, leaving the
    accumulator at its zero-initialised value. The downstream ``Write`` is
    still guarded by the outer boundary Cond, so the zero (or whatever
    state the accumulator landed in) is never emitted.

    Body shape rewrite, when a leaf body has any unsafe Load:
        before: [safe_stmt, unsafe_load, dependent_assign, dependent_accum, …]
        after:  [safe_stmt, Cond(pred, [unsafe_load, dependent_assign,
                                       dependent_accum])]
    """
    if smem_names is None:
        smem_names = frozenset(_collect_smem_names(stmt))
    nested = stmt.nested()
    if not nested:
        return stmt
    new_bodies = []
    for body in nested:
        new_bodies.append(_guard_unsafe_loads_in_body(body, predicate, gated_vars, smem_names))
    return stmt.with_bodies(tuple(new_bodies))


def _guard_unsafe_loads_in_body(body: Body, predicate, gated_vars: frozenset[str], smem_names: frozenset[str]) -> Body:
    """Apply the per-Load guard inside ``body`` and recurse into nested
    bodies of any block stmts that aren't leaf bodies themselves."""
    stmts = list(body)
    # First recurse into each stmt's nested bodies (descent first so
    # leaf-level guards are placed at the right scope).
    stmts = [_guard_unsafe_loads(s, predicate, gated_vars, smem_names) for s in stmts]

    # Find unsafe gmem Loads at THIS body level (smem Loads skip the
    # check — smem extents already match the per-cell cache shape).
    unsafe_idxs: list[int] = []
    for i, s in enumerate(stmts):
        if not isinstance(s, Load):
            continue
        if s.input in smem_names:
            continue
        load_vars: set[str] = set()
        for e in s.index:
            load_vars |= e.free_vars()
        if load_vars & gated_vars:
            unsafe_idxs.append(i)
    if not unsafe_idxs:
        return Body(tuple(stmts))

    # Compute forward SSA cone seeded by the unsafe Loads' defined names.
    # A stmt joins the cone when any of its ``deps()`` is already in the
    # cone's defined-names set. Iterate to fixpoint.
    cone_names: set[str] = set()
    cone_idxs: set[int] = set(unsafe_idxs)
    for i in unsafe_idxs:
        cone_names.update(stmts[i].defines())
    changed = True
    while changed:
        changed = False
        for i, s in enumerate(stmts):
            if i in cone_idxs:
                continue
            if any(d in cone_names for d in s.deps()):
                cone_idxs.add(i)
                cone_names.update(s.defines())
                changed = True

    # Wrap the contiguous range covering all cone stmts in a Cond.
    # The cone is typically contiguous for matmul-style bodies (Load
    # immediately followed by its consumer ``Assign`` + ``Accum``).
    # Non-cone stmts that happen to sit between the first and last cone
    # index come along for the ride — they're scoped to the inner Cond
    # too. Acceptable: those middle stmts have no gated-axis deps (else
    # they'd be in the cone), so wrapping them is a no-op semantically.
    lo, hi = min(cone_idxs), max(cone_idxs)
    cone_stmts = tuple(stmts[lo : hi + 1])
    wrapped = Cond(cond=predicate, body=Body(cone_stmts))
    return Body((*stmts[:lo], wrapped, *stmts[hi + 1 :]))
