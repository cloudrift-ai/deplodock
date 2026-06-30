"""Fuse sibling masked-cell ``Cond`` runs left by ``010_split_register_axes``.

When ``010`` replicates a ``RegisterTile(N_r)`` whose body is wrapped in a
``Cond`` with an N_r-dependent predicate (the masked-tile overhang on a
non-divisor output extent — e.g. vocab=151669 and FN=64), it emits F× sibling
``Cond`` blocks at the same scope, each containing the **whole** loop nest
including the N-invariant prologue (RMSNorm Loads, the per-iter weight Load,
the per-iter multiply). ``011_dedup_replicated`` then dedups **within** a
scope but never **across** sibling scopes, so on a 64-cell linear+rmsnorm
fused matmul the four-Stmt RMSNorm prologue chain duplicates 64× — the
unrolled NVRTC body explodes (1000+ lines for one kernel; cicc -O1 spends
5-6 s on it; the autotune watchdog times the variant out at the 2 s compile
budget or the 6 s wall budget on the assembled kernel).

This pass is the cross-sibling complement to ``011``. It walks each scope
looking for runs of ≥ 2 sibling ``Cond``s whose bodies are structured
``[Loop[a5] [Loop[a4] [<invariant prefix>, <per-cell tail>]], <per-cell
post-loop stmts>]``. For each run it drills through the matching loop
nest, finds the longest common **prefix** of structurally-identical stmts
at the innermost level, and rebuilds as:

    Loop[a5]:                       # shared loop nest, lifted out of the Conds
        Loop[a4]:
            <common prefix>          # the N-invariant Loads + Assigns, once
            Cond(P_0):
                <residual_0>          # cell 0's N-dep inner tail
            Cond(P_1):
                <residual_1>          # cell 1's N-dep inner tail
            …
    Cond(P_0):
        <post_0>                      # cell 0's stmts after the loop (e.g. Write)
    Cond(P_1):
        <post_1>
    …

Semantics-preserving because (a) the loop nest is shared by every sibling
(same axis names + extents — verified per axis), (b) the common prefix is
side-effect-free SSA (Load + Assign), so computing it unconditionally costs
nothing more than what the unmasked cell (cell 0, ``Cond(1)``) was already
paying, and (c) every cell's residual + post-loop stmts get rewrapped in
the same ``Cond.cond`` they had before, so the masked guards still gate the
N-dep work. ``Accum`` targets carry their finalized value across the Loop
boundary in the original single-Cond form; after fusion the Accum stays
inside the sub-Cond at the same loop depth and its named accumulator's
identity-init scope is unchanged.

Run order: AFTER ``011_dedup_replicated`` (within-scope dedup runs first to
keep the structural comparison cheap — same-scope duplicates are already
folded) and BEFORE ``050_vectorize_loads`` (its consecutive-Load detection
depends on the post-fusion shape).
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.ir.stmt.blocks import Cond, Loop
from deplodock.compiler.ir.tile.ir import SerialTile, StridedTile, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped

# Both ``SerialTile`` / ``StridedTile`` (Tile IR) and ``Loop`` (Kernel IR
# generic loop) carry an ``axis`` + a ``body`` and render identically as a
# C-for. The materializer hasn't run yet at this pass's point, so the
# replicated bodies are still in Tile-IR shape — match on whichever wrapper
# the producer planted.
_LOOP_LIKE = (Loop, SerialTile, StridedTile)

PATTERN = [Pattern("root", TileOp)]


def rewrite(root: Node) -> Graph | None:
    new_body = _walk(root.op.body)
    if new_body == root.op.body:
        raise RuleSkipped("no fusable sibling Cond runs")
    return TileOp(body=new_body, name=root.op.name, knobs=dict(root.op.knobs))


def _walk(body: Body) -> Body:
    """Bottom-up: rewrite nested bodies first, then look for fusable sibling
    Cond runs at this scope."""
    rewritten = []
    for s in body:
        nested = s.nested()
        if nested:
            new_bodies = tuple(_walk(b) for b in nested)
            if new_bodies != nested:
                s = s.with_bodies(new_bodies)
        rewritten.append(s)
    return _try_fuse_runs(Body(tuple(rewritten)))


def _try_fuse_runs(body: Body) -> Body:
    """Collect maximal runs of sibling ``Cond``s whose bodies share a leading
    loop-nest skeleton, and fuse each run in place. Compatibility is the
    *leading* skeleton only — the Conds may legitimately diverge in tail
    stmts after the matched loops (per-cell ``Write``s, for instance), and
    those tails become per-cell post-loop ``Cond``s in the fused result."""
    out: list[Stmt] = []
    i = 0
    body_list = list(body)
    while i < len(body_list):
        s = body_list[i]
        if not isinstance(s, Cond):
            out.append(s)
            i += 1
            continue
        run_end = i + 1
        while (
            run_end < len(body_list) and isinstance(body_list[run_end], Cond) and _has_matching_lead_loop(s.body, body_list[run_end].body)
        ):
            run_end += 1
        if run_end - i >= 2:
            fused = _fuse_run(body_list[i:run_end])
            if fused is not None:
                out.extend(fused)
                i = run_end
                continue
        out.append(s)
        i += 1
    return Body(tuple(out))


def _has_matching_lead_loop(a: Body, b: Body) -> bool:
    """Two bodies are run-compatible iff their leading statement is a
    loop-like stmt over the same axis name + extent (``Dim`` equality, so a
    symbolic ``seq_len`` matches a symbolic ``seq_len``). The tail stmts are
    not required to match — they're handled per-cell in the fused output."""
    if not a or not b:
        return False
    la, lb = a[0], b[0]
    if not isinstance(la, _LOOP_LIKE) or not isinstance(lb, _LOOP_LIKE) or type(la) is not type(lb):
        return False
    if la.axis.name != lb.axis.name or la.axis.extent != lb.axis.extent:
        return False
    return True


def _fuse_run(run: list[Cond]) -> list[Stmt] | None:
    """Fuse a run of sibling Conds whose bodies start with matching Loops.

    Returns the replacement stmt list, or ``None`` if there is no common
    prefix worth hoisting. The result is ``[<fused loop nest>, <per-cell
    post-loop Cond>...]`` — the per-cell tail stmts (anything in each
    original Cond body after the matched Loop) become individual sub-Conds
    after the fused loop nest, so a per-cell ``Write`` that originally sat
    inside the masked guard still gets gated by the same predicate."""
    lead_loops = [c.body[0] for c in run]
    tail_per_cell = [list(c.body[1:]) for c in run]
    fused_loop = _fuse_loop_run(lead_loops, [c.cond for c in run])
    if fused_loop is None:
        return None
    out: list[Stmt] = [fused_loop]
    for cond, tail in zip(run, tail_per_cell, strict=True):
        if tail:
            out.append(Cond(cond=cond.cond, body=Body(tuple(tail)), else_body=cond.else_body))
    return out


def _fuse_loop_run(loops: list, preds: list):
    """Recursive: given N matched ``Loop``s over the same axis, drill into
    their bodies. If those bodies also share a leading Loop, recurse to fuse
    deeper. Otherwise extract the common prefix at the innermost level and
    wrap per-cell residuals in their original predicates.

    Returns the rebuilt ``Loop`` over the shared axis, or ``None`` if the
    drill bottoms out with no common prefix to hoist."""
    inner_bodies = [list(loop.body) for loop in loops]
    # If every inner body is a single matched Loop on the same axis, drill.
    if _all_single_matched_loop(inner_bodies):
        inner_fused = _fuse_loop_run([b[0] for b in inner_bodies], preds)
        if inner_fused is None:
            return None
        return loops[0].with_bodies((Body((inner_fused,)),))
    # Otherwise the inner bodies are statement lists with a (hopefully)
    # shared invariant prefix and per-cell residual. Extract.
    common_prefix_len = 0
    min_len = min(len(b) for b in inner_bodies)
    while common_prefix_len < min_len:
        head = inner_bodies[0][common_prefix_len]
        if not all(inner_bodies[k][common_prefix_len] == head for k in range(1, len(inner_bodies))):
            break
        common_prefix_len += 1
    if common_prefix_len == 0:
        return None
    common_prefix = list(inner_bodies[0][:common_prefix_len])
    per_cell_residuals = [list(b[common_prefix_len:]) for b in inner_bodies]
    inner_stmts: list[Stmt] = list(common_prefix)
    for pred, residual in zip(preds, per_cell_residuals, strict=True):
        if not residual:
            continue
        inner_stmts.append(Cond(cond=pred, body=Body(tuple(residual)), else_body=()))
    return loops[0].with_bodies((Body(tuple(inner_stmts)),))


def _all_single_matched_loop(bodies: list[list[Stmt]]) -> bool:
    """Lockstep drill predicate: every body is exactly one loop-like stmt of
    the same flavor (``Loop`` / ``SerialTile`` / ``StridedTile``) over the
    same axis name + extent (``Dim`` equality — symbolic extents included)."""
    if not all(len(b) == 1 and isinstance(b[0], _LOOP_LIKE) for b in bodies):
        return False
    first = bodies[0][0]
    return all(type(b[0]) is type(first) and b[0].axis.name == first.axis.name and b[0].axis.extent == first.axis.extent for b in bodies)
