"""Launch-geometry pass.

Decides the THREAD/BLOCK partition of every parallel axis and lifts
body free Loops over output dimensions into ``Tile.axes``. Replaces the
legacy ``003_block_matmul`` rule with a single launch-geometry decision
informed by the actual per-block thread budget.

Two transformations, applied in order:

1. **Lift body free Loops over output dims.** A free Loop whose body
   contains a ``Write`` indexed by the loop's axis is iterating a
   distinct output position per iteration — independent work,
   parallelisable. Lift the axis into ``Tile.axes`` (initially THREAD)
   and replace the loop with its body. Fixes the fused-SDPA case where
   the head-dim free loop (``a7:64``) sat in the body and every thread
   re-ran the inner reduce 64 times serially.

2. **Apportion THREAD/BLOCK.** Every axis in ``Tile.axes`` starts as
   THREAD (from ``001_lower_loopop``). Walk innermost-to-outermost
   filling THREAD up to ``_THREAD_BUDGET``. An axis that exceeds the
   remaining budget gets split into inner-THREAD + outer-BLOCK. Axes
   past the budget go straight to BLOCK.

Idempotent: if every body Loop is already lifted and every Tile axis
already fits the THREAD/BLOCK partition, returns None.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Cond, Loop, Stmt, StridedLoop, Tile, Write
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.engine import Pattern

from deplodock.compiler.tuning import per_axis_threads, thread_budget

PATTERN = [Pattern("root", TileOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body):
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        return None
    idx, tile = tiles[0]
    if tile.block_axes:
        return None  # already partitioned

    lifted = _lift_output_loops(tile)
    partitioned = _partition_threads(lifted)
    if partitioned is None:
        return None
    return body[:idx] + (partitioned,) + body[idx + 1 :]


# ---------------------------------------------------------------------------
# 1. Lift body free Loops over output dims
# ---------------------------------------------------------------------------


def _lift_output_loops(tile: Tile) -> Tile:
    """Find body free Loops that wrap a ``Write`` whose index varies
    with the loop's axis; lift them into Tile.axes.

    Only top-level body stmts are considered — nested promotions would
    mix loop ordering decisions with reduction structure. SDPA's a7
    loop sits at top level (after the two softmax reduces) so this
    catches the case we care about."""
    new_axes = list(tile.axes)
    new_body: list[Stmt] = []
    changed = False
    for s in tile.body:
        if isinstance(s, Loop) and not s.is_reduce and _writes_with_axis(s.body, s.axis.name):
            new_axes.append(BoundAxis(axis=s.axis, bind=BIND_THREAD))
            new_body.extend(s.body)
            changed = True
        else:
            new_body.append(s)
    if not changed:
        return tile
    return Tile(axes=tuple(new_axes), body=tuple(new_body))


def _writes_with_axis(stmts: tuple, axis_name: str) -> bool:
    for s in stmts:
        if isinstance(s, Write):
            free = set()
            for e in s.index:
                free |= e.free_vars()
            if axis_name in free:
                return True
        if isinstance(s, (Loop, StridedLoop)) and _writes_with_axis(s.body, axis_name):
            return True
        if isinstance(s, Cond):
            if _writes_with_axis(s.body, axis_name) or _writes_with_axis(s.else_body, axis_name):
                return True
    return False


# ---------------------------------------------------------------------------
# 2. Apportion THREAD / BLOCK
# ---------------------------------------------------------------------------


def _partition_threads(tile: Tile) -> Tile | None:
    """Walk axes innermost→outermost. Each axis with extent ≥
    ``_PER_AXIS_THREADS`` gets a ``_PER_AXIS_THREADS``-thread inner
    slice + remainder BLOCK; smaller axes go THREAD whole. Stop adding
    THREAD slices once the running product reaches ``_THREAD_BUDGET``;
    remaining outer axes go BLOCK whole.

    The per-axis 16-thread tile matches the legacy ``003`` semantics: a
    matmul (M, N both ≥16) gets a 16×16 thread tile, leaving each
    operand load with its own thread axis for cooperative reuse.
    Single-parallel-axis kernels (RMSNorm row-reduction) get a single
    16-thread axis — same as before."""
    pat = per_axis_threads(tile)
    tb = thread_budget()
    axes = list(tile.axes)
    new_axes_inner_first: list[BoundAxis] = []
    sigma_map: dict[str, object] = {}
    threads_used = 1

    for ba in reversed(axes):
        ext = int(ba.axis.extent)
        if threads_used >= tb:
            new_axes_inner_first.append(BoundAxis(axis=ba.axis, bind=BIND_BLOCK))
            continue
        if ext < pat:
            # Small axis — keep whole as THREAD if it'd fit, else BLOCK.
            if threads_used * ext <= tb:
                new_axes_inner_first.append(BoundAxis(axis=ba.axis, bind=BIND_THREAD))
                threads_used *= ext
            else:
                new_axes_inner_first.append(BoundAxis(axis=ba.axis, bind=BIND_BLOCK))
            continue
        if ext == pat:
            if threads_used * ext <= tb:
                new_axes_inner_first.append(BoundAxis(axis=ba.axis, bind=BIND_THREAD))
                threads_used *= ext
            else:
                new_axes_inner_first.append(BoundAxis(axis=ba.axis, bind=BIND_BLOCK))
            continue
        # Larger than per-axis tile → split.
        if ext % pat != 0:
            # Non-divisible — keep whole; if it'd overflow, BLOCK; else THREAD.
            if threads_used * ext <= tb:
                new_axes_inner_first.append(BoundAxis(axis=ba.axis, bind=BIND_THREAD))
                threads_used *= ext
            else:
                new_axes_inner_first.append(BoundAxis(axis=ba.axis, bind=BIND_BLOCK))
            continue
        inner = Axis(f"{ba.axis.name}_i", pat)
        outer = Axis(f"{ba.axis.name}_o", ext // pat)
        new_axes_inner_first.append(BoundAxis(axis=inner, bind=BIND_THREAD))
        new_axes_inner_first.append(BoundAxis(axis=outer, bind=BIND_BLOCK))
        sigma_map[ba.axis.name] = Var(outer.name) * Literal(pat, "int") + Var(inner.name)
        threads_used *= pat

    new_axes_inner_first.reverse()
    new_axes = new_axes_inner_first

    # No-op short-circuit: identical axes, no split.
    if not sigma_map and len(new_axes) == len(axes):
        same = all(a.axis is b.axis and a.bind == b.bind for a, b in zip(new_axes, axes, strict=True))
        if same:
            return None

    sigma = Sigma(sigma_map) if sigma_map else Sigma.IDENTITY
    new_body = tuple(s.rewrite(_id, sigma) for s in tile.body) if sigma_map else tile.body
    return Tile(axes=tuple(new_axes), body=new_body)


def _id(name: str) -> str:
    return name
