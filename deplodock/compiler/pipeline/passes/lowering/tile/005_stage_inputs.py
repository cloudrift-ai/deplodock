"""Bare-minimum operand staging.

Per-buffer Stage placement, derived directly from where each Load sits
in the post-blockify IR. No reuse heuristics, no depth iteration, no
canonical-origin search — just walk the Loads and emit one Stage per
buffer at the scope just inside its Loads' common loop prefix.

Algorithm
---------

For each input buffer ``B`` with one or more Loads in the Tile body:

1. **Common prefix**: longest list of enclosing loops shared by every
   Load of ``B`` (compared by axis identity).
2. **Below axes**: for each Load, the loops past the common prefix
   that vary the Load's address (i.e. their axis appears in
   ``load.index``). Plus thread axes whose names appear in the
   index — those vary across threads in the cooperative load.
3. **Cache axes**: the union, ordered by appearance in
   ``ref_load.index``. Each cache axis attaches to the slab dim of
   the index expression it appears in.
4. **Origin**: ``ref_load.index`` with every cache-axis Var zeroed
   and simplified — block-uniform anchor.
5. **Place**: insert the Stage at the start of ``common[-1].body``
   (or Tile body head when ``common`` is empty), then redirect every
   in-scope Load of ``B`` to the staged buffer with cache-relative
   indices.

Skip conditions
---------------

- Tile has no BLOCK axes (no smem cooperation possible).
- Tile already contains a Stage (some upstream pattern rule staged it).
- All Loads of ``B`` are block-uniform (no thread/inner-loop variance
  in the index) — nothing to cache cooperatively.
- Multiple cache axes would land on the same slab dim of the source
  buffer (sub-tile pattern that the bare-min path doesn't decompose).
- Smem footprint exceeds ``STAGE_BYTES_LIMIT``.
- Loads disagree on the block-uniform part of the index (different
  origins per Load).
"""

from __future__ import annotations

from collections.abc import Iterable

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, SimplifyCtx, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Cond, Load, Loop, Stmt, StridedLoop
from deplodock.compiler.ir.tile.ir import Stage, Tile, TileOp
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", TileOp)]

STAGE_BYTES_LIMIT = 16 * 1024
DTYPE_BYTES = 4


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_stage(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_stage(body):
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        return None
    idx, tile = tiles[0]
    if not tile.block_axes:
        return None
    if _any_stage(tile.body):
        return None

    new_tile_body = tile.body
    thread_axes = [ba.axis for ba in tile.axes if ba.bind == BIND_THREAD]

    # Collect Loads grouped by buf, in source order.
    loads_per_buf: dict[str, list[tuple[Load, tuple]]] = {}
    for load, path in _walk_loads(new_tile_body):
        loads_per_buf.setdefault(load.input, []).append((load, path))

    changed = False
    for buf, entries in loads_per_buf.items():
        plan = _plan(buf, entries, thread_axes)
        if plan is None:
            continue
        new_tile_body = _apply(new_tile_body, plan)
        changed = True

    if not changed:
        return None
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def _plan(buf: str, entries: list[tuple[Load, tuple]], thread_axes: list[Axis]):
    paths = [path for _, path in entries]
    common = _common_loop_prefix(paths)

    ref_load = entries[0][0]

    # Below-axes: every loop axis past the common prefix, across all loads,
    # whose name appears in the load's index.
    below_axes_by_name: dict[str, Axis] = {}
    for load, path in entries:
        free = _free_vars(load.index)
        for loop in path[len(common) :]:
            if loop.axis.name in free:
                below_axes_by_name[loop.axis.name] = loop.axis

    thread_by_name = {a.name: a for a in thread_axes}

    # Cache axes from ref_load.index, in dim order.
    cache: list[tuple[Axis, int]] = []
    seen: set[str] = set()
    per_dim: dict[int, int] = {}
    ref_free = _free_vars(ref_load.index)
    for dim, e in enumerate(ref_load.index):
        for name in e.free_vars():
            if name in seen or name not in ref_free:
                continue
            ax = thread_by_name.get(name) or below_axes_by_name.get(name)
            if ax is None:
                continue
            cache.append((ax, dim))
            seen.add(name)
            per_dim[dim] = per_dim.get(dim, 0) + 1

    if not cache:
        return None
    if any(c > 1 for c in per_dim.values()):
        return None  # sub-tile pattern; bare-min skips

    footprint = 1
    for ax, _ in cache:
        footprint *= int(ax.extent)
    if footprint * DTYPE_BYTES > STAGE_BYTES_LIMIT:
        return None

    cache_names = {ax.name for ax, _ in cache}
    origin = tuple(_zero_and_simplify(e, cache_names) for e in ref_load.index)

    # Multi-load consistency: every Load must reduce to the same origin.
    for load, _ in entries:
        if tuple(_zero_and_simplify(e, cache_names).pretty() for e in load.index) != tuple(o.pretty() for o in origin):
            return None

    # Reshape views (``/``, ``%`` in the index) need the full index
    # template — the additive ``origin + decoded`` path doesn't compose
    # with the layout transform. Materialize substitutes cache-axis Vars
    # into the template per cache position.
    non_affine = any(_has_div_or_mod(e) for e in ref_load.index)
    template = tuple(ref_load.index) if non_affine else None

    stage = Stage(
        name=f"{buf}_stage",
        buf=buf,
        origin=origin,
        axes=tuple(ax for ax, _ in cache),
        slab_dims=tuple(d for _, d in cache),
        source_index_template=template,
    )
    return (buf, stage, common)


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def _apply(body: tuple, plan) -> tuple:
    buf, stage, common = plan
    return _insert_at(body, common, 0, buf, stage)


def _insert_at(body: tuple, common: tuple, idx: int, buf: str, stage: Stage) -> tuple:
    if idx == len(common):
        rewritten = tuple(_rewrite_loads(s, buf, stage) for s in body)
        return (stage,) + rewritten
    target_name = common[idx].axis.name
    out = []
    for s in body:
        if isinstance(s, (Loop, StridedLoop)) and s.axis.name == target_name:
            out.append(_clone_loop(s, _insert_at(s.body, common, idx + 1, buf, stage)))
        else:
            out.append(s)
    return tuple(out)


def _rewrite_loads(stmt: Stmt, buf: str, stage: Stage) -> Stmt:
    if isinstance(stmt, Load) and stmt.input == buf:
        smem_index = tuple(Var(ax.name) for ax in stage.axes)
        return Load(name=stmt.name, input=stage.name, index=smem_index)
    if isinstance(stmt, Loop):
        return Loop(axis=stmt.axis, body=tuple(_rewrite_loads(c, buf, stage) for c in stmt.body))
    if isinstance(stmt, StridedLoop):
        return StridedLoop(
            axis=stmt.axis,
            start=stmt.start,
            step=stmt.step,
            body=tuple(_rewrite_loads(c, buf, stage) for c in stmt.body),
        )
    if isinstance(stmt, Cond):
        return Cond(
            cond=stmt.cond,
            body=tuple(_rewrite_loads(c, buf, stage) for c in stmt.body),
            else_body=tuple(_rewrite_loads(c, buf, stage) for c in stmt.else_body),
        )
    return stmt


def _clone_loop(loop, body: tuple):
    if isinstance(loop, StridedLoop):
        return StridedLoop(axis=loop.axis, start=loop.start, step=loop.step, body=body)
    return Loop(axis=loop.axis, body=body)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _walk_loads(stmts: tuple, path: tuple = ()) -> Iterable[tuple[Load, tuple]]:
    for s in stmts:
        if isinstance(s, Load):
            yield (s, path)
        elif isinstance(s, (Loop, StridedLoop)):
            yield from _walk_loads(s.body, path + (s,))
        elif isinstance(s, Cond):
            yield from _walk_loads(s.body, path)
            yield from _walk_loads(s.else_body, path)


def _common_loop_prefix(paths: list[tuple]) -> tuple:
    if not paths:
        return ()
    common: list = []
    for level in zip(*paths, strict=False):
        first = level[0]
        if all(loop is first for loop in level):
            common.append(first)
        else:
            break
    return tuple(common)


def _any_stage(stmts: tuple) -> bool:
    for s in stmts:
        if isinstance(s, Stage):
            return True
        if isinstance(s, (Loop, StridedLoop, Cond)):
            if _any_stage(getattr(s, "body", ())):
                return True
            if _any_stage(getattr(s, "else_body", ())):
                return True
    return False


def _free_vars(index: tuple) -> set[str]:
    out: set[str] = set()
    for e in index:
        out |= e.free_vars()
    return out


def _zero_and_simplify(e: Expr, names: set[str]) -> Expr:
    sigma = Sigma({n: Literal(0, "int") for n in names})
    return sigma.apply(e).simplify(SimplifyCtx.empty())


def _has_div_or_mod(e: Expr) -> bool:
    if isinstance(e, BinaryExpr):
        if e.op in ("/", "%"):
            return True
        return _has_div_or_mod(e.left) or _has_div_or_mod(e.right)
    return False
