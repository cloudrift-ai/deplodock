"""Merge contiguous sibling Stages at the same scope into a single
larger slab.

After per-Load staging in ``007_stage_inputs``, register-tiled matmul
kernels emit one Stage per cell offset (e.g. ``p_weight_smem`` for the
+0 offset and ``p_weight_smem_1`` for the +1 offset). Each individual
slab covers a stride-2 access pattern; their union covers a contiguous
DRAM range. This pass detects such groups and replaces them with a
single Stage carrying an extra "cell" cache axis.

Trigger: 2+ Stages at the same scope sharing
``(buf, cache_axes_shape, slab_dims, non-const-part-of-origin,
non-const-part-of-template)``, whose integer constants in origin /
template differ in **exactly one** source dim and form a contiguous
range starting from any base.

Rewrite: replace the first Stage with a merged Stage; drop the others.
The merged Stage adds a new cache axis ``<base>_cell:N`` mapping to the
target source dim. Origin shifts to absorb the minimum offset.
Template[target_dim] gets ``+ Var(cell) + min_delta``.

Each consumer Load that referenced an old slab gets one new index
slot at the end with the cell offset (a Literal) for the slab it used
to read. The materializer's existing ``source_index_template`` path
fetches all elements correctly because the cooperative load decodes
the cell axis as just another slab dim.

Idempotent: a merged Stage's name carries an ``_m`` suffix; if a
group's stages already have that suffix, skip.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from deplodock.compiler.ir.stmt import Body, Load, Loop, Stmt, Tile
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

PATTERN = [Pattern("root", TileOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: Body) -> Body | None:
    idx, tile = single_tile(body)
    new_tile_body = _process_scope_body(tile.body)
    if new_tile_body == tile.body:
        raise RuleSkipped("no contiguous sibling Stages share group key — nothing to merge")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process_scope_body(body: Body) -> Body:
    """Walk a body. For each free Loop scope, recurse. At each level,
    merge contiguous sibling Stages."""
    walked: list[Stmt] = []
    for s in body:
        if isinstance(s, Loop) and not s.is_reduce:
            walked.append(dc_replace(s, body=_process_scope_body(s.body)))
        else:
            walked.append(s)
    body = tuple(walked)

    stage_idx = [(i, s) for i, s in enumerate(body) if isinstance(s, Stage)]
    if len(stage_idx) < 2:
        return body

    groups: dict[tuple, list[tuple[int, Stage]]] = defaultdict(list)
    for i, s in stage_idx:
        if s.source_index_template is None:
            continue
        if s.name.endswith("_m"):
            return body  # already merged — idempotence
        key = _group_key(s)
        if key is None:
            continue
        groups[key].append((i, s))

    replacements: dict[int, Stage | None] = {}
    load_rewrites: dict[str, tuple[str, int]] = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        merged = _try_merge_group(group)
        if merged is None:
            continue
        merged_stage, offset_per_old_name = merged
        first_idx = group[0][0]
        replacements[first_idx] = merged_stage
        for i, _ in group[1:]:
            replacements[i] = None
        for _, s in group:
            load_rewrites[s.name] = (merged_stage.name, offset_per_old_name[s.name])

    if not replacements:
        return body

    new_body: list[Stmt] = []
    for i, s in enumerate(body):
        if i in replacements:
            ns = replacements[i]
            if ns is not None:
                new_body.append(ns)
            continue
        new_body.append(_rewrite_loads(s, load_rewrites))
    return tuple(new_body)


def _group_key(s: Stage) -> tuple | None:
    """Stages with the same key are merge candidates: same buf, cache-
    axis shape, slab_dims, and non-constant parts of origin / template."""
    cache_key = tuple((ax.name, int(ax.extent)) for ax in s.axes)
    origin_nc = tuple(_split_const(e)[0].pretty() for e in s.origin)
    template_nc = tuple(_split_const(e)[0].pretty() for e in s.source_index_template or ())
    return (s.buf, cache_key, s.slab_dims, origin_nc, template_nc)


def _try_merge_group(group: list[tuple[int, Stage]]) -> tuple[Stage, dict[str, int]] | None:
    """Merge a group of Stages whose constants differ in exactly one source
    dim and form a contiguous range. Returns (merged_stage, name → offset)."""
    base_stage = group[0][1]
    base_origin_const = tuple(_split_const(e)[1] for e in base_stage.origin)
    base_template_const = tuple(_split_const(e)[1] for e in base_stage.source_index_template)
    n_dims = len(base_origin_const)
    if n_dims != len(base_template_const):
        return None

    # Per-stage per-dim integer offset relative to base.
    offsets_by_name: dict[str, tuple[int, ...]] = {}
    for _, s in group:
        oc = tuple(_split_const(e)[1] for e in s.origin)
        tc = tuple(_split_const(e)[1] for e in s.source_index_template)
        if len(oc) != n_dims or len(tc) != n_dims:
            return None
        per_dim = tuple(o - b for o, b in zip(oc, base_origin_const, strict=True))
        per_dim_t = tuple(t - b for t, b in zip(tc, base_template_const, strict=True))
        if per_dim != per_dim_t:
            return None
        offsets_by_name[s.name] = per_dim

    # Find the single dim that varies across the group.
    varying = [d for d in range(n_dims) if len({off[d] for off in offsets_by_name.values()}) > 1]
    if len(varying) != 1:
        return None
    target = varying[0]
    deltas = sorted({off[target] for off in offsets_by_name.values()})
    n = len(deltas)
    if n < 2:
        return None
    # Contiguous integer range.
    if deltas != list(range(deltas[0], deltas[0] + n)):
        return None

    # Build merged Stage. New cell axis sits at the end of the cache axes.
    cell_axis = Axis(name=f"{base_stage.name}_cell", extent=n)
    new_axes = (*base_stage.axes, cell_axis)
    new_slab_dims = (*base_stage.slab_dims, target)

    new_origin = list(base_stage.origin)
    new_origin[target] = new_origin[target] + Literal(deltas[0], "int")
    new_template = list(base_stage.source_index_template)
    new_template[target] = new_template[target] + Var(cell_axis.name) + Literal(deltas[0], "int")

    merged_stage = Stage(
        name=f"{base_stage.name}_m",
        buf=base_stage.buf,
        origin=tuple(new_origin),
        axes=new_axes,
        slab_dims=new_slab_dims,
        source_index_template=tuple(new_template),
    )

    offset_per_name = {name: off[target] - deltas[0] for name, off in offsets_by_name.items()}
    return merged_stage, offset_per_name


def _rewrite_loads(s: Stmt, rewrites: dict[str, tuple[str, int]]) -> Stmt:
    """Recursively rewrite Loads that point at old (now-merged) slabs."""

    def fn(c: Stmt) -> Stmt:
        if isinstance(c, Load) and c.input in rewrites:
            new_name, cell_offset = rewrites[c.input]
            return Load(name=c.name, input=new_name, index=(*c.index, Literal(cell_offset, "int")))
        return c

    return Body((s,)).map(fn)[0]


def _split_const(e: Expr) -> tuple[Expr, int]:
    """Split additive expr into (non-const sum, integer constant). The
    non-const sum collapses to ``Literal(0)`` if every term is constant."""
    terms = _flatten_add(e)
    const = 0
    rest: list[Expr] = []
    for t in terms:
        if isinstance(t, Literal) and t.dtype == "int" and isinstance(t.value, int):
            const += t.value
        else:
            rest.append(t)
    if not rest:
        return Literal(0, "int"), const
    sum_rest = rest[0]
    for t in rest[1:]:
        sum_rest = sum_rest + t
    return sum_rest, const


def _flatten_add(e: Expr) -> list[Expr]:
    if isinstance(e, BinaryExpr) and e.op == "+":
        return _flatten_add(e.left) + _flatten_add(e.right)
    return [e]
