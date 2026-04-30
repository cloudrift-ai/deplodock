"""Top up the per-CTA thread count to ``_TARGET_THREADS`` by absorbing
inner slices of ``BIND_BLOCK`` axes into ``BIND_THREAD``.

After ``008_register_tile`` splits each of the two PAT-extent THREAD
axes by F (each becomes ``PAT/F``), matmul kernels drop to ``(PAT/F)²``
threads / CTA — typically 64 (PAT=16, F=2). This pass brings the count
back up to ``_TARGET_THREADS`` (256) by carving an inner slice off a
BLOCK axis and binding it to THREAD instead.

By the time this pass runs, ``007_stage_inputs`` has already produced
``Stage`` stmts whose ``origin`` is anchored on BLOCK axes. If the
carved BLOCK axis appears in any Stage's origin, the blanket
``orig → outer*slice + inner`` substitution would push a per-thread
``inner`` Var into ``origin`` — violating the "origin is CTA-uniform"
invariant. We handle these Stages surgically:

- substitute ``orig → outer*slice`` in ``origin`` (drop the per-thread
  part);
- append ``inner`` to ``Stage.axes`` and the affected source dim to
  ``Stage.slab_dims`` — the slab grows by ``slice_size`` along that
  dim so different threads of the new axis cache different rows;
- consumer Loads of the staged buffer get ``Var(inner)`` appended to
  their cache-coord index (parallel to the new cache axis).

The other stmts get the usual blanket Sigma rewrite. The chosen axis
is the largest BLOCK axis whose extent has a divisor in ``[2, gap]``
where ``gap = _TARGET_THREADS / current_threads``.

Idempotence: skip when ``current_threads >= _TARGET_THREADS`` or no
suitable BLOCK axis exists.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Load, Stmt, Tile
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

PATTERN = [Pattern("root", TileOp)]

_TARGET_THREADS = 256
# Conservative bound on staged-smem floats post-augmentation. The carved
# BLOCK axis becomes a new cache axis on every Stage that anchored on
# it, multiplying those Stages' slabs by ``slice_size``. If the total
# would exceed the consumer 48 KB smem limit (× double-buffer head-
# room), skip rebalance — running on the un-rebalanced thread count
# (typically 64) is preferable to a compile failure.
_SMEM_FLOAT_BUDGET = 48 * 1024 // 4 // 2  # 6144 floats — leaves 50% for double-buffer + pad


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: tuple) -> tuple | None:
    idx, tile = single_tile(body)

    threads_used = 1
    for ba in tile.axes:
        if ba.bind == BIND_THREAD:
            threads_used *= int(ba.axis.extent)
    if threads_used >= _TARGET_THREADS:
        raise RuleSkipped(f"already at thread budget: threads_used={threads_used} >= target={_TARGET_THREADS}")

    gap = _TARGET_THREADS // threads_used
    chosen: tuple[int, BoundAxis, int] | None = None
    for i, ba in enumerate(tile.axes):
        if ba.bind != BIND_BLOCK:
            continue
        ext = int(ba.axis.extent)
        slice_size = max((d for d in range(2, gap + 1) if ext % d == 0), default=1)
        if slice_size <= 1:
            continue
        if chosen is None or ext > int(chosen[1].axis.extent):
            chosen = (i, ba, slice_size)
    if chosen is None:
        raise RuleSkipped(f"no BLOCK axis with a divisor in [2,{gap}] to slice for more threads")

    axis_idx, ba, slice_size = chosen
    orig = ba.axis
    ext = int(orig.extent)
    outer = Axis(name=f"{orig.name}_o", extent=ext // slice_size)
    inner = Axis(name=f"{orig.name}_i", extent=slice_size)

    new_axes = (
        *tile.axes[:axis_idx],
        BoundAxis(axis=outer, bind=BIND_BLOCK),
        *tile.axes[axis_idx + 1 :],
        BoundAxis(axis=inner, bind=BIND_THREAD),
    )

    # Identify Stages whose origin references the carved axis. These get
    # cache-axis-augmented surgically; their consumer Loads need the new
    # cache-coord var appended to their index.
    stages_to_augment: dict[str, int] = {}
    total_floats = 0
    for s in tile.body.iter():
        if isinstance(s, Stage):
            n = 1
            for ax in s.axes:
                n *= int(ax.extent)
            dims = [d for d, e in enumerate(s.origin) if orig.name in e.free_vars()]
            if not dims:
                total_floats += n
                continue
            if len(dims) > 1:
                raise RuleSkipped(f"Stage {s.name} references {orig.name} in multiple origin dims — unsupported")
            stages_to_augment[s.name] = dims[0]
            total_floats += n * slice_size

    if total_floats > _SMEM_FLOAT_BUDGET:
        raise RuleSkipped(
            f"post-rebalance Stage smem ≈ {total_floats} floats > budget {_SMEM_FLOAT_BUDGET}; "
            f"rebalance would cause smem overflow on {len(stages_to_augment)} augmented Stage(s)"
        )

    outer_only = Sigma({orig.name: Var(outer.name) * Literal(slice_size, "int")})
    inner_var = Var(inner.name)

    full_sigma = Sigma({orig.name: Var(outer.name) * Literal(slice_size, "int") + inner_var})

    def pre(s: Stmt) -> Stmt:
        if isinstance(s, Stage) and s.name in stages_to_augment:
            d = stages_to_augment[s.name]
            new_origin = tuple(outer_only.apply(e) for e in s.origin)
            # Build the full source-index template. With two cache axes
            # at source dim d (the existing one + the new inner), the
            # additive ``origin[d] + Var(cache_axis)`` form can't carry
            # the strides correctly: ``a_orig`` had a coefficient C in
            # origin[d], so ``inner`` must contribute ``inner * C`` to
            # source dim d (since ``a_orig = outer*slice + inner``).
            # Applying ``full_sigma`` to the original (implicit or
            # explicit) template captures this naturally — the resulting
            # template references ``Var(inner)`` with the right
            # multiplier baked in.
            if s.source_index_template is not None:
                base_template = s.source_index_template
            else:
                # Build the implicit template: origin[d] + Var(cache_axis_at_d) for each dim.
                ax_by_dim: dict[int, Axis] = {}
                for ax, dim in zip(s.axes, s.slab_dims, strict=True):
                    ax_by_dim.setdefault(dim, ax)
                base_template = tuple(
                    s.origin[dim_idx] + Var(ax_by_dim[dim_idx].name) if dim_idx in ax_by_dim else s.origin[dim_idx]
                    for dim_idx in range(len(s.origin))
                )
            new_template = tuple(full_sigma.apply(e) for e in base_template)
            new_cache_axes = (*s.axes, inner)
            new_slab_dims = (*s.slab_dims, d)
            return Stage(
                name=s.name,
                buf=s.buf,
                origin=new_origin,
                axes=new_cache_axes,
                slab_dims=new_slab_dims,
                source_index_template=new_template,
                pad=s.pad,
                buffer_count=s.buffer_count,
                phase=s.phase,
                async_load=s.async_load,
                pipelined=s.pipelined,
            )
        if isinstance(s, Load) and s.input in stages_to_augment:
            return Load(name=s.name, input=s.input, index=(*s.index, inner_var))
        return s

    augmented_body = tile.body.map(pre)

    # Blanket Sigma everywhere else. Augmented Stage origins / Load
    # indices no longer reference ``orig.name`` so Sigma is a no-op on
    # them; non-augmented stmts that referenced ``orig.name`` get
    # ``orig → outer*slice + inner`` substituted as usual.
    sigma = Sigma({orig.name: Var(outer.name) * Literal(slice_size, "int") + inner_var})
    new_body_inner = tuple(s.rewrite(_id, sigma) for s in augmented_body)
    new_tile = Tile(axes=new_axes, body=new_body_inner)
    return body[:idx] + (new_tile,) + body[idx + 1 :]


def _id(name: str) -> str:
    return name
