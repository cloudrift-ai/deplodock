"""``assemble`` — the one deterministic step (block-DAG ``TileGraph`` → tower).

``plans/tile-ir-block-dag.md`` makes staging / pipelining / warp-spec / register
tiling / split-K / placement all the same kind of operation: a :class:`Schedule`
annotation over an invariant algorithm. ``assemble`` applies the Schedule to the
algorithm and emits today's ``TileOp`` tower (the migration oracle is
byte-identical CUDA — the downstream kernel/cuda passes stay untouched during
coexistence).

This module ports the regimes incrementally. **Covered so far: pointwise** — one
block, free axes already σ-split by the ``tile_axis`` body move, bound
GRID/THREAD/REGISTER. ``assemble`` reconstructs the layer order the legacy
``materialize._assemble`` produced (``REGISTER`` cells innermost, then THREAD,
then GRID, extra-outer GRID axes last) and wraps via the shared
:func:`_wrap_tower`, so the output is the same ``TileOp``.
"""

from __future__ import annotations

from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.tile.blockdag import Binding, Block, TileGraph
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.passes.lowering.tile.partition._tower import Role, _wrap_tower

# Schedule ``Binding`` → tower ``Role``. SERIAL has no free-axis use yet (the K
# re-bracket emits its own SERIAL_OUTER / STAGE_INNER layers); mapped to plain
# serial for completeness.
_ROLE_OF: dict[Binding, Role] = {
    Binding.GRID: Role.BLOCK,
    Binding.THREAD: Role.THREAD,
    Binding.REGISTER: Role.REGISTER,
    Binding.WARP: Role.WARP,
    Binding.ATOM: Role.ATOM,
}


def _pointwise_layers(block: Block, sched) -> list[tuple]:
    """The innermost-first ``(axis, Role)`` layers for a pointwise block, in the
    exact order ``materialize._assemble`` emitted them: all REGISTER cells, then
    all THREAD axes, then all GRID axes — each tier in ``block.domain`` order (so
    the inner ``N`` axis precedes the outer ``M`` axis, and extra-outer GRID axes
    trail last)."""
    binding = sched.binding

    def tier(b: Binding) -> list[tuple]:
        return [(a, _ROLE_OF[b]) for a in block.domain if binding.get(a.name) is b]

    return [*tier(Binding.REGISTER), *tier(Binding.THREAD), *tier(Binding.GRID)]


def assemble_pointwise(
    graph: TileGraph,
    *,
    knobs: dict,
    base_knobs: dict,
    kernel_name: str,
    leading: tuple = (),
) -> TileOp:
    """Assemble a single-block pointwise ``TileGraph`` into its ``TileOp`` tower.

    The block's ``compute`` is the σ-rewritten inner body (the ``tile_axis`` move
    already ran); ``assemble`` only reconstructs the binding tower. ``knobs`` /
    ``base_knobs`` / ``kernel_name`` are the deployed-variant stamp the downstream
    passes + perf DB key on (not part of the pure algorithm)."""
    if len(graph.blocks) != 1:
        raise NotImplementedError("assemble_pointwise: expected exactly one block")
    block = graph.blocks[0]
    layers = _pointwise_layers(block, graph.schedule)
    chain_body = _wrap_tower(layers, tuple(block.compute))

    knobs_full = {**base_knobs, **knobs}
    inner_defs = {n for s in Body.coerce(chain_body).iter() for n in s.defines()}
    kept_leading = tuple(s for s in leading if not (set(s.defines()) & inner_defs))
    return TileOp(body=kept_leading + chain_body, name=kernel_name, knobs=knobs_full)
