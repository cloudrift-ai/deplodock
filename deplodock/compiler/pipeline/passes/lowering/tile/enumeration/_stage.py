"""The ``stage`` move's offer set — ranked stageable read-sites of a kernel.

A pure function over a built :class:`TileGraph`'s DERIVED projections
(``Block.reads`` + ``Schedule.binding`` + the K-tower in ``Block.compute``),
never tower shape — the ``plans/tile-ir-block-dag.md`` discipline. A gmem input
read is **stageable** iff:

- it indexes its buffer AFFINELY (a ``TEMPLATE`` collapsed-reshape declines — the
  slab can't be additively sized);
- it has **reuse**: at least one ``GRID`` / ``THREAD`` parallel axis is *absent*
  from the read's free axes (fan-in — many threads share the cached tile); and
- the block has a **K-tower** to stage through (a reduce regime — a per-stage smem
  slab only pays off inside the serial K loop; a pointwise nest has no stage).

Returns the ranked stageable ``Edge``s (most-bytes-first, buffer name as a stable
tiebreak). ``050_stage`` masks them into ``Schedule.staged`` and ``assembly/_slab``
materializes each into one smem slab + cooperative producer. R1 covers the scalar
(no-MMA) reduce regimes (matmul / fused-prologue); warp/atom + symbolic-K staging
return with later tiers.
"""

from __future__ import annotations

from deplodock.compiler.ir.tile.ir import AddrKind, Binding, Block, Edge, SerialTile, TileGraph


def _has_k_tower(block: Block) -> bool:
    """True iff ``compute`` has a K serial loop to stage through — a ``serial_outer``
    (the multi-stage K loop, reloaded per stage) or, when ``BK == K`` collapses it,
    the ``stage_inner`` loop alone (the single-stage whole-K slab). Pointwise nests
    have neither."""
    return any(isinstance(s, SerialTile) and s.kind in ("serial_outer", "stage_inner") for s in block.compute.iter())


def _stage_k_extent(block: Block) -> int:
    """Per-stage K extent (the slab's K span): the product of the stage-inner
    serial K axis and any reduce ``RegisterTile`` (the ``FK`` strip) extents. Used
    only for byte-ranking the candidates."""
    from deplodock.compiler.ir.tile.ir import RegisterTile  # noqa: PLC0415

    ext = 1
    for s in block.compute.iter():
        if isinstance(s, SerialTile) and s.kind == "stage_inner" and s.axis.extent.is_static:
            ext *= s.axis.extent.as_static()
        elif isinstance(s, RegisterTile) and s.reduce:
            for ax in s.axes:
                if ax.extent.is_static:
                    ext *= ax.extent.as_static()
    return ext


def stage_candidates(graph: TileGraph) -> list[Edge]:
    """The ranked stageable input read-sites of ``graph``'s single block."""
    block = graph.blocks[0]
    sched = graph.schedule
    if not _has_k_tower(block):
        return []
    parallel = {a.name for a in block.domain if sched.binding.get(a.name) in (Binding.GRID, Binding.THREAD)}
    written = {p.buffer for b in graph.blocks for p in b.writes}
    k_ext = _stage_k_extent(block)

    ranked: list[tuple[int, str, Edge]] = []
    seen: set[str] = set()
    for p in block.reads:
        buf = p.buffer
        if buf in written or buf in seen:  # only input reads; one slab per buffer
            continue
        acc = p.access
        if acc.kind is not AddrKind.AFFINE:
            continue  # TEMPLATE collapsed reshape — not slab-sizable (R1)
        free = acc.free_axes()
        if not (parallel - free):
            continue  # every parallel axis present → no fan-in reuse, skip
        # Slab bytes: the non-GRID (THREAD/REGISTER) tile span × the K stage span ×
        # element width. GRID axes are CTA-uniform (they fold into the slab origin),
        # so they don't enlarge the slab.
        tile_elems = 1
        for a in block.domain:
            if a.name in free and sched.binding.get(a.name) is not Binding.GRID and a.extent.is_static:
                tile_elems *= a.extent.as_static()
        nbytes = graph.buffers[buf].dtype.nbytes if buf in graph.buffers and graph.buffers[buf].dtype else 4
        seen.add(buf)
        ranked.append((tile_elems * k_ext * nbytes, buf, Edge(src=buf, dst=block.name, buffer=buf)))

    ranked.sort(key=lambda t: (-t[0], t[1]))
    return [e for _, _, e in ranked]
