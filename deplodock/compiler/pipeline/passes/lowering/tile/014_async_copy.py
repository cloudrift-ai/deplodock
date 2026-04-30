"""Mark Stages for ``cp.async`` (synchronous-style) on sm_80+.

For each ``Stage`` in the Tile body, set ``async_load = True`` when:

- The target GPU's compute capability is sm_80 or higher (Ampere+).
- The slab footprint is large enough that the commit/wait overhead
  amortizes (currently ``≥ 4 elements per thread per Stage``).

The materializer's ``_emit_stage`` then replaces the per-thread
``Load(reg) + Write(smem)`` pair with ``CpAsyncCopy`` (a single PTX
instruction that DMAs DRAM → smem without a register temp), trailing
``CpAsyncCommit`` + ``CpAsyncWait(0)``. The surrounding ``Sync``
remains — ``cp.async.wait_group`` only synchronizes per-thread, not
across the CTA.

This is the synchronous form of ``cp.async``. The pipelined / overlap
form (``wait_group(N)`` staggering across iterations of a
double-buffered K-outer loop) is a separate follow-up that will reuse
the same ``async_load`` flag plus a ``CpAsyncWait`` reordering pass.

Idempotence: a Stage with ``async_load`` already set is left alone.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD
from deplodock.compiler.ir.stmt import Loop, Stmt, Tile
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped

logger = logging.getLogger(__name__)

PATTERN = [Pattern("root", TileOp)]

_MIN_CAPABILITY = (8, 0)
_MIN_ELEMENTS_PER_THREAD = 4


@functools.cache
def _compute_capability() -> tuple[int, int]:
    """Query the active CUDA device's compute capability via cupy. Cached.
    Returns ``(0, 0)`` if cupy is unavailable — treated as no async support."""
    try:
        import cupy as cp

        dev = cp.cuda.Device()
        # cupy returns the capability as a string ``"MMm"``: ``"86"`` for
        # sm_86, ``"120"`` for sm_12.0. Minor is always the last digit.
        cap = str(dev.compute_capability)
        return (int(cap[:-1]), int(cap[-1]))
    except Exception as e:  # pragma: no cover
        logger.debug("cp.async gate: compute_capability query failed (%s)", e)
        return (0, 0)


def _supports_cp_async() -> bool:
    return _compute_capability() >= _MIN_CAPABILITY


def rewrite(graph: Graph, root: Node) -> Graph | None:
    if not _supports_cp_async():
        raise RuleSkipped(f"cp.async requires compute capability >= {_MIN_CAPABILITY}, got {_compute_capability()}")
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: tuple[Stmt, ...]) -> tuple[Stmt, ...] | None:
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        raise RuleSkipped(f"need exactly one Tile in TileOp.body, found {len(tiles)}")
    idx, tile = tiles[0]

    n_threads = 1
    for ba in tile.axes:
        if ba.bind == BIND_THREAD:
            n_threads *= int(ba.axis.extent)

    new_tile_body = _process(tile.body, n_threads)
    if new_tile_body is tile.body or new_tile_body == tile.body:
        raise RuleSkipped(f"no Stage eligible for cp.async (need >= {_MIN_ELEMENTS_PER_THREAD} elts/thread)")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process(body: tuple[Stmt, ...], n_threads: int) -> tuple[Stmt, ...]:
    """Walk a body. Mark eligible Stages async; recurse into free Loops
    so Stages inside (e.g. the K-outer chunk loop) get processed."""
    new_body: list[Stmt] = list(body)
    changed = False
    for i, s in enumerate(body):
        if isinstance(s, Stage) and not s.async_load:
            if _eligible(s, n_threads):
                new_body[i] = dc_replace(s, async_load=True)
                changed = True
        elif isinstance(s, Loop):
            inner = _process(s.body, n_threads)
            if inner is not s.body and inner != s.body:
                new_body[i] = dc_replace(s, body=inner)
                changed = True
    return tuple(new_body) if changed else body


def _eligible(stage: Stage, n_threads: int) -> bool:
    slab_floats = 1
    for ax in stage.axes:
        slab_floats *= int(ax.extent)
    elems_per_thread = max(1, slab_floats // max(1, n_threads))
    return elems_per_thread >= _MIN_ELEMENTS_PER_THREAD
