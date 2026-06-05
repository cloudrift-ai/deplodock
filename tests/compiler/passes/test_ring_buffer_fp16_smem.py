"""Regression: ``040_use_ring_buffers`` must size the fp16 smem budget at the
real 2 bytes/elem, not the fp32-assuming ``BYTES_PER_ELEM=4`` fallback.

At the tile stage ``Source.dtype`` is unstamped (``030_stamp_types`` is a
downstream kernel pass), so ``Source.smem_bytes`` falls back to
``BYTES_PER_ELEM=4`` and **2×-over-counts** every fp16 slab. That wrongly
pruned ``BUFFER_COUNT`` 3-4 for fp16 matmul tiles whose deep ring actually
fits — leaving the kernel stuck at depth 2 and ~14 % off the achievable
latency (deeper pipelines hide the TMA load latency: `long_scoreboard`
≈ 4 → 0.18 at depth 3-4, measured 2048² fp16 RTX 5090). The fix reads the
true dtype off the ``TileOp`` input tensors so the budget check matches the
materializer's real allocation.

The test pins a 64×256 fp16 warp tile and a smem cap chosen so the depth-4
ring fits at the real 2 B/elem (4 × 20 KB = 80 KB) but NOT at the
over-counting 4 B/elem (4 × 40 KB = 160 KB). With the fix ``BUFFER_COUNT=4``
fires; without it the pass would reject and fall back to a shallower ring.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.context import Context
from deplodock.compiler.dtype import F16
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Assign
from deplodock.compiler.ir.tile.ir import StageBundle
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline

# sm_120's real per-block dynamic-smem cap. Picked as the test cap so the
# 64×256 fp16 tile's depth-4 ring (80 KB real) fits but the 2×-over-counted
# estimate (160 KB) does not — the exact window where the bug bit.
_SM120_SMEM_CAP = 101376


def _mma_matmul_graph(*, M: int, N: int, K: int) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N), dtype=F16), node_id="b")
    i, j, k = Axis("i", M), Axis("j", N), Axis("k", K)
    g.add_node(
        op=LoopOp(
            body=(
                Loop(
                    axis=i,
                    body=(
                        Loop(
                            axis=j,
                            body=(
                                Loop(
                                    axis=k,
                                    body=(
                                        Load(name="a_v", input="a", index=(Var("i"), Var("k"))),
                                        Load(name="b_v", input="b", index=(Var("k"), Var("j"))),
                                        Assign(name="p", op=ElementwiseImpl("multiply"), args=("a_v", "b_v")),
                                        Accum(name="acc", value="p"),
                                    ),
                                ),
                                Write(output="c", index=(Var("i"), Var("j")), value="acc"),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        inputs=["a", "b"],
        output=Tensor("c", (M, N), dtype=F16),
        node_id="c",
    )
    g.inputs = ["a", "b"]
    g.outputs = ["c"]
    return g


def _max_buffer_count(graph: Graph) -> int:
    best = 1
    for node in graph.nodes.values():
        body = getattr(node.op, "body", None)
        if body is None:
            continue
        for stmt in body.iter():
            if isinstance(stmt, StageBundle):
                best = max(best, stmt.buffer_count)
    return best


def _pin(monkeypatch, **knobs: int | str) -> None:
    for k, v in knobs.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", str(v))


def test_fp16_ring_buffer_uses_real_dtype_bytes(monkeypatch):
    """A 64×256 fp16 warp tile pinned to ``BUFFER_COUNT=4`` admits the depth-4
    ring under the sm_120 cap — the depth-4 slab is 80 KB at the real 2 B/elem,
    which fits; the pre-fix fp32 over-count reported 160 KB and rejected it."""
    _pin(monkeypatch, MMA="mma_m16n8k16_f16", WM=1, WN=4, FM=4, FN=8, BK=2, BUFFER_COUNT=4)
    g = _mma_matmul_graph(M=2048, N=2048, K=2048)
    ctx = Context(compute_capability=(9, 0), max_dynamic_smem=_SM120_SMEM_CAP)
    out = Pipeline.build(TILE_PASSES).run(g, ctx=ctx)
    kop = out.nodes["c"].op
    assert kop.knobs.get("MMA") == "mma_m16n8k16_f16"
    got = kop.knobs.get("BUFFER_COUNT")
    assert got == 4, f"BUFFER_COUNT=4 should fire at the real fp16 byte count, got {got}"
    assert _max_buffer_count(out) == 4, "expected a depth-4 ring StageBundle in the lowered body"


def test_fp16_ring_buffer_rejects_when_real_bytes_overflow(monkeypatch):
    """Sanity floor: a cap below even the real depth-4 footprint (80 KB) must
    still reject depth 4 — the fix corrects the byte count, it doesn't disable
    the smem budget guard. At a 64 KB cap the depth-4 80 KB ring genuinely
    overflows, so ``BUFFER_COUNT=4`` cannot fire."""
    _pin(monkeypatch, MMA="mma_m16n8k16_f16", WM=1, WN=4, FM=4, FN=8, BK=2, BUFFER_COUNT=4)
    g = _mma_matmul_graph(M=2048, N=2048, K=2048)
    ctx = Context(compute_capability=(9, 0), max_dynamic_smem=64 * 1024)
    # Pinned-but-unfittable BUFFER_COUNT surfaces as a hard error (the pass
    # raises rather than silently dropping to SYNC — see 040's `_fail`).
    with pytest.raises(Exception):  # noqa: B017,PT011 — ValueError from 040 (or a wrapped LoweringError)
        Pipeline.build(TILE_PASSES).run(g, ctx=ctx)
