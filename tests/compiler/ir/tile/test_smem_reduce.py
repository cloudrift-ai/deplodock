"""Tests for the cooperative smem-reduce strategy pass.

The pass rewrites a single-Loop reduction inside a ``Tile`` so each CUDA
block owns one output slot and threads cooperate on the reduction axis
via ``__shared__`` + tree-halve.

These tests exercise the strategy at three levels:
- structural: post-rewrite IR shape via ``run_pass``;
- render: golden CUDA fragments for a known input;
- threshold: small reductions (K below ``COOP_THRESHOLD``) skip the
  rewrite and stay one-thread-per-row.
"""

from __future__ import annotations

from pathlib import Path

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Accum, Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.tile.ir import (
    Enclosure,
    Smem,
    StridedLoop,
    Sync,
    Tile,
    TileOp,
    TreeHalve,
)
from deplodock.compiler.ir.tile.lower import lower_naive
from deplodock.compiler.ir.tile.render import render_tileop
from deplodock.compiler.pipeline.engine import run_pass

_TILE_PASS_DIR = Path(__file__).resolve().parents[4] / "deplodock/compiler/pipeline/passes/lowering/tile"


def _reduction_graph(rows: int, cols: int) -> Graph:
    """Build a Graph with one LoopOp summing along the inner axis."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (rows, cols)), node_id="x")
    i = Axis("i", rows)
    k = Axis("k", cols)
    body = (
        Loop(
            axis=i,
            body=(
                Loop(
                    axis=k,
                    body=(
                        Load(name="x_v", input="x", index=(Var("i"), Var("k"))),
                        Accum(name="acc", value="x_v", op=ElementwiseImpl("add")),
                    ),
                ),
                Write(output="y", index=(Var("i"),), value="acc"),
            ),
        ),
    )
    g.add_node(op=LoopOp(body=body), inputs=["x"], output=Tensor("y", (rows,)), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def _lower_to_tile(graph: Graph) -> None:
    """Replace each LoopOp in the graph with its ``lower_naive`` TileOp."""
    for node in list(graph.nodes.values()):
        if isinstance(node.op, LoopOp):
            node.op = lower_naive(node.op, kernel_name=f"k_{node.id}_reduce")


def test_strategy_rewrites_above_threshold():
    """K=512 ≥ COOP_THRESHOLD → cooperative shape with Smem/Sync/TreeHalve/Cond."""
    g = _reduction_graph(rows=4, cols=512)
    _lower_to_tile(g)
    g = run_pass(g, _TILE_PASS_DIR)

    tile_op = g.nodes["y"].op
    assert isinstance(tile_op, TileOp)
    encl = next(s for s in tile_op.body if isinstance(s, Enclosure))
    assert len(encl.block_axes) == 1
    assert len(encl.thread_axes) == 1
    assert encl.thread_axes[0].name == "t"
    assert encl.thread_axes[0].extent == 256

    tile = encl.body[0]
    assert isinstance(tile, Tile)
    body_kinds = [type(s).__name__ for s in tile.body]
    assert body_kinds[0] == "Smem"
    assert "StridedLoop" in body_kinds
    assert "Sync" in body_kinds
    assert "TreeHalve" in body_kinds
    assert body_kinds[-1] == "Cond"


def test_strategy_skips_below_threshold():
    """K=64 < COOP_THRESHOLD → unchanged one-thread-per-row shape."""
    g = _reduction_graph(rows=4, cols=64)
    _lower_to_tile(g)
    g = run_pass(g, _TILE_PASS_DIR)

    tile_op = g.nodes["y"].op
    encl = next(s for s in tile_op.body if isinstance(s, Enclosure))
    assert encl.block_axes == ()
    tile = encl.body[0]
    assert isinstance(tile, Tile)
    # No Smem / Sync / TreeHalve nodes remain — original Loop + Write only.
    for s in tile.body:
        assert not isinstance(s, (Smem, Sync, TreeHalve, StridedLoop))


def test_render_cooperative_kernel_shape():
    """Render the rewritten kernel and assert the load-bearing fragments."""
    g = _reduction_graph(rows=4, cols=512)
    _lower_to_tile(g)
    g = run_pass(g, _TILE_PASS_DIR)
    tile_op = g.nodes["y"].op

    src = render_tileop(tile_op, shapes={"x": (4, 512), "y": (4,)})
    # smem allocation (accumulator may be renamed by upstream normalization)
    assert "__shared__ float " in src and "_smem[256];" in src
    # one CUDA block per output row
    assert "blockIdx.x" in src
    # strided per-thread reduction loop
    assert "for (int" in src and "+= 256" in src and "< 512" in src
    # tree-halve over the smem buffer
    assert "for (int s = 128; s > 0; s >>= 1)" in src
    # final write guarded by t == 0
    assert "if (t == 0)" in src
    # at least one explicit barrier (between partial-store and tree-halve)
    assert src.count("__syncthreads();") >= 2


def test_strategy_idempotent():
    """Running the pass twice must not double-rewrite."""
    g = _reduction_graph(rows=4, cols=512)
    _lower_to_tile(g)
    g = run_pass(g, _TILE_PASS_DIR)
    src_once = render_tileop(g.nodes["y"].op, shapes={"x": (4, 512), "y": (4,)})

    g2 = run_pass(g, _TILE_PASS_DIR)
    src_twice = render_tileop(g2.nodes["y"].op, shapes={"x": (4, 512), "y": (4,)})

    assert src_once == src_twice
