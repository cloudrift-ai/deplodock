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
from deplodock.compiler.ir.loop import Accum, Assign, Axis, Load, Loop, LoopOp, Write
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
    assert "Smem" in body_kinds
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


def _two_phase_loopop(rows: int, cols: int) -> LoopOp:
    """Hand-built LoopOp with the softmax-style shape: max-reduce → sum-reduce → write.
    The second reduction reads the first reduction's accumulator scalar."""
    i = Axis("i", rows)
    k1 = Axis("k1", cols)
    k2 = Axis("k2", cols)
    body = (
        Loop(
            axis=i,
            body=(
                Loop(
                    axis=k1,
                    body=(
                        Load(name="x_v", input="x", index=(Var("i"), Var("k1"))),
                        Accum(name="acc_max", value="x_v", op=ElementwiseImpl("maximum")),
                    ),
                ),
                Loop(
                    axis=k2,
                    body=(
                        Load(name="x_v2", input="x", index=(Var("i"), Var("k2"))),
                        Assign(name="diff", op=ElementwiseImpl("subtract"), args=("x_v2", "acc_max")),
                        Assign(name="ediff", op=ElementwiseImpl("exp"), args=("diff",)),
                        Accum(name="acc_sum", value="ediff", op=ElementwiseImpl("add")),
                    ),
                ),
                Write(output="y", index=(Var("i"),), value="acc_sum"),
            ),
        ),
    )
    return LoopOp(body=body)


def test_strategy_handles_two_phase_softmax_shape():
    """Two reduction phases under one Tile → two smem buffers, two halves,
    second phase references first via a broadcast Load (renamed acc_max → acc_max_b)."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 512)), node_id="x")
    g.add_node(op=_two_phase_loopop(4, 512), inputs=["x"], output=Tensor("y", (4,)), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    _lower_to_tile(g)
    g = run_pass(g, _TILE_PASS_DIR)

    tile_op = g.nodes["y"].op
    encl = next(s for s in tile_op.body if isinstance(s, Enclosure))
    tile = encl.body[0]
    smem_decls = [s for s in tile.body if isinstance(s, Smem)]
    halves = [s for s in tile.body if isinstance(s, TreeHalve)]
    assert len(smem_decls) == 2
    assert len(halves) == 2
    # Second halve uses the sum op; first uses maximum.
    assert halves[0].op.name == "maximum"
    assert halves[1].op.name == "add"

    src = render_tileop(tile_op, shapes={"x": (4, 512), "y": (4,)})
    # Two distinct smem buffers (names may be normalized to acc0_smem / acc1_smem).
    assert src.count("__shared__ float ") == 2
    # Broadcast load of the first phase's result into phase 2 (the *_b name).
    assert "_b = " in src and "_smem[0]" in src
    # Two tree-halves rendered.
    assert src.count("for (int s = 128; s > 0; s >>= 1)") == 2
