"""Tests for the cooperative-reduce strategy + materialization pipeline.

The strategy (``lowering/tile/002_cooperative_reduce``) flips bindings
on a Tile-IR ``Tile``; the materialization pass
(``lowering/kernel/001_materialize_block``) then produces a Kernel-IR
``KernelOp`` with ``Enclosure`` / ``Smem`` / ``Sync`` / ``TreeHalve`` /
``StridedLoop``. These tests run both passes and assert on the
resulting Kernel-IR shape.
"""

from __future__ import annotations

from pathlib import Path

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.kernel.ir import (
    Enclosure,
    KernelOp,
    Smem,
    StridedLoop,
    Sync,
    TreeHalve,
)
from deplodock.compiler.ir.kernel.render import render_kernelop
from deplodock.compiler.ir.loop import Accum, Assign, Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.pipeline.engine import run_pass
from tests.compiler.ir.tile._helpers import lower_naive

_TILE_PASS_DIR = Path(__file__).resolve().parents[4] / "deplodock/compiler/pipeline/passes/lowering/tile"
_KERNEL_PASS_DIR = Path(__file__).resolve().parents[4] / "deplodock/compiler/pipeline/passes/lowering/kernel"


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


def _lower_and_run(graph: Graph) -> Graph:
    """lower_naive each LoopOp, then run lowering/tile + lowering/kernel."""
    for node in list(graph.nodes.values()):
        if isinstance(node.op, LoopOp):
            node.op = lower_naive(node.op, kernel_name=f"k_{node.id}_reduce")
    graph = run_pass(graph, _TILE_PASS_DIR)
    graph = run_pass(graph, _KERNEL_PASS_DIR)
    return graph


def test_strategy_rewrites_above_threshold():
    """K=512 ≥ COOP_THRESHOLD → cooperative KernelOp with Smem/Sync/TreeHalve/Cond."""
    g = _lower_and_run(_reduction_graph(rows=4, cols=512))

    kernel_op = g.nodes["y"].op
    assert isinstance(kernel_op, KernelOp)
    encl = next(s for s in kernel_op.body if isinstance(s, Enclosure))
    assert len(encl.block_axes) == 1
    assert len(encl.thread_axes) == 1
    assert encl.thread_axes[0].name == "t"
    assert encl.thread_axes[0].extent == 256

    body_kinds = [type(s).__name__ for s in encl.body]
    assert "Smem" in body_kinds
    assert "StridedLoop" in body_kinds
    assert "Sync" in body_kinds
    assert "TreeHalve" in body_kinds
    assert body_kinds[-1] == "Cond"


def test_strategy_skips_below_threshold():
    """K=64 < COOP_THRESHOLD → one-thread-per-output-row shape, no smem."""
    g = _lower_and_run(_reduction_graph(rows=4, cols=64))

    kernel_op = g.nodes["y"].op
    encl = next(s for s in kernel_op.body if isinstance(s, Enclosure))
    assert encl.block_axes == ()
    for s in encl.body:
        assert not isinstance(s, (Smem, Sync, TreeHalve, StridedLoop))


def test_render_cooperative_kernel_shape():
    """Render the materialized KernelOp and assert the load-bearing fragments."""
    g = _lower_and_run(_reduction_graph(rows=4, cols=512))
    kernel_op = g.nodes["y"].op

    src = render_kernelop(kernel_op, shapes={"x": (4, 512), "y": (4,)})
    assert "__shared__ float " in src and "_smem[256];" in src
    assert "blockIdx.x" in src
    assert "for (int" in src and "+= 256" in src and "< 512" in src
    assert "for (int s = 128; s > 0; s >>= 1)" in src
    assert "if (t == 0)" in src
    assert src.count("__syncthreads();") >= 2


def test_pipeline_idempotent():
    """Re-running the passes over an already-materialized graph is a no-op."""
    g = _lower_and_run(_reduction_graph(rows=4, cols=512))
    src_once = render_kernelop(g.nodes["y"].op, shapes={"x": (4, 512), "y": (4,)})

    g = run_pass(g, _TILE_PASS_DIR)
    g = run_pass(g, _KERNEL_PASS_DIR)
    src_twice = render_kernelop(g.nodes["y"].op, shapes={"x": (4, 512), "y": (4,)})

    assert src_once == src_twice


def _two_phase_loopop(rows: int, cols: int) -> LoopOp:
    """Hand-built softmax-shape LoopOp: max → sum(exp(x - max)) → write."""
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
    """Two reductions → two smem buffers, two tree-halves, broadcast Load between."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 512)), node_id="x")
    g.add_node(op=_two_phase_loopop(4, 512), inputs=["x"], output=Tensor("y", (4,)), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g = _lower_and_run(g)

    kernel_op = g.nodes["y"].op
    encl = next(s for s in kernel_op.body if isinstance(s, Enclosure))
    smem_decls = [s for s in encl.body if isinstance(s, Smem)]
    halves = [s for s in encl.body if isinstance(s, TreeHalve)]
    assert len(smem_decls) == 2
    assert len(halves) == 2
    assert halves[0].op.name == "maximum"
    assert halves[1].op.name == "add"

    src = render_kernelop(kernel_op, shapes={"x": (4, 512), "y": (4,)})
    assert src.count("__shared__ float ") == 2
    assert "_b = " in src and "_smem[0]" in src
    assert src.count("for (int s = 128; s > 0; s >>= 1)") == 2
