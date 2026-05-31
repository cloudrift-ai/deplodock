"""Unit tests for ``120_assemble_fma_clusters`` (plans/inline-fma-cluster.md M2).

The pass recognizes the flat per-thread matmul outer-product cell that
``split_register_axes`` + ``materialize_tile`` leave in the inner K-loop and
wraps it in a single ``FmaCluster``. In M2 this is a behavior-neutral
round-trip (the cluster re-emits its body verbatim); these tests pin the
detector — it fires on the clean cell, extracts the right ``fm``/``fn`` and
row-major ``acc`` ordering, and skips anything that isn't a clean A×B outer
product (masked cells, non-matmul bodies). The byte-identical-CUDA proof lives
in the end-to-end behavior check.
"""

from __future__ import annotations

import importlib

import pytest

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.kernel.ir import FmaCluster, KernelOp
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load
from deplodock.compiler.ir.stmt.blocks import Cond
from deplodock.compiler.ir.tile.ir import SerialTile
from deplodock.compiler.pipeline import RuleSkipped

_mod = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.kernel.120_assemble_fma_clusters")

MUL = ElementwiseImpl("multiply")
ADD = ElementwiseImpl("add")


class _FakeNode:
    """Minimal Node shim — rewrite() only reads ``.op``."""

    def __init__(self, op):
        self.op = op


def _kloop(cell: tuple) -> SerialTile:
    """Wrap a cell in a serial K-reduce loop (what the pass walks into)."""
    return SerialTile(axis=Axis(name="k", extent=Dim(64)), body=Body(cell))


def _outer_product_cell(fm: int, fn: int) -> tuple:
    """Clean FM×FN cell: FM A-loads, FN B-loads, FM*FN products + accums.

    Products are emitted in FN-major order (``multiply(b_n, a_m)``) — the
    detector must still recover the A/B split by source buffer and order the
    accumulators row-major ``acc[m*fn+n]`` regardless of emission order."""
    stmts: list = []
    for m in range(fm):
        stmts.append(Load(names=(f"a{m}",), input="a_smem", index=(Var("k"), Var(f"r{m}"))))
    for n in range(fn):
        stmts.append(Load(names=(f"b{n}",), input="b_smem", index=(Var("k"), Var(f"c{n}"))))
    for n in range(fn):
        for m in range(fm):
            stmts.append(Assign(name=f"v_{m}_{n}", op=MUL, args=(f"b{n}", f"a{m}")))
    for n in range(fn):
        for m in range(fm):
            stmts.append(Accum(name=f"acc_{m}_{n}", value=f"v_{m}_{n}", op=ADD))
    return tuple(stmts)


def test_detects_clean_outer_product():
    cell = _outer_product_cell(fm=3, fn=2)
    cluster = _mod._match_outer_product(Body(cell))
    assert isinstance(cluster, FmaCluster)
    assert (cluster.fm, cluster.fn) == (3, 2)
    assert set(cluster.a_names) == {"a0", "a1", "a2"}
    assert set(cluster.b_names) == {"b0", "b1"}
    # acc_names are row-major acc[m*fn+n]: (m,n) = (0,0),(0,1),(1,0),(1,1),(2,0),(2,1)
    assert cluster.acc_names == ("acc_0_0", "acc_0_1", "acc_1_0", "acc_1_1", "acc_2_0", "acc_2_1")
    # the carried body is the original cell verbatim (round-trip payload)
    assert cluster.body == cell


def test_rewrite_wraps_kloop_body_in_cluster():
    op = KernelOp(body=Body((_kloop(_outer_product_cell(fm=2, fn=2)),)), name="k_matmul", knobs={})
    rewritten = _mod.rewrite(_FakeNode(op))
    (kloop,) = rewritten.body
    assert isinstance(kloop, SerialTile)
    (cluster,) = kloop.body
    assert isinstance(cluster, FmaCluster)
    assert (cluster.fm, cluster.fn) == (2, 2)


def test_masked_cell_not_matched():
    """A per-cell boundary ``Cond`` (masked-tile overhang) aborts the match —
    no cluster, no regression."""
    cell = _outer_product_cell(fm=2, fn=2)
    masked = cell + (Cond(cond=Literal(1, "int"), body=Body(())),)
    assert _mod._match_outer_product(Body(masked)) is None


def test_non_matmul_body_not_matched():
    body = Body((Load(names=("x",), input="buf", index=(Var("i"),)), Assign(name="y", op=ADD, args=("x", "x"))))
    assert _mod._match_outer_product(body) is None


def test_single_cell_not_matched():
    """A 1×1 cell isn't a cluster (no operand reuse to exploit)."""
    assert _mod._match_outer_product(Body(_outer_product_cell(fm=1, fn=1))) is None


def test_no_cell_raises_rule_skipped():
    op = KernelOp(body=Body((Load(names=("x",), input="buf", index=(Var("i"),)),)), name="k_plain", knobs={})
    with pytest.raises(RuleSkipped):
        _mod.rewrite(_FakeNode(op))


# --- End-to-end behavior-neutrality -----------------------------------------


def _matmul_graph(n: int):
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (n, n)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (n, n)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (n, n)), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _count_clusters(graph) -> int:
    return sum(len(node.op.body.iter_of_type(FmaCluster)) for node in graph.nodes.values() if isinstance(node.op, KernelOp))


def test_end_to_end_behavior_neutral(monkeypatch):
    """On a 512³ fp32 matmul (a shape whose tiling yields the clean cell), the
    cluster fires under the default ``FMA_CLUSTER=1`` yet renders byte-identical
    CUDA to ``FMA_CLUSTER=0`` — the M2 assembly pass is a behavior-neutral
    round-trip. Pinned to sm_80 so the tiling (hence the cell shape) is
    deterministic regardless of the live device."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.ir.cuda import CudaOp
    from deplodock.compiler.pipeline import KERNEL_PASSES, Pipeline

    monkeypatch.setenv("DEPLODOCK_COMPUTE_CAPABILITY", "8.0")

    # Non-vacuity: the cluster actually fires at the kernel stage under default.
    kern_on = Pipeline.build(KERNEL_PASSES).run(_matmul_graph(512))
    assert _count_clusters(kern_on) > 0, "expected the matmul cell to assemble into FmaCluster(s)"

    monkeypatch.setenv("DEPLODOCK_FMA_CLUSTER", "0")
    kern_off = Pipeline.build(KERNEL_PASSES).run(_matmul_graph(512))
    assert _count_clusters(kern_off) == 0, "FMA_CLUSTER=0 must skip the assembly pass"
    monkeypatch.delenv("DEPLODOCK_FMA_CLUSTER")

    def _src(knob_off: bool) -> str:
        if knob_off:
            monkeypatch.setenv("DEPLODOCK_FMA_CLUSTER", "0")
        else:
            monkeypatch.delenv("DEPLODOCK_FMA_CLUSTER", raising=False)
        compiled = CudaBackend().compile(_matmul_graph(512))
        ops = [n.op for n in compiled.nodes.values() if isinstance(n.op, CudaOp)]
        return "\n".join(op.kernel_source for op in ops)

    assert _src(knob_off=False) == _src(knob_off=True), "FmaCluster round-trip must render identical CUDA"
