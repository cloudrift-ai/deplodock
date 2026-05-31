"""Unit tests for ``120_assemble_fma_clusters`` (plans/inline-fma-cluster.md M2+M3).

The pass recognizes the flat per-thread matmul outer-product cell that
``split_register_axes`` + ``materialize_tile`` leave in the inner K-loop and
wraps it in a single ``FmaCluster`` (rendered, in M3, as one inline-PTX
``asm volatile`` FFMA block). These tests pin the detector — it fires on the
clean f32 cell, extracts the right ``fm``/``fn`` and row-major ``acc``
ordering, and skips anything that isn't a clean A×B outer product over f32
operand buffers (masked cells, non-matmul bodies, fp16 operands). The
end-to-end test confirms the cluster fires and emits inline PTX.
"""

from __future__ import annotations

import importlib

import pytest

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.kernel.ir import FmaCluster, KernelOp, Smem
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load
from deplodock.compiler.ir.stmt.blocks import Cond
from deplodock.compiler.ir.tile.ir import SerialTile
from deplodock.compiler.pipeline import RuleSkipped

_mod = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.kernel.120_assemble_fma_clusters")

MUL = ElementwiseImpl("multiply")
ADD = ElementwiseImpl("add")
# The operand buffers used by the synthetic cells, declared f32.
F32_BUFS = frozenset({"a_smem", "b_smem"})


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
    cluster = _mod._match_outer_product(Body(cell), F32_BUFS)
    assert isinstance(cluster, FmaCluster)
    assert (cluster.fm, cluster.fn) == (3, 2)
    assert set(cluster.a_names) == {"a0", "a1", "a2"}
    assert set(cluster.b_names) == {"b0", "b1"}
    # acc_names are row-major acc[m*fn+n]: (m,n) = (0,0),(0,1),(1,0),(1,1),(2,0),(2,1)
    assert cluster.acc_names == ("acc_0_0", "acc_0_1", "acc_1_0", "acc_1_1", "acc_2_0", "acc_2_1")
    # the carried body is the original cell verbatim (round-trip payload)
    assert cluster.body == cell


def _matmul_kernel_op() -> KernelOp:
    # f32 Smem decls make a_smem / b_smem provably-f32 operand buffers (the
    # render is f32-only, so the pass gates on this).
    body = Body(
        (
            Smem(name="a_smem", extents=(64,), dtype="float"),
            Smem(name="b_smem", extents=(64,), dtype="float"),
            _kloop(_outer_product_cell(fm=2, fn=2)),
        )
    )
    return KernelOp(body=body, name="k_matmul", knobs={})


def test_rewrite_wraps_kloop_body_in_cluster(monkeypatch):
    monkeypatch.setenv("DEPLODOCK_FMA_CLUSTER", "1")  # opt in — off by default (M4)
    rewritten = _mod.rewrite(_FakeNode(_matmul_kernel_op()))
    kloop = rewritten.body[-1]
    assert isinstance(kloop, SerialTile)
    (cluster,) = kloop.body
    assert isinstance(cluster, FmaCluster)
    assert (cluster.fm, cluster.fn) == (2, 2)


def test_off_by_default(monkeypatch):
    """The knob is off by default (M4: no measured gain on sm_120) — a matmul
    cell that *would* cluster is skipped unless DEPLODOCK_FMA_CLUSTER=1."""
    monkeypatch.delenv("DEPLODOCK_FMA_CLUSTER", raising=False)
    with pytest.raises(RuleSkipped):
        _mod.rewrite(_FakeNode(_matmul_kernel_op()))


def test_masked_cell_not_matched():
    """A per-cell boundary ``Cond`` (masked-tile overhang) aborts the match —
    no cluster, no regression."""
    cell = _outer_product_cell(fm=2, fn=2)
    masked = cell + (Cond(cond=Literal(1, "int"), body=Body(())),)
    assert _mod._match_outer_product(Body(masked), F32_BUFS) is None


def test_non_matmul_body_not_matched():
    body = Body((Load(names=("x",), input="buf", index=(Var("i"),)), Assign(name="y", op=ADD, args=("x", "x"))))
    assert _mod._match_outer_product(body, F32_BUFS) is None


def test_single_cell_not_matched():
    """A 1×1 cell isn't a cluster (no operand reuse to exploit)."""
    assert _mod._match_outer_product(Body(_outer_product_cell(fm=1, fn=1)), F32_BUFS) is None


def test_no_cell_raises_rule_skipped(monkeypatch):
    monkeypatch.setenv("DEPLODOCK_FMA_CLUSTER", "1")  # enabled, so we exercise the no-cell path
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


def test_end_to_end_emits_inline_ptx(monkeypatch):
    """On a 512³ fp32 matmul (a shape whose tiling yields the clean cell), opting
    in with ``DEPLODOCK_FMA_CLUSTER=1`` makes the cluster fire and render a single
    inline-PTX ``asm volatile`` block of ``fma.rn.f32`` per cell, while the
    default (off) keeps the plain-C ``Load + Accum`` body (no inline PTX, no
    cluster node). Pinned to sm_80 so the tiling (hence the cell shape) is
    deterministic regardless of the live device."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.ir.cuda import CudaOp
    from deplodock.compiler.pipeline import KERNEL_PASSES, Pipeline

    monkeypatch.setenv("DEPLODOCK_COMPUTE_CAPABILITY", "8.0")

    # Default (off): no cluster.
    monkeypatch.delenv("DEPLODOCK_FMA_CLUSTER", raising=False)
    kern_off = Pipeline.build(KERNEL_PASSES).run(_matmul_graph(512))
    assert _count_clusters(kern_off) == 0, "off by default — no cluster"

    # Opt in: the cluster fires.
    monkeypatch.setenv("DEPLODOCK_FMA_CLUSTER", "1")
    kern_on = Pipeline.build(KERNEL_PASSES).run(_matmul_graph(512))
    assert _count_clusters(kern_on) > 0, "DEPLODOCK_FMA_CLUSTER=1 must assemble FmaCluster(s)"

    def _src(enabled: bool) -> str:
        if enabled:
            monkeypatch.setenv("DEPLODOCK_FMA_CLUSTER", "1")
        else:
            monkeypatch.delenv("DEPLODOCK_FMA_CLUSTER", raising=False)
        compiled = CudaBackend().compile(_matmul_graph(512))
        ops = [n.op for n in compiled.nodes.values() if isinstance(n.op, CudaOp)]
        return "\n".join(op.kernel_source for op in ops)

    src_on = _src(enabled=True)
    src_off = _src(enabled=False)
    assert "asm volatile(" in src_on and "fma.rn.f32" in src_on, "cluster must emit an inline-PTX FFMA block"
    assert "fma.rn.f32" not in src_off, "default keeps plain-C body — no inline PTX"
    assert src_on.count("fma.rn.f32") >= _count_clusters(kern_on), "one FFMA group per cluster, at least"
