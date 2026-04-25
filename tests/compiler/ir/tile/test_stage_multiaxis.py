"""Tests for multi-axis ``Stage`` materialization in
``lowering/kernel/001_materialize_tile.py``.

Single-axis Stage (the softmax path) keeps the existing direct-stride
emission. Multi-axis Stage flattens the cache axes into one synthetic
linear index and decodes per-thread back into per-axis coords for the
source Load and smem Write — needed for matmul-style A/B operand caching.
"""

from __future__ import annotations

from pathlib import Path

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_BLOCK_STRIDED, Axis, BoundAxis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel.ir import KernelOp, Smem, StridedLoop, Sync
from deplodock.compiler.ir.kernel.render import render_kernelop
from deplodock.compiler.ir.stmt import Accum, Assign, Load, Write
from deplodock.compiler.ir.tile.ir import BoundLoop, Stage, Tile, TileOp
from deplodock.compiler.pipeline.engine import run_pass

_KERNEL_PASS_DIR = Path(__file__).resolve().parents[4] / "deplodock/compiler/pipeline/passes/lowering/kernel"


def _matmul_tile_with_2d_stages() -> TileOp:
    """Hand-constructed cooperative tile whose body opens with two 2D
    Stages (A: BM·BK, B: BK·BN). Exercises the multi-axis stage emit
    path. Realistic matmul placement (Stages inside the K_o loop) needs
    a separate generalization to ``_materialize_cooperative`` — out of
    scope for this slice."""
    BM, BN, BK = 16, 16, 16
    m_o = Axis("m_o", 8)
    n_o = Axis("n_o", 8)
    k_i = Axis("k_i", BK)
    m_i = Axis("m_i", BM)
    n_i = Axis("n_i", BN)

    a_idx = (Var("m_o") * Literal(BM, "int") + Var("m_i"), Var("k_i"))
    b_idx = (Var("k_i"), Var("n_o") * Literal(BN, "int") + Var("n_i"))
    c_idx = (Var("m_o") * Literal(BM, "int") + Var("m_i"), Var("n_o") * Literal(BN, "int") + Var("n_i"))

    k_loop = BoundLoop(
        axis=BoundAxis(axis=k_i, bind=BIND_BLOCK_STRIDED),
        body=(
            Load(name="a", input="A", index=a_idx),
            Load(name="b", input="B", index=b_idx),
            Assign(name="t", op=ElementwiseImpl("multiply"), args=("a", "b")),
            Accum(name="acc", value="t", op=ElementwiseImpl("add")),
        ),
    )

    tile = Tile(
        axes=(
            BoundAxis(axis=m_o, bind=BIND_BLOCK),
            BoundAxis(axis=n_o, bind=BIND_BLOCK),
        ),
        body=(
            Stage(buf="A", index=a_idx, axes=(m_i, k_i)),
            Stage(buf="B", index=b_idx, axes=(k_i, n_i)),
            k_loop,
            Write(output="C", index=c_idx, value="acc"),
        ),
    )
    return TileOp(body=(tile,), name="k_matmul")


def test_multiaxis_stage_materializes_to_smem_with_flat_decode():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("A", (128, 64)), node_id="A")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("B", (64, 128)), node_id="B")
    g.add_node(op=_matmul_tile_with_2d_stages(), inputs=["A", "B"], output=Tensor("C", (128, 128)), node_id="C")
    g.inputs = ["A", "B"]
    g.outputs = ["C"]

    g = run_pass(g, _KERNEL_PASS_DIR)
    kernel_op = g.nodes["C"].op
    assert isinstance(kernel_op, KernelOp)

    smems = [s for s in kernel_op if isinstance(s, Smem)]
    # A_stage and B_stage, each declared with 2D extents (BM, BK) / (BK, BN).
    smem_by_name = {s.name: s for s in smems}
    assert "A_stage" in smem_by_name and smem_by_name["A_stage"].extents == (16, 16)
    assert "B_stage" in smem_by_name and smem_by_name["B_stage"].extents == (16, 16)

    # The cooperative load is a flat StridedLoop driven by the synthetic axis.
    loops = [s for s in kernel_op if isinstance(s, StridedLoop)]
    flat_loops = [s for s in loops if s.axis.name.startswith("_stage_")]
    assert len(flat_loops) == 2
    # Flat extent = BM·BK = 256.
    assert all(s.axis.extent == 256 for s in flat_loops)
    assert all(s.step == 256 for s in flat_loops)

    # Each cooperative load's body has Load(global) + Write(smem); the smem
    # Write index has two entries (the decoded m_i, k_i / k_i, n_i coords).
    for sl in flat_loops:
        load = sl.body[0]
        write = sl.body[1]
        assert isinstance(load, Load) and load.input in {"A", "B"}
        assert isinstance(write, Write) and write.output.endswith("_stage")
        assert len(write.index) == 2

    # Sync after each cooperative load.
    syncs = [s for s in kernel_op if isinstance(s, Sync)]
    assert len(syncs) >= 2


def test_multiaxis_stage_renders_without_errors():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("A", (128, 64)), node_id="A")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("B", (64, 128)), node_id="B")
    g.add_node(op=_matmul_tile_with_2d_stages(), inputs=["A", "B"], output=Tensor("C", (128, 128)), node_id="C")
    g.inputs = ["A", "B"]
    g.outputs = ["C"]

    g = run_pass(g, _KERNEL_PASS_DIR)
    src = render_kernelop(g.nodes["C"].op, shapes={"A": (128, 64), "B": (64, 128), "C": (128, 128)})
    # Sanity: rendered source contains both smem decls and the flat decode arithmetic.
    assert "__shared__" in src
    assert "A_stage" in src and "B_stage" in src
    # Flat-decode % / / appearing somewhere in the staging section.
    assert "% 16" in src and "/ 16" in src


def test_flat_decode_sigma_shape():
    """The decode helper produces ``flat % BK`` for innermost and ``flat / BK``
    for outermost in a 2D cache."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_mod",
        Path(__file__).resolve().parents[4] / "deplodock/compiler/pipeline/passes/lowering/kernel/001_materialize_tile.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    BM, BK = 16, 16
    sigma = mod._flat_decode_sigma((Axis("m_i", BM), Axis("k_i", BK)), "F")
    # m_i: F / 16 (no mod, since outermost). k_i: F % 16.
    m_i_expr = sigma.apply(Var("m_i"))
    assert isinstance(m_i_expr, BinaryExpr) and m_i_expr.op == "/"
    assert isinstance(m_i_expr.left, Var) and m_i_expr.left.name == "F"
    assert isinstance(m_i_expr.right, Literal) and m_i_expr.right.value == BK

    k_i_expr = sigma.apply(Var("k_i"))
    assert isinstance(k_i_expr, BinaryExpr) and k_i_expr.op == "%"
    assert isinstance(k_i_expr.left, Var) and k_i_expr.left.name == "F"
    assert isinstance(k_i_expr.right, Literal) and k_i_expr.right.value == BK
