"""Tests for matmul-style materialization in
``lowering/kernel/001_materialize_tile.py``.

Discrimination: a cooperative ``Tile`` whose ``BIND_BLOCK_STRIDED`` output
axes' extents·product equals ``BLOCK_SIZE`` is matmul-style — each thread
owns exactly one output slot. The ``BIND_BLOCK_STRIDED`` axes promote to
``BIND_THREAD`` in the Enclosure (handled by the render's multi-axis
thread decode); body ``BoundLoop`` over those axes is stripped (the
thread decode already binds the Var); ``Write`` is unconditional.

Renderer-level correctness for nested-reduce ``Accum`` init is a
*separate* gap — the current ``_render_loop`` emits Accum init at the
immediate parent Loop, which gives wrong semantics for chunked-K matmul
(``Loop(k_o) > Loop(k_i) > Accum`` resets ``acc`` per ``k_o`` iteration).
Tests here assert on the structural KernelOp shape only.
"""

from __future__ import annotations

from pathlib import Path

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_BLOCK_STRIDED, BIND_SERIAL, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel.ir import Enclosure, KernelOp, Smem, StridedLoop
from deplodock.compiler.ir.stmt import Accum, Assign, Cond, Load, Loop, Write
from deplodock.compiler.ir.tile.ir import BoundLoop, Stage, Tile, TileOp
from deplodock.compiler.pipeline.engine import run_pass

_KERNEL_PASS_DIR = Path(__file__).resolve().parents[4] / "deplodock/compiler/pipeline/passes/lowering/kernel"


def _matmul_post_blockify_tile() -> TileOp:
    """A post-blockify matmul Tile with BM·BN = BLOCK_SIZE (16·16 = 256)
    and Stages at the head for A and B operand caching."""
    BM, BN, BK = 16, 16, 16
    m_o = Axis("m_o", 8)
    n_o = Axis("n_o", 8)
    m_i = Axis("m_i", BM)
    n_i = Axis("n_i", BN)
    k_o = Axis("k_o", 4)
    k_i = Axis("k_i", BK)

    a_idx = (Var("m_o") * Literal(BM, "int") + Var("m_i"), Var("k_i"))
    b_idx = (Var("k_i"), Var("n_o") * Literal(BN, "int") + Var("n_i"))
    c_idx = (Var("m_o") * Literal(BM, "int") + Var("m_i"), Var("n_o") * Literal(BN, "int") + Var("n_i"))

    inner_compute = (
        Load(name="a", input="A", index=a_idx),
        Load(name="b", input="B", index=b_idx),
        Assign(name="t", op=ElementwiseImpl("multiply"), args=("a", "b")),
        Accum(name="acc", value="t", op=ElementwiseImpl("add")),
    )

    k_o_loop = BoundLoop(
        axis=BoundAxis(axis=k_o, bind=BIND_SERIAL),
        body=(BoundLoop(axis=BoundAxis(axis=k_i, bind=BIND_SERIAL), body=inner_compute),),
    )

    m_i_loop = BoundLoop(
        axis=BoundAxis(axis=m_i, bind=BIND_BLOCK_STRIDED),
        body=(
            BoundLoop(
                axis=BoundAxis(axis=n_i, bind=BIND_BLOCK_STRIDED),
                body=(k_o_loop, Write(output="C", index=c_idx, value="acc")),
            ),
        ),
    )

    tile = Tile(
        axes=(
            BoundAxis(axis=m_o, bind=BIND_BLOCK),
            BoundAxis(axis=n_o, bind=BIND_BLOCK),
            BoundAxis(axis=m_i, bind=BIND_BLOCK_STRIDED),
            BoundAxis(axis=n_i, bind=BIND_BLOCK_STRIDED),
        ),
        body=(
            Stage(buf="A", index=a_idx, axes=(m_i, k_i)),
            Stage(buf="B", index=b_idx, axes=(k_i, n_i)),
            m_i_loop,
        ),
    )
    return TileOp(body=(tile,), name="k_matmul")


def _materialize(tile_op: TileOp) -> KernelOp:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("A", (128, 16)), node_id="A")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("B", (16, 128)), node_id="B")
    g.add_node(op=tile_op, inputs=["A", "B"], output=Tensor("C", (128, 128)), node_id="C")
    g.inputs = ["A", "B"]
    g.outputs = ["C"]
    g = run_pass(g, _KERNEL_PASS_DIR)
    return g.nodes["C"].op


def test_matmul_enclosure_has_thread_and_block_axes():
    kernel_op = _materialize(_matmul_post_blockify_tile())
    encl = next(s for s in kernel_op.body if isinstance(s, Enclosure))

    # Order: thread axes (m_i, n_i) first, then block axes (m_o, n_o).
    binds = [ba.bind for ba in encl.axes]
    names = [ba.axis.name for ba in encl.axes]
    assert binds == [BIND_THREAD, BIND_THREAD, BIND_BLOCK, BIND_BLOCK]
    assert names == ["m_i", "n_i", "m_o", "n_o"]


def test_matmul_strips_body_boundloops_over_thread_axes():
    """The ``m_i`` and ``n_i`` BoundLoops in the input are stripped — their
    Vars come from the Enclosure thread decode instead."""
    kernel_op = _materialize(_matmul_post_blockify_tile())
    encl = next(s for s in kernel_op.body if isinstance(s, Enclosure))

    # Body should contain Smem/StridedLoop/Sync from staging, then Loop(k_o), then Write.
    # No BoundLoop or Loop with axis name in {m_i, n_i}.
    def names_of_loops(stmts):
        for s in stmts:
            if isinstance(s, Loop):
                yield s.axis.name
                yield from names_of_loops(s.body)
            elif isinstance(s, StridedLoop):
                yield s.axis.name
                yield from names_of_loops(s.body)

    loop_names = list(names_of_loops(encl.body))
    assert "m_i" not in loop_names
    assert "n_i" not in loop_names
    # k_o, k_i still present (they're real reduction loops).
    assert "k_o" in loop_names
    assert "k_i" in loop_names


def test_matmul_write_is_unconditional():
    """No ``Cond on tid==0`` — every thread writes its own output slot."""
    kernel_op = _materialize(_matmul_post_blockify_tile())
    # Walk the entire kernel body — there should be no Cond statements.
    for s in kernel_op:
        assert not isinstance(s, Cond), f"Unexpected Cond in matmul-materialized body: {s}"
    # The Write should be present (somewhere in the body).
    writes = [s for s in kernel_op if isinstance(s, Write) and s.output == "C"]
    assert len(writes) == 1


def test_matmul_stage_uses_linear_tid_from_thread_axes():
    """Cooperative load's StridedLoop.start = ``m_i * BN + n_i`` (no
    synthesized ``t`` axis — the linear tid is computed from the
    Enclosure thread axes directly)."""
    kernel_op = _materialize(_matmul_post_blockify_tile())
    flat_loops = [s for s in kernel_op if isinstance(s, StridedLoop) and s.axis.name.startswith("_stage_")]
    assert len(flat_loops) == 2
    for sl in flat_loops:
        # start = m_i * 16 + n_i  (BinaryExpr +, with left = m_i*16, right = n_i)
        assert isinstance(sl.start, BinaryExpr) and sl.start.op == "+"
        left = sl.start.left
        right = sl.start.right
        assert isinstance(left, BinaryExpr) and left.op == "*"
        assert isinstance(left.left, Var) and left.left.name == "m_i"
        assert isinstance(left.right, Literal) and left.right.value == 16
        assert isinstance(right, Var) and right.name == "n_i"


def test_matmul_smems_declared_with_2d_extents():
    kernel_op = _materialize(_matmul_post_blockify_tile())
    smem_by_name = {s.name: s for s in kernel_op if isinstance(s, Smem)}
    assert smem_by_name["A_stage"].extents == (16, 16)
    assert smem_by_name["B_stage"].extents == (16, 16)


def _matmul_with_stages_inside_k_o() -> TileOp:
    """Realistic post-blockify matmul: Stages live inside the K_o loop
    (so smem refills per K-chunk), not at the Tile head. Tests that
    matmul materialization recurses into outer-serial loops to expand
    Stages."""
    BM, BN, BK = 16, 16, 16
    m_o = Axis("m_o", 8)
    n_o = Axis("n_o", 8)
    m_i = Axis("m_i", BM)
    n_i = Axis("n_i", BN)
    k_o = Axis("k_o", 4)
    k_i = Axis("k_i", BK)

    a_idx = (Var("m_o") * Literal(BM, "int") + Var("m_i"), Var("k_o") * Literal(BK, "int") + Var("k_i"))
    b_idx = (Var("k_o") * Literal(BK, "int") + Var("k_i"), Var("n_o") * Literal(BN, "int") + Var("n_i"))
    c_idx = (Var("m_o") * Literal(BM, "int") + Var("m_i"), Var("n_o") * Literal(BN, "int") + Var("n_i"))

    inner_compute = (
        Load(name="a", input="A", index=a_idx),
        Load(name="b", input="B", index=b_idx),
        Assign(name="t", op=ElementwiseImpl("multiply"), args=("a", "b")),
        Accum(name="acc", value="t", op=ElementwiseImpl("add")),
    )

    k_o_loop = BoundLoop(
        axis=BoundAxis(axis=k_o, bind=BIND_SERIAL),
        body=(
            Stage(buf="A", index=a_idx, axes=(m_i, k_i)),
            Stage(buf="B", index=b_idx, axes=(k_i, n_i)),
            BoundLoop(axis=BoundAxis(axis=k_i, bind=BIND_SERIAL), body=inner_compute),
        ),
    )

    m_i_loop = BoundLoop(
        axis=BoundAxis(axis=m_i, bind=BIND_BLOCK_STRIDED),
        body=(
            BoundLoop(
                axis=BoundAxis(axis=n_i, bind=BIND_BLOCK_STRIDED),
                body=(k_o_loop, Write(output="C", index=c_idx, value="acc")),
            ),
        ),
    )

    tile = Tile(
        axes=(
            BoundAxis(axis=m_o, bind=BIND_BLOCK),
            BoundAxis(axis=n_o, bind=BIND_BLOCK),
            BoundAxis(axis=m_i, bind=BIND_BLOCK_STRIDED),
            BoundAxis(axis=n_i, bind=BIND_BLOCK_STRIDED),
        ),
        body=(m_i_loop,),
    )
    return TileOp(body=(tile,), name="k_matmul")


def test_matmul_stages_inside_k_o_loop_expand_in_place():
    """When Stages live inside the K_o serial loop, they should expand to
    Smem/StridedLoop/Sync at that scope (so smem refills per K-chunk)."""
    kernel_op = _materialize(_matmul_with_stages_inside_k_o())
    encl = next(s for s in kernel_op.body if isinstance(s, Enclosure))

    # Top-level body of Enclosure has the (rewritten) k_o Loop and the Write.
    k_o_loop = next(s for s in encl.body if isinstance(s, Loop) and s.axis.name == "k_o")
    # Inside k_o: Smem, StridedLoop (cooperative load), Sync, Smem, StridedLoop, Sync, Loop(k_i).
    smems_inside = [s for s in k_o_loop.body if isinstance(s, Smem)]
    assert len(smems_inside) == 2  # A_stage, B_stage
    inner_loops = [s for s in k_o_loop.body if isinstance(s, Loop)]
    assert len(inner_loops) == 1 and inner_loops[0].axis.name == "k_i"


def test_matmul_renders_with_single_acc_init_above_outer_loop():
    """End-to-end: the full matmul materialization should produce CUDA
    with exactly one ``float acc = 0.0f;`` declaration, placed above the
    outer K loop (not inside it). Verifies the Init hoisting closes the
    nested-reduce render gap."""
    from deplodock.compiler.ir.kernel.render import render_kernelop

    kernel_op = _materialize(_matmul_with_stages_inside_k_o())
    src = render_kernelop(kernel_op, shapes={"A": (128, 64), "B": (64, 128), "C": (128, 128)})
    # Single declaration of acc.
    assert src.count("float acc = 0.0f;") == 1
    # acc declared before any for loop.
    lines = src.splitlines()
    init_line = next(i for i, ln in enumerate(lines) if "float acc = 0.0f;" in ln)
    first_for = next(i for i, ln in enumerate(lines) if "for (int" in ln)
    assert init_line < first_for


def test_softmax_style_still_uses_synthetic_t_axis():
    """A Tile with a single BIND_BLOCK_STRIDED axis whose extent != BLOCK_SIZE
    is *not* matmul-style; should route to the original cooperative path."""
    a0 = Axis("a0", 4)
    a2 = Axis("a2", 512)  # cooperative reduction axis, extent != BLOCK_SIZE
    tile = Tile(
        axes=(BoundAxis(axis=a0, bind=BIND_BLOCK), BoundAxis(axis=a2, bind=BIND_BLOCK_STRIDED)),
        body=(
            BoundLoop(
                axis=BoundAxis(axis=a2, bind=BIND_BLOCK_STRIDED),
                body=(
                    Load(name="x", input="X", index=(Var("a0"), Var("a2"))),
                    Accum(name="acc", value="x", op=ElementwiseImpl("add")),
                ),
            ),
            Write(output="Y", index=(Var("a0"),), value="acc"),
        ),
    )
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("X", (4, 512)), node_id="X")
    g.add_node(
        op=TileOp(body=(tile,), name="k_softmax"),
        inputs=["X"],
        output=Tensor("Y", (4,)),
        node_id="Y",
    )
    g.inputs = ["X"]
    g.outputs = ["Y"]
    g = run_pass(g, _KERNEL_PASS_DIR)

    encl = next(s for s in g.nodes["Y"].op.body if isinstance(s, Enclosure))
    # Softmax path: synthesized t axis prepended.
    assert encl.axes[0].axis.name == "t"
    assert encl.axes[0].bind == BIND_THREAD
