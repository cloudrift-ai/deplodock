"""Tests for the block-tiled matmul rule (``lowering/tile/_block_matmul``).

The rule is not auto-loaded yet (filename is ``_``-prefixed) so these
tests invoke it directly via ``run_rule``. They assert on the post-
rewrite Tile-IR shape; end-to-end materialization is gated on multi-axis
``Stage`` support and multi-axis ``BIND_BLOCK_STRIDED`` flattening, both
deferred to a follow-up.
"""

from __future__ import annotations

from pathlib import Path

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_BLOCK_STRIDED, BIND_SERIAL
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.loop import Accum, Assign, Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.tile.ir import BoundLoop, Tile, TileOp
from deplodock.compiler.ir.tile.lower import lower_naive
from deplodock.compiler.pipeline.engine import run_rule

_RULE_PATH = Path(__file__).resolve().parents[4] / "deplodock/compiler/pipeline/passes/lowering/tile/_block_matmul.py"


def _matmul_graph(M: int, N: int, K: int) -> Graph:
    """Build a Graph with one fused-matmul LoopOp computing C = A @ B."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("A", (M, K)), node_id="A")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("B", (K, N)), node_id="B")
    m = Axis("m", M)
    n = Axis("n", N)
    k = Axis("k", K)
    body = (
        Loop(
            axis=m,
            body=(
                Loop(
                    axis=n,
                    body=(
                        Loop(
                            axis=k,
                            body=(
                                Load(name="a", input="A", index=(Var("m"), Var("k"))),
                                Load(name="b", input="B", index=(Var("k"), Var("n"))),
                                Assign(name="t", op=ElementwiseImpl("multiply"), args=("a", "b")),
                                Accum(name="acc", value="t", op=ElementwiseImpl("add")),
                            ),
                        ),
                        Write(output="C", index=(Var("m"), Var("n")), value="acc"),
                    ),
                ),
            ),
        ),
    )
    g.add_node(op=LoopOp(body=body), inputs=["A", "B"], output=Tensor("C", (M, N)), node_id="C")
    g.inputs = ["A", "B"]
    g.outputs = ["C"]
    return g


def _lower_and_blockify(graph: Graph) -> Graph:
    for node in list(graph.nodes.values()):
        if isinstance(node.op, LoopOp):
            node.op = lower_naive(node.op, kernel_name=f"k_{node.id}_matmul")
    return run_rule(graph, _RULE_PATH)


def _the_tile(graph: Graph) -> Tile:
    tile_op = graph.nodes["C"].op
    assert isinstance(tile_op, TileOp)
    tiles = [s for s in tile_op.body if isinstance(s, Tile)]
    assert len(tiles) == 1
    return tiles[0]


def test_blockify_rewrites_matmul_tile_axes():
    g = _lower_and_blockify(_matmul_graph(M=128, N=128, K=64))
    tile = _the_tile(g)

    # Four output axes after blockify.
    assert len(tile.axes) == 4
    binds = [ba.bind for ba in tile.axes]
    assert binds == [BIND_BLOCK, BIND_BLOCK, BIND_BLOCK_STRIDED, BIND_BLOCK_STRIDED]

    # Outer axes get _o suffix and divided extents; inner are the BM/BN tiles.
    # Pre-rule axis names are canonicalized to a0/a1; post-rule names suffix _o/_i.
    names = [ba.axis.name for ba in tile.axes]
    extents = [ba.axis.extent for ba in tile.axes]
    assert names == ["a0_o", "a1_o", "a0_i", "a1_i"]
    assert extents == [128 // 64, 128 // 64, 64, 64]


def test_blockify_nests_loops_and_chunks_k():
    g = _lower_and_blockify(_matmul_graph(M=128, N=128, K=64))
    tile = _the_tile(g)

    # Body is exactly one outer BoundLoop (a0_i) wrapping a1_i wrapping (a2_o, write).
    assert len(tile.body) == 1
    m_i_loop = tile.body[0]
    assert isinstance(m_i_loop, BoundLoop)
    assert m_i_loop.axis.axis.name == "a0_i" and m_i_loop.bind == BIND_BLOCK_STRIDED

    n_i_loop = m_i_loop.body[0]
    assert isinstance(n_i_loop, BoundLoop)
    assert n_i_loop.axis.axis.name == "a1_i" and n_i_loop.bind == BIND_BLOCK_STRIDED

    # Under n_i: a serial k_o BoundLoop and the Write.
    k_o_loop, write = n_i_loop.body
    assert isinstance(k_o_loop, BoundLoop)
    assert k_o_loop.axis.axis.name == "a2_o" and k_o_loop.bind == BIND_SERIAL
    assert k_o_loop.axis.axis.extent == 64 // 16

    k_i_loop = k_o_loop.body[0]
    assert isinstance(k_i_loop, BoundLoop)
    assert k_i_loop.axis.axis.name == "a2_i" and k_i_loop.bind == BIND_SERIAL
    assert k_i_loop.axis.axis.extent == 16

    # Inner compute: Load A, Load B, Assign mul, Accum add — same shape as
    # before, only indices substituted.
    inner = k_i_loop.body
    assert len(inner) == 4
    load_a, load_b, mul, accum = inner
    assert isinstance(load_a, Load) and load_a.input == "A"
    assert isinstance(load_b, Load) and load_b.input == "B"
    assert isinstance(mul, Assign) and mul.op.name == "multiply"
    assert isinstance(accum, Accum) and accum.op.name == "add"

    # Index substitution: A[a0_o*64 + a0_i, a2_o*16 + a2_i].
    _assert_split_index(load_a.index, ("a0_o", 64, "a0_i"), ("a2_o", 16, "a2_i"))
    _assert_split_index(load_b.index, ("a2_o", 16, "a2_i"), ("a1_o", 64, "a1_i"))
    assert isinstance(write, Write)
    _assert_split_index(write.index, ("a0_o", 64, "a0_i"), ("a1_o", 64, "a1_i"))


def _assert_split_index(index, *expected) -> None:
    """Each expected entry is (outer_var_name, factor, inner_var_name).
    Asserts the corresponding index Expr is ``outer * factor + inner``."""
    assert len(index) == len(expected)
    for expr, (o, f, i) in zip(index, expected, strict=True):
        assert isinstance(expr, BinaryExpr) and expr.op == "+"
        mul = expr.left
        inner = expr.right
        assert isinstance(mul, BinaryExpr) and mul.op == "*"
        assert isinstance(mul.left, Var) and mul.left.name == o
        assert isinstance(mul.right, Literal) and mul.right.value == f
        assert isinstance(inner, Var) and inner.name == i


def test_blockify_idempotent():
    g = _lower_and_blockify(_matmul_graph(M=128, N=128, K=64))
    tile_first = _the_tile(g)
    g_again = run_rule(g, _RULE_PATH)
    tile_second = _the_tile(g_again)
    # Second pass must not re-rewrite (already block-bound).
    assert tile_first.axes == tile_second.axes


def test_blockify_skips_non_divisible_extents():
    # M=130 not divisible by BM=64 → rule must not fire.
    g = _lower_and_blockify(_matmul_graph(M=130, N=128, K=64))
    tile = _the_tile(g)
    assert len(tile.axes) == 2  # untouched (still pre-blockify thread-axes shape)


def test_blockify_skips_non_matmul_shape():
    """A pure 2D pointwise (no reduce loop) must not match."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("X", (128, 128)), node_id="X")
    m = Axis("m", 128)
    n = Axis("n", 128)
    body = (
        Loop(
            axis=m,
            body=(
                Loop(
                    axis=n,
                    body=(
                        Load(name="x", input="X", index=(Var("m"), Var("n"))),
                        Write(output="Y", index=(Var("m"), Var("n")), value="x"),
                    ),
                ),
            ),
        ),
    )
    g.add_node(op=LoopOp(body=body), inputs=["X"], output=Tensor("Y", (128, 128)), node_id="Y")
    g.inputs = ["X"]
    g.outputs = ["Y"]

    g = _lower_and_blockify(g)
    tile_op = g.nodes["Y"].op
    tile = next(s for s in tile_op.body if isinstance(s, Tile))
    assert len(tile.axes) == 2  # unchanged
