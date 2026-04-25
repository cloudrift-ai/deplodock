"""Tile IR construction smoke tests.

After dropping ``Let`` / ``Index`` / ``Store`` / ``AccumFold`` / ``Acc``,
Tile IR re-uses Loop IR's leaf stmts (``Load`` / ``Assign`` / ``Select`` /
``Write`` / ``Accum``). These tests verify that the schedule wrappers
(``Loop`` / ``Reduce`` / ``Tile`` / ``Coop`` / ``Cond`` / ``Sync``) +
``TileOp`` accept the Loop IR leaves and can express each kernel shape we
plan to lower to.
"""

from __future__ import annotations

from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.ir.tile import (
    Accum,
    Assign,
    Axis,
    BinaryExpr,
    Cond,
    Coop,
    ElementwiseImpl,
    Enclosure,
    Literal,
    Load,
    Loop,
    Reduce,
    SmemBuf,
    Sync,
    Tile,
    TileOp,
    Var,
    Write,
)

# ---------------------------------------------------------------------------
# Per-node smoke tests
# ---------------------------------------------------------------------------


def test_load_construction():
    ld = Load(name="t", input="X", index=(Var("i"), Var("j")))
    assert ld.input == "X" and ld.name == "t"


def test_write_construction():
    w = Write(output="out", index=(Var("i"),), value="t")
    assert w.output == "out" and w.value == "t"


def test_accum_carries_elementwiseimpl_op():
    acc = Accum(name="c", value="p", op="add")
    assert isinstance(acc.op, ElementwiseImpl) and acc.op.name == "add"
    assert acc.op.identity == 0.0


def test_sync_cond():
    branch = Cond(
        cond=BinaryExpr("<", Var("tid"), Literal(256, "int")),
        body=(Sync(),),
        else_body=(),
    )
    assert isinstance(branch.body[0], Sync)
    assert branch.else_body == ()


def test_loops():
    a0 = Axis("a0", 4)
    a1 = Axis("a1", 8)
    free = Loop(axis=a0, body=())
    red = Reduce(axis=a1, body=())
    tile = Tile(axis=a1, bk=4, body=())
    coop = Coop(cover=32, var="i", body=())
    assert (free.axis.name, red.axis.name, tile.bk, coop.cover) == ("a0", "a1", 4, 32)


def test_reduce_extent_default_none():
    red = Reduce(axis=Axis("k", 32), body=())
    assert red.extent is None  # caller / render uses axis.extent


def test_smembuf():
    sb = SmemBuf(name="A_tile", dtype="float", dims=(128, 16))
    assert sb.dims == (128, 16)


def test_tileop_defaults():
    k = TileOp(name="k0")
    assert k.body == ()
    assert k.smem == ()
    assert k.name == "k0"
    assert k.inputs == ()
    assert k.outputs == ()


def test_tileop_inputs_outputs_derived_from_body():
    body = (
        Load("a", input="X", index=(Var("i"),)),
        Load("b", input="Y", index=(Var("i"),)),
        Load("c", input="X", index=(Var("j"),)),  # duplicate input
        Write(output="out", index=(Var("i"),), value="a"),
    )
    k = TileOp(name="k", body=body)
    # First-use order, deduped.
    assert k.inputs == ("X", "Y")
    assert k.outputs == ("out",)


def test_enclosure_construction():
    enc = Enclosure(thread_axes=(Axis("a0", 4),), block_axes=(), body=(Sync(),))
    assert enc.thread_axes[0].name == "a0"
    assert enc.block_axes == ()
    assert isinstance(enc.body[0], Sync)


# ---------------------------------------------------------------------------
# Shape fixtures — hand-rolled kernels mirroring expected post-lowering form
# ---------------------------------------------------------------------------


def test_pointwise_add_shape():
    """``c[a0, a1] = a[a0, a1] + b[a0, a1]`` — single Loop nest, Loop IR
    leaves directly."""
    i, j = Axis("a0", 4), Axis("a1", 8)
    body = (
        Loop(
            axis=i,
            body=(
                Loop(
                    axis=j,
                    body=(
                        Load("a_v", input="A", index=(Var("a0"), Var("a1"))),
                        Load("b_v", input="B", index=(Var("a0"), Var("a1"))),
                        Assign("c_v", ElementwiseOp("add"), ("a_v", "b_v")),
                        Write(output="out", index=(Var("a0"), Var("a1")), value="c_v"),
                    ),
                ),
            ),
        ),
    )
    k = TileOp(name="add", body=body)
    assert isinstance(k.body[0], Loop)
    assert isinstance(k.body[0].body[0], Loop)
    assert isinstance(k.body[0].body[0].body[-1], Write)
    assert k.inputs == ("A", "B")
    assert k.outputs == ("out",)


def test_rmsnorm_shape():
    """RMSNorm post-lowering: scalar Loads sit in body before the Loop(i),
    which contains Reduce(k), interlude Assigns, Loop(j) ending in Write."""
    i = Axis("a0", 4)
    k_axis = Axis("a1", 32)
    j = Axis("a2", 32)
    reduce_block = Reduce(
        axis=k_axis,
        body=(
            Load("xk", input="X", index=(Var("a0"), Var("a1"))),
            Assign("sq", ElementwiseOp("multiply"), ("xk", "xk")),
            Accum(name="s", value="sq", op="add"),
        ),
    )
    output_loop = Loop(
        axis=j,
        body=(
            Load("xj", input="X", index=(Var("a0"), Var("a2"))),
            Load("wj", input="W", index=(Var("a2"),)),
            Assign("xr", ElementwiseOp("multiply"), ("xj", "s")),  # contrived
            Assign("y", ElementwiseOp("multiply"), ("xr", "wj")),
            Write(output="out", index=(Var("a0"), Var("a2")), value="y"),
        ),
    )
    # Pre-Enclosure scalar Loads + outer free Loop carry the work.
    body = (
        Load("eps", input="Eps", index=()),
        Load("mean_n", input="MeanN", index=()),
        Loop(axis=i, body=(reduce_block, output_loop)),
    )
    k = TileOp(name="rmsnorm", body=body)
    assert isinstance(k.body[0], Load) and k.body[0].input == "Eps"
    outer = k.body[2]
    assert isinstance(outer, Loop) and outer.axis.name == "a0"
    assert isinstance(outer.body[0], Reduce)
    assert isinstance(outer.body[-1], Loop) and outer.body[-1].axis.name == "a2"


def test_matmul_naive_shape():
    """Matmul ``c[m, n] = sum_k a[m, k] * b[k, n]`` — Loop+Loop+Reduce+Write."""
    m, n, k = Axis("a0", 64), Axis("a1", 64), Axis("a2", 32)
    body = (
        Loop(
            axis=m,
            body=(
                Loop(
                    axis=n,
                    body=(
                        Reduce(
                            axis=k,
                            body=(
                                Load("a_v", input="A", index=(Var("a0"), Var("a2"))),
                                Load("b_v", input="B", index=(Var("a2"), Var("a1"))),
                                Assign("p", ElementwiseOp("multiply"), ("a_v", "b_v")),
                                Accum(name="c", value="p", op="add"),
                            ),
                        ),
                        Write(output="out", index=(Var("a0"), Var("a1")), value="c"),
                    ),
                ),
            ),
        ),
    )
    krn = TileOp(
        name="matmul",
        body=body,
    )
    inner = krn.body[0].body[0]
    assert isinstance(inner.body[0], Reduce)
    # First Accum in the reduce body fixes the op.
    accums = [s for s in inner.body[0].body if isinstance(s, Accum)]
    assert accums[0].op.name == "add"
    assert isinstance(inner.body[-1], Write)


def test_matmul_smem_tiled_shape():
    """Matmul after ``TileReduce(BK=16)`` + ``SmemStageReduce`` — verifies
    that ``Tile`` + ``Coop`` + ``Sync`` + ``Reduce`` (with smem-backed Loads)
    compose into a kernel."""
    m, n, k = Axis("a0", 128), Axis("a1", 128), Axis("a2", 64)
    smem = (
        SmemBuf("A_tile", "float", (16, 16)),
        SmemBuf("B_tile", "float", (16, 16)),
    )
    inner_reduce = Reduce(
        axis=Axis("k_inner", 16),
        body=(
            Load("a_v", input="A_tile", index=(Var("m_local"), Var("k_inner"))),
            Load("b_v", input="B_tile", index=(Var("k_inner"), Var("n_local"))),
            Assign("p", ElementwiseOp("multiply"), ("a_v", "b_v")),
            Accum(name="c", value="p", op="add"),
        ),
    )
    tile_loop = Tile(
        axis=k,
        bk=16,
        body=(
            Coop(
                cover=16 * 16,
                var="i",
                body=(
                    Load("g", input="A", index=(Var("a0"), BinaryExpr("+", Var("a2"), Var("i")))),
                    Write(output="A_tile", index=(Var("i"),), value="g"),
                ),
            ),
            Sync(),
            inner_reduce,
            Sync(),
        ),
    )
    # Wrap the schedulable body in an Enclosure that binds (m, n) to thread coords.
    enclosed = Enclosure(
        thread_axes=(m, n),
        block_axes=(),
        body=(tile_loop, Write(output="out", index=(Var("a0"), Var("a1")), value="c")),
    )
    krn = TileOp(
        name="matmul_tiled",
        smem=smem,
        body=(enclosed,),
    )
    assert len(krn.smem) == 2
    enc = krn.body[0]
    assert isinstance(enc, Enclosure) and enc.thread_axes[0].name == "a0"
    outer = enc.body[0]
    assert isinstance(outer, Tile) and outer.bk == 16
    assert isinstance(outer.body[0], Coop)
    assert isinstance(outer.body[1], Sync)
    assert isinstance(outer.body[2], Reduce)
