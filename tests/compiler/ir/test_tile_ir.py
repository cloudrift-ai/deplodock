"""Tile IR construction smoke tests.

Step 1 of the Tile IR refactor adds dataclass definitions only — no
emitter, no lowering, no pass. The verification here is that the types
compile, can be instantiated with realistic args, and can express each of
the kernel shapes we know about (pointwise, RMSNorm, matmul). The next
step (``ir/tile/render.py``) will turn these fixtures into CUDA source.
"""

from __future__ import annotations

from deplodock.compiler.ir.tile import (
    Acc,
    AccumFold,
    Axis,
    BinaryExpr,
    Cond,
    Coop,
    FreeLoop,
    FuncCallExpr,
    Index,
    Kernel,
    Let,
    Literal,
    Param,
    Reduce,
    SmemBuf,
    Store,
    Sync,
    Tile,
    Var,
)

# ---------------------------------------------------------------------------
# Per-node smoke tests — every concrete class instantiates with realistic args
# ---------------------------------------------------------------------------


def test_index_expression():
    idx = Index("A", (Var("i"), Var("j")))
    assert idx.buf == "A"
    assert idx.indices[0] == Var("i")


def test_index_uses_expr_ops_mixin():
    """``Index`` inherits ``_ExprOps`` so arithmetic returns ``BinaryExpr``."""
    a = Index("A", (Var("i"), Var("k")))
    b = Index("B", (Var("k"), Var("j")))
    assert isinstance(a + b, BinaryExpr)
    assert (a + b).op == "+"


def test_let_store_accumfold():
    let = Let("t", Index("X", (Var("i"),)))
    store = Store("out", (Var("i"), Var("j")), Var("t"))
    fold = AccumFold("c", "add", Var("p"))
    assert (let.name, store.buf, fold.op.name) == ("t", "out", "add")


def test_sync_cond():
    sync = Sync()
    branch = Cond(
        cond=BinaryExpr("<", Var("tid"), Literal(256, "int")),
        body=(sync,),
        else_body=(),
    )
    assert isinstance(branch.body[0], Sync)
    assert branch.else_body == ()


def test_loops():
    a0 = Axis("a0", 4)
    a1 = Axis("a1", 8)
    free = FreeLoop(axis=a0, body=())
    red = Reduce(axis=a1, accs=(Acc("c", "add"),), body=())
    tile = Tile(axis=a1, bk=4, body=())
    coop = Coop(cover=32, var="i", body=())
    assert (free.axis.name, red.axis.name, tile.bk, coop.cover) == ("a0", "a1", 4, 32)


def test_reduce_extent_default_none():
    red = Reduce(axis=Axis("k", 32), accs=(Acc("s", "add"),), body=())
    assert red.extent is None  # caller / render uses axis.extent


def test_param_smembuf():
    p = Param(name="x", dtype="const float*")
    sb = SmemBuf(name="A_tile", dtype="float", dims=(128, 16))
    assert (p.dtype, sb.dims) == ("const float*", (128, 16))


def test_kernel_defaults():
    k = Kernel(name="k0", params=(), body=())
    assert k.thread_axes == ()
    assert k.block_axes == ()
    assert k.grid == (1, 1, 1)
    assert k.block == (1, 1, 1)
    assert k.smem == ()
    assert k.prologue == ()


# ---------------------------------------------------------------------------
# Shape fixtures — hand-rolled kernels mirroring expected post-lowering form
# ---------------------------------------------------------------------------


def test_pointwise_add_shape():
    """``c[i, j] = a[i, j] + b[i, j]`` — single FreeLoop nest, Let + Store.

    Mirrors what ``lower_naive`` should produce for a 2D pointwise add
    before ``ExtractGlobalSchedule`` strips the outer loops to thread axes.
    """
    i, j = Axis("a0", 4), Axis("a1", 8)
    body = (
        FreeLoop(
            axis=i,
            body=(
                FreeLoop(
                    axis=j,
                    body=(
                        Let("a_v", Index("A", (Var("a0"), Var("a1")))),
                        Let("b_v", Index("B", (Var("a0"), Var("a1")))),
                        Let("c_v", BinaryExpr("+", Var("a_v"), Var("b_v"))),
                        Store("out", (Var("a0"), Var("a1")), Var("c_v")),
                    ),
                ),
            ),
        ),
    )
    k = Kernel(
        name="add",
        params=(Param("A", "const float*"), Param("B", "const float*"), Param("out", "float*")),
        body=body,
    )
    # Walk: Kernel.body[0] = FreeLoop(i), .body[0] = FreeLoop(j), .body[-1] = Store.
    assert isinstance(k.body[0], FreeLoop)
    assert isinstance(k.body[0].body[0], FreeLoop)
    assert isinstance(k.body[0].body[0].body[-1], Store)


def test_rmsnorm_shape():
    """RMSNorm post-lowering: prologue scalar Loads, FreeLoop(i), Reduce(k),
    interlude Lets, FreeLoop(j) ending in Store."""
    i = Axis("a0", 4)
    k_axis = Axis("a1", 32)
    j = Axis("a2", 32)
    prologue = (
        Let("eps", Index("Eps", (Literal(0, "int"),))),
        Let("mean_n", Index("MeanN", (Literal(0, "int"),))),
    )
    reduce_block = Reduce(
        axis=k_axis,
        accs=(Acc("s", "add"),),
        body=(
            Let("x", Index("X", (Var("a0"), Var("a1")))),
            Let("sq", BinaryExpr("*", Var("x"), Var("x"))),
            AccumFold("s", "add", Var("sq")),
        ),
    )
    interlude = (
        Let("m", BinaryExpr("/", Var("s"), Var("mean_n"))),
        Let("me", BinaryExpr("+", Var("m"), Var("eps"))),
        Let("r", FuncCallExpr("rsqrt", [Var("me")])),
    )
    output_loop = FreeLoop(
        axis=j,
        body=(
            Let("xj", Index("X", (Var("a0"), Var("a2")))),
            Let("wj", Index("W", (Var("a2"),))),
            Let("xr", BinaryExpr("*", Var("xj"), Var("r"))),
            Let("y", BinaryExpr("*", Var("xr"), Var("wj"))),
            Store("out", (Var("a0"), Var("a2")), Var("y")),
        ),
    )
    body = (FreeLoop(axis=i, body=(reduce_block, *interlude, output_loop)),)
    k = Kernel(
        name="rmsnorm",
        params=(
            Param("X", "const float*"),
            Param("Eps", "const float*"),
            Param("MeanN", "const float*"),
            Param("W", "const float*"),
            Param("out", "float*"),
        ),
        prologue=prologue,
        body=body,
    )
    assert len(k.prologue) == 2
    outer = k.body[0]
    assert isinstance(outer, FreeLoop) and outer.axis.name == "a0"
    assert isinstance(outer.body[0], Reduce)
    assert isinstance(outer.body[-1], FreeLoop) and outer.body[-1].axis.name == "a2"


def test_matmul_naive_shape():
    """Matmul ``c[m, n] = sum_k a[m, k] * b[k, n]`` — naive form, no Tile/Coop yet."""
    m, n, k = Axis("a0", 64), Axis("a1", 64), Axis("a2", 32)
    body = (
        FreeLoop(
            axis=m,
            body=(
                FreeLoop(
                    axis=n,
                    body=(
                        Reduce(
                            axis=k,
                            accs=(Acc("c", "add"),),
                            body=(
                                Let("a_v", Index("A", (Var("a0"), Var("a2")))),
                                Let("b_v", Index("B", (Var("a2"), Var("a1")))),
                                Let("p", BinaryExpr("*", Var("a_v"), Var("b_v"))),
                                AccumFold("c", "add", Var("p")),
                            ),
                        ),
                        Store("out", (Var("a0"), Var("a1")), Var("c")),
                    ),
                ),
            ),
        ),
    )
    krn = Kernel(
        name="matmul",
        params=(Param("A", "const float*"), Param("B", "const float*"), Param("out", "float*")),
        body=body,
    )
    inner = krn.body[0].body[0]
    assert isinstance(inner.body[0], Reduce)
    assert inner.body[0].accs[0].op.name == "add"
    assert isinstance(inner.body[-1], Store)


def test_matmul_smem_tiled_shape():
    """Matmul after ``TileReduce(BK=16)`` + ``SmemStageReduce`` — verifies
    that ``Tile`` + ``Coop`` + ``Sync`` + ``Reduce`` (with smem-backed Index
    in the body) all compose into a kernel.
    """
    m, n, k = Axis("a0", 128), Axis("a1", 128), Axis("a2", 64)
    smem = (
        SmemBuf("A_tile", "float", (16, 16)),
        SmemBuf("B_tile", "float", (16, 16)),
    )
    inner_reduce = Reduce(
        axis=Axis("k_inner", 16),
        accs=(Acc("c", "add"),),
        body=(
            Let("a_v", Index("A_tile", (Var("m_local"), Var("k_inner")))),
            Let("b_v", Index("B_tile", (Var("k_inner"), Var("n_local")))),
            Let("p", BinaryExpr("*", Var("a_v"), Var("b_v"))),
            AccumFold("c", "add", Var("p")),
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
                    Let("g", Index("A", (Var("a0"), BinaryExpr("+", Var("a2"), Var("i"))))),
                    Store("A_tile", (Var("i"),), Var("g")),
                ),
            ),
            Coop(
                cover=16 * 16,
                var="i",
                body=(
                    Let("g", Index("B", (BinaryExpr("+", Var("a2"), Var("i")), Var("a1")))),
                    Store("B_tile", (Var("i"),), Var("g")),
                ),
            ),
            Sync(),
            inner_reduce,
            Sync(),
        ),
    )
    body = (tile_loop, Store("out", (Var("a0"), Var("a1")), Var("c")))
    krn = Kernel(
        name="matmul_tiled",
        params=(Param("A", "const float*"), Param("B", "const float*"), Param("out", "float*")),
        smem=smem,
        thread_axes=(m, n),
        body=body,
    )
    assert len(krn.smem) == 2
    outer = krn.body[0]
    assert isinstance(outer, Tile) and outer.bk == 16
    assert isinstance(outer.body[0], Coop)
    assert isinstance(outer.body[2], Sync)
    assert isinstance(outer.body[3], Reduce)
