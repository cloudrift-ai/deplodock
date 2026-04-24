"""Tile IR → CUDA renderer tests.

Step 2 of the Tile IR refactor: ``render_kernel`` turns a ``Kernel`` into
a complete ``extern "C" __global__`` CUDA function. These tests assert
exact CUDA snippets against per-node fixtures and structural pieces
against whole-kernel fixtures (pointwise / RMSNorm / matmul / smem-tiled
matmul).
"""

from __future__ import annotations

from deplodock.compiler.ir.tile import (
    Acc,
    AccumFold,
    Axis,
    BinaryExpr,
    Builtin,
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
from deplodock.compiler.ir.tile.render import render_kernel


def _kernel(*, name: str = "k", params=(), body=(), **kw) -> Kernel:
    """Compact constructor for one-off renderer fixtures."""
    return Kernel(name=name, params=params, body=body, **kw)


# ---------------------------------------------------------------------------
# Per-node rendering
# ---------------------------------------------------------------------------


def test_let_renders_as_var_decl():
    out = render_kernel(_kernel(body=(Let("t", Literal(3.0)),)))
    assert "float t = 3.0f;" in out


def test_let_with_binary_expr():
    out = render_kernel(_kernel(body=(Let("t", BinaryExpr("+", Var("a"), Var("b"))),)))
    assert "float t = a + b;" in out


def test_store_scalar_index():
    params = (Param("out", "float*", shape=(8,)),)
    body = (Store("out", (Var("i"),), Var("v")),)
    out = render_kernel(_kernel(params=params, body=body))
    assert "out[i] = v;" in out


def test_store_2d_row_major_flatten():
    params = (Param("out", "float*", shape=(4, 8)),)
    body = (Store("out", (Var("i"), Var("j")), Var("v")),)
    out = render_kernel(_kernel(params=params, body=body))
    assert "out[i * 8 + j] = v;" in out


def test_index_2d_in_let():
    params = (Param("X", "const float*", shape=(4, 8)),)
    body = (Let("v", Index("X", (Var("i"), Var("j")))),)
    out = render_kernel(_kernel(params=params, body=body))
    assert "float v = X[i * 8 + j];" in out


def test_accumfold_add_uses_compound_assign():
    out = render_kernel(_kernel(body=(AccumFold("a", "add", Var("x")),)))
    assert "a += x;" in out


def test_accumfold_max_uses_fmaxf():
    out = render_kernel(_kernel(body=(AccumFold("a", "max", Var("x")),)))
    assert "a = fmaxf(a, x);" in out


def test_accumfold_mul_uses_compound_assign():
    out = render_kernel(_kernel(body=(AccumFold("a", "mul", Var("x")),)))
    assert "a *= x;" in out


def test_sync_renders_syncthreads():
    out = render_kernel(_kernel(body=(Sync(),)))
    assert "__syncthreads();" in out


def test_cond_if_only():
    body = (Cond(BinaryExpr("<", Var("i"), Literal(8, "int")), body=(Sync(),)),)
    out = render_kernel(_kernel(body=body))
    assert "if (i < 8) {" in out
    assert "__syncthreads();" in out
    assert "} else {" not in out


def test_cond_if_else():
    body = (
        Cond(
            cond=BinaryExpr("<", Var("i"), Literal(8, "int")),
            body=(Sync(),),
            else_body=(Let("t", Literal(0.0)),),
        ),
    )
    out = render_kernel(_kernel(body=body))
    assert "if (i < 8) {" in out
    assert "} else {" in out
    assert "float t = 0.0f;" in out


def test_freeloop_renders_for():
    body = (FreeLoop(axis=Axis("k", 32), body=(Sync(),)),)
    out = render_kernel(_kernel(body=body))
    assert "for (int k = 0; k < 32; k++) {" in out


def test_reduce_declares_acc_then_loops():
    body = (
        Reduce(
            axis=Axis("k", 32),
            accs=(Acc("s", "add", Literal(0.0)),),
            body=(AccumFold("s", "add", Var("x")),),
        ),
    )
    out = render_kernel(_kernel(body=body))
    assert "float s = 0.0f;" in out
    assert "for (int k = 0; k < 32; k++) {" in out
    assert "s += x;" in out


def test_reduce_with_explicit_extent():
    body = (
        Reduce(
            axis=Axis("k", 1024),
            accs=(Acc("s", "add", Literal(0.0)),),
            body=(),
            extent=16,
        ),
    )
    out = render_kernel(_kernel(body=body))
    # Inner loop uses the explicit extent (the slab-local count), not axis.extent.
    assert "for (int k = 0; k < 16; k++) {" in out


def test_tile_renders_for_with_step():
    body = (Tile(axis=Axis("k", 64), bk=16, body=(Sync(),)),)
    out = render_kernel(_kernel(body=body))
    assert "for (int k = 0; k < 64; k += 16) {" in out


def test_coop_uses_thread_idx_and_block_dim():
    body = (Coop(cover=64, var="i", body=(Sync(),)),)
    out = render_kernel(_kernel(body=body))
    assert "for (int i = threadIdx.x; i < 64; i += blockDim.x) {" in out


def test_intrinsic_translates_rsqrt():
    body = (Let("r", FuncCallExpr("rsqrt", [Var("x")])),)
    out = render_kernel(_kernel(body=body))
    assert "float r = rsqrtf(x);" in out


def test_builtin_translates_thread_idx():
    body = (Let("t", Builtin("thread_idx.x")),)
    out = render_kernel(_kernel(body=body))
    assert "float t = threadIdx.x;" in out


# ---------------------------------------------------------------------------
# Kernel-level wrapping
# ---------------------------------------------------------------------------


def test_header_extern_c_global():
    out = render_kernel(_kernel(name="my_k", params=(Param("X", "const float*"),)))
    assert 'extern "C" __global__' in out
    assert "void my_k(const float* X)" in out


def test_launch_bounds_set_for_small_blocks():
    out = render_kernel(_kernel(block=(256, 1, 1)))
    assert "__launch_bounds__(256)" in out


def test_launch_bounds_skipped_for_huge_blocks():
    out = render_kernel(_kernel(block=(1025, 1, 1)))
    assert "__launch_bounds__" not in out


def test_smem_decl_top_of_body():
    smem = (SmemBuf("A_tile", "float", (16, 16)),)
    out = render_kernel(_kernel(smem=smem))
    assert "__shared__ float A_tile[16][16];" in out


def test_prologue_emitted_above_tid_guard():
    params = (Param("Eps", "const float*", shape=(1,)),)
    prologue = (Let("eps", Index("Eps", (Literal(0, "int"),))),)
    out = render_kernel(_kernel(params=params, prologue=prologue, thread_axes=(Axis("a0", 4),)))
    # Prologue line must appear before the tid decode.
    assert out.index("float eps = Eps[0];") < out.index("long long tid =")


def test_thread_axes_emit_tid_decode_and_guard_1d():
    out = render_kernel(_kernel(thread_axes=(Axis("a0", 32),)))
    assert "long long tid = blockIdx.x * blockDim.x + threadIdx.x;" in out
    assert "if (tid < 32) {" in out
    assert "int a0 = tid;" in out


def test_thread_axes_2d_decode():
    out = render_kernel(_kernel(thread_axes=(Axis("a0", 4), Axis("a1", 8))))
    assert "if (tid < 32) {" in out
    # innermost (a1) = tid % 8; outermost (a0) = tid / 8.
    assert "int a1 = tid % 8;" in out
    assert "int a0 = tid / 8;" in out


def test_no_thread_axes_no_guard():
    out = render_kernel(_kernel(thread_axes=()))
    assert "if (tid <" not in out


# ---------------------------------------------------------------------------
# Whole-kernel fixtures (mirror step 1's structural fixtures)
# ---------------------------------------------------------------------------


def _pointwise_add_kernel() -> Kernel:
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
    return Kernel(
        name="add",
        params=(
            Param("A", "const float*", shape=(4, 8)),
            Param("B", "const float*", shape=(4, 8)),
            Param("out", "float*", shape=(4, 8)),
        ),
        body=body,
    )


def test_render_pointwise_add_kernel():
    src = render_kernel(_pointwise_add_kernel())
    assert "for (int a0 = 0; a0 < 4; a0++) {" in src
    assert "for (int a1 = 0; a1 < 8; a1++) {" in src
    assert "float a_v = A[a0 * 8 + a1];" in src
    assert "float c_v = a_v + b_v;" in src
    assert "out[a0 * 8 + a1] = c_v;" in src


def _rmsnorm_kernel() -> Kernel:
    i, k_axis, j = Axis("a0", 4), Axis("a1", 32), Axis("a2", 32)
    prologue = (
        Let("eps", Index("Eps", (Literal(0, "int"),))),
        Let("mean_n", Index("MeanN", (Literal(0, "int"),))),
    )
    reduce_block = Reduce(
        axis=k_axis,
        accs=(Acc("s", "add", Literal(0.0)),),
        body=(
            Let("x", Index("X", (Var("a0"), Var("a1")))),
            Let("sq", BinaryExpr("*", Var("x"), Var("x"))),
            AccumFold("s", "add", Var("sq")),
        ),
    )
    body = (
        FreeLoop(
            axis=i,
            body=(
                reduce_block,
                Let("m", BinaryExpr("/", Var("s"), Var("mean_n"))),
                Let("me", BinaryExpr("+", Var("m"), Var("eps"))),
                Let("r", FuncCallExpr("rsqrt", [Var("me")])),
                FreeLoop(
                    axis=j,
                    body=(
                        Let("xj", Index("X", (Var("a0"), Var("a2")))),
                        Let("wj", Index("W", (Var("a2"),))),
                        Let("xr", BinaryExpr("*", Var("xj"), Var("r"))),
                        Let("y", BinaryExpr("*", Var("xr"), Var("wj"))),
                        Store("out", (Var("a0"), Var("a2")), Var("y")),
                    ),
                ),
            ),
        ),
    )
    return Kernel(
        name="rmsnorm",
        params=(
            Param("X", "const float*", shape=(4, 32)),
            Param("Eps", "const float*", shape=(1,)),
            Param("MeanN", "const float*", shape=(1,)),
            Param("W", "const float*", shape=(32,)),
            Param("out", "float*", shape=(4, 32)),
        ),
        prologue=prologue,
        body=body,
    )


def test_render_rmsnorm_kernel():
    src = render_kernel(_rmsnorm_kernel())
    assert "float eps = Eps[0];" in src
    assert "float mean_n = MeanN[0];" in src
    assert "float s = 0.0f;" in src
    assert "for (int a1 = 0; a1 < 32; a1++) {" in src
    assert "s += sq;" in src
    assert "float r = rsqrtf(me);" in src
    assert "for (int a2 = 0; a2 < 32; a2++) {" in src
    assert "out[a0 * 32 + a2] = y;" in src


def _matmul_naive_kernel() -> Kernel:
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
                            accs=(Acc("c", "add", Literal(0.0)),),
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
    return Kernel(
        name="matmul",
        params=(
            Param("A", "const float*", shape=(64, 32)),
            Param("B", "const float*", shape=(32, 64)),
            Param("out", "float*", shape=(64, 64)),
        ),
        body=body,
    )


def test_render_matmul_naive():
    src = render_kernel(_matmul_naive_kernel())
    assert "float c = 0.0f;" in src
    assert "for (int a2 = 0; a2 < 32; a2++) {" in src
    assert "c += p;" in src
    assert "out[a0 * 64 + a1] = c;" in src


def _matmul_smem_tiled_kernel() -> Kernel:
    """Mirrors the post-`SmemStageReduce` shape from step 1's fixture."""
    m, n, k = Axis("a0", 128), Axis("a1", 128), Axis("a2", 64)
    smem = (
        SmemBuf("A_tile", "float", (16, 16)),
        SmemBuf("B_tile", "float", (16, 16)),
    )
    inner_reduce = Reduce(
        axis=Axis("k_inner", 16),
        accs=(Acc("c", "add", Literal(0.0)),),
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
            Coop(cover=256, var="i", body=(Sync(),)),
            Coop(cover=256, var="i", body=(Sync(),)),
            Sync(),
            inner_reduce,
            Sync(),
        ),
    )
    body = (tile_loop, Store("out", (Var("a0"), Var("a1")), Var("c")))
    return Kernel(
        name="matmul_tiled",
        params=(
            Param("A", "const float*", shape=(128, 64)),
            Param("B", "const float*", shape=(64, 128)),
            Param("out", "float*", shape=(128, 128)),
        ),
        smem=smem,
        thread_axes=(m, n),
        body=body,
    )


def test_render_matmul_smem_tiled():
    src = render_kernel(_matmul_smem_tiled_kernel())
    assert "__shared__ float A_tile[16][16];" in src
    assert "__shared__ float B_tile[16][16];" in src
    # Outer Tile slab walk over k with stride 16.
    assert "for (int a2 = 0; a2 < 64; a2 += 16) {" in src
    # Cooperative loops use threadIdx.x.
    assert "for (int i = threadIdx.x; i < 256; i += blockDim.x) {" in src
    # Inner Reduce uses smem-backed Index — flatten over (16, 16) shape.
    assert "float a_v = A_tile[m_local * 16 + k_inner];" in src
    # Sync between stage and reduce.
    assert "__syncthreads();" in src
    # Thread-axes guard wraps the body.
    assert "if (tid < 16384) {" in src  # 128 * 128
