"""Tile IR → CUDA renderer tests.

After the Loop-IR-leaf reuse, the renderer dispatches on Loop IR's
``Load`` / ``Assign`` / ``Select`` / ``Write`` / ``Accum`` plus Tile IR's
schedule wrappers. These tests assert exact CUDA snippets for each node
type and structural pieces of whole-kernel fixtures.
"""

from __future__ import annotations

from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.ir.tile import (
    Accum,
    Assign,
    Axis,
    BinaryExpr,
    Builtin,
    Cond,
    Coop,
    Enclosure,
    Literal,
    Load,
    Loop,
    Param,
    Reduce,
    SmemBuf,
    Sync,
    Tile,
    TileOp,
    Var,
    Write,
)
from deplodock.compiler.ir.tile.render import render_tileop


def _kernel(*, name: str = "k", params=(), body=(), **kw) -> TileOp:
    return TileOp(name=name, params=params, body=body, **kw)


# ---------------------------------------------------------------------------
# Per-node rendering
# ---------------------------------------------------------------------------


def test_load_renders_as_var_decl():
    params = (Param("X", "const float*", shape=(8,)),)
    body = (Load("t", input="X", index=(Var("i"),)),)
    out = render_tileop(_kernel(params=params, body=body))
    assert "float t = X[i];" in out


def test_load_2d_row_major_flatten():
    params = (Param("X", "const float*", shape=(4, 8)),)
    body = (Load("v", input="X", index=(Var("i"), Var("j"))),)
    assert "float v = X[i * 8 + j];" in render_tileop(_kernel(params=params, body=body))


def test_assign_binary_op():
    body = (Assign("t", ElementwiseOp("add"), ("a", "b")),)
    assert "float t = a + b;" in render_tileop(_kernel(body=body))


def test_assign_intrinsic():
    body = (Assign("t", ElementwiseOp("rsqrt"), ("x",)),)
    assert "float t = rsqrtf(x);" in render_tileop(_kernel(body=body))


def test_write_2d_row_major_flatten():
    params = (Param("out", "float*", shape=(4, 8)),)
    body = (Write(output="out", index=(Var("i"), Var("j")), value="v"),)
    assert "out[i * 8 + j] = v;" in render_tileop(_kernel(params=params, body=body))


def test_accum_add_uses_compound_assign():
    body = (Accum(name="a", value="x", op="add"),)
    assert "a += x;" in render_tileop(_kernel(body=body))


def test_accum_max_uses_fmaxf():
    body = (Accum(name="a", value="x", op="maximum"),)
    assert "a = fmaxf(a, x);" in render_tileop(_kernel(body=body))


def test_accum_mul_uses_compound_assign():
    body = (Accum(name="a", value="x", op="multiply"),)
    assert "a *= x;" in render_tileop(_kernel(body=body))


def test_sync_renders_syncthreads():
    out = render_tileop(_kernel(body=(Sync(),)))
    assert "__syncthreads();" in out


def test_cond_if_only():
    body = (Cond(BinaryExpr("<", Var("i"), Literal(8, "int")), body=(Sync(),)),)
    out = render_tileop(_kernel(body=body))
    assert "if (i < 8) {" in out
    assert "__syncthreads();" in out
    assert "} else {" not in out


def test_cond_if_else():
    body = (
        Cond(
            cond=BinaryExpr("<", Var("i"), Literal(8, "int")),
            body=(Sync(),),
            else_body=(Load("t", input="X", index=(Literal(0, "int"),)),),
        ),
    )
    out = render_tileop(_kernel(params=(Param("X", "const float*", shape=(1,)),), body=body))
    assert "if (i < 8) {" in out
    assert "} else {" in out


def test_freeloop_renders_for():
    body = (Loop(axis=Axis("k", 32), body=(Sync(),)),)
    assert "for (int k = 0; k < 32; k++) {" in render_tileop(_kernel(body=body))


def test_reduce_declares_acc_then_loops():
    body = (
        Reduce(
            axis=Axis("k", 32),
            body=(
                Load("x", input="X", index=(Var("k"),)),
                Accum(name="s", value="x", op="add"),
            ),
        ),
    )
    out = render_tileop(_kernel(params=(Param("X", "const float*", shape=(32,)),), body=body))
    assert "float s = 0.0f;" in out
    assert "for (int k = 0; k < 32; k++) {" in out
    assert "s += x;" in out


def test_reduce_with_explicit_extent():
    body = (Reduce(axis=Axis("k", 1024), body=(), extent=16),)
    out = render_tileop(_kernel(body=body))
    # Inner loop uses the explicit extent (the slab-local count), not axis.extent.
    assert "for (int k = 0; k < 16; k++) {" in out


def test_tile_renders_for_with_step():
    body = (Tile(axis=Axis("k", 64), bk=16, body=(Sync(),)),)
    assert "for (int k = 0; k < 64; k += 16) {" in render_tileop(_kernel(body=body))


def test_coop_uses_thread_idx_and_block_dim():
    body = (Coop(cover=64, var="i", body=(Sync(),)),)
    assert "for (int i = threadIdx.x; i < 64; i += blockDim.x) {" in render_tileop(_kernel(body=body))


def test_builtin_translates_thread_idx():
    body = (Cond(cond=Builtin("thread_idx.x"), body=(Sync(),)),)
    assert "if (threadIdx.x) {" in render_tileop(_kernel(body=body))


def test_func_call_translates():
    body = (Assign("t", ElementwiseOp("rsqrt"), ("x",)),)
    assert "rsqrtf(x)" in render_tileop(_kernel(body=body))


# ---------------------------------------------------------------------------
# TileOp-level wrapping
# ---------------------------------------------------------------------------


def test_header_extern_c_global():
    out = render_tileop(_kernel(name="my_k", params=(Param("X", "const float*"),)))
    assert 'extern "C" __global__' in out
    assert "void my_k(const float* X)" in out


def test_launch_bounds_always_emitted():
    assert "__launch_bounds__(256)" in render_tileop(_kernel())


def test_smem_decl_top_of_body():
    smem = (SmemBuf("A_tile", "float", (16, 16)),)
    assert "__shared__ float A_tile[16][16];" in render_tileop(_kernel(smem=smem))


def test_pre_enclosure_loads_emit_above_tid_guard():
    """Loads sitting in body before an Enclosure render above the tid decode."""
    params = (Param("Eps", "const float*", shape=(1,)),)
    body = (
        Load("eps", input="Eps", index=(Literal(0, "int"),)),
        Enclosure(thread_axes=(Axis("a0", 4),), block_axes=(), body=()),
    )
    out = render_tileop(_kernel(params=params, body=body))
    assert out.index("float eps = Eps[0];") < out.index("long long tid =")


def test_enclosure_emits_tid_decode_and_guard_1d():
    body = (Enclosure(thread_axes=(Axis("a0", 32),), block_axes=(), body=()),)
    out = render_tileop(_kernel(body=body))
    assert "long long tid = blockIdx.x * blockDim.x + threadIdx.x;" in out
    assert "if (tid < 32) {" in out
    assert "int a0 = tid;" in out


def test_enclosure_2d_axes_decode():
    body = (Enclosure(thread_axes=(Axis("a0", 4), Axis("a1", 8)), block_axes=(), body=()),)
    out = render_tileop(_kernel(body=body))
    assert "if (tid < 32) {" in out
    assert "int a1 = tid % 8;" in out
    assert "int a0 = tid / 8;" in out


def test_no_enclosure_no_guard():
    """A TileOp with no Enclosure in its body is single-thread serial — no tid guard."""
    assert "if (tid <" not in render_tileop(_kernel(body=()))


# ---------------------------------------------------------------------------
# Whole-kernel fixtures
# ---------------------------------------------------------------------------


def _pointwise_add_kernel() -> TileOp:
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
    return TileOp(
        name="add",
        params=(
            Param("A", "const float*", shape=(4, 8)),
            Param("B", "const float*", shape=(4, 8)),
            Param("out", "float*", shape=(4, 8)),
        ),
        body=body,
    )


def test_render_pointwise_add_kernel():
    src = render_tileop(_pointwise_add_kernel())
    assert "for (int a0 = 0; a0 < 4; a0++) {" in src
    assert "for (int a1 = 0; a1 < 8; a1++) {" in src
    assert "float a_v = A[a0 * 8 + a1];" in src
    assert "float c_v = a_v + b_v;" in src
    assert "out[a0 * 8 + a1] = c_v;" in src


def _matmul_naive_kernel() -> TileOp:
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
    return TileOp(
        name="matmul",
        params=(
            Param("A", "const float*", shape=(64, 32)),
            Param("B", "const float*", shape=(32, 64)),
            Param("out", "float*", shape=(64, 64)),
        ),
        body=body,
    )


def test_render_matmul_naive():
    src = render_tileop(_matmul_naive_kernel())
    assert "float c = 0.0f;" in src
    assert "for (int a2 = 0; a2 < 32; a2++) {" in src
    assert "c += p;" in src
    assert "out[a0 * 64 + a1] = c;" in src


def _matmul_smem_tiled_kernel() -> TileOp:
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
            Coop(cover=256, var="i", body=(Sync(),)),
            Sync(),
            inner_reduce,
            Sync(),
        ),
    )
    enclosed = Enclosure(
        thread_axes=(m, n),
        block_axes=(),
        body=(tile_loop, Write(output="out", index=(Var("a0"), Var("a1")), value="c")),
    )
    return TileOp(
        name="matmul_tiled",
        params=(
            Param("A", "const float*", shape=(128, 64)),
            Param("B", "const float*", shape=(64, 128)),
            Param("out", "float*", shape=(128, 128)),
        ),
        smem=smem,
        body=(enclosed,),
    )


def test_render_matmul_smem_tiled():
    src = render_tileop(_matmul_smem_tiled_kernel())
    assert "__shared__ float A_tile[16][16];" in src
    assert "__shared__ float B_tile[16][16];" in src
    assert "for (int a2 = 0; a2 < 64; a2 += 16) {" in src
    assert "for (int i = threadIdx.x; i < 256; i += blockDim.x) {" in src
    assert "float a_v = A_tile[m_local * 16 + k_inner];" in src
    assert "__syncthreads();" in src
    assert "if (tid < 16384) {" in src  # 128 * 128
