"""Matmul register-tile coverage — one accuracy + structure matrix over (TILE variant × shape).

The scalar contraction's free-axis output tile (the ``TILE`` codec — ``n<N>[xm<M>]`` parallel
thread-tile / ``f<fn>[xf<fm>]`` register sub-tile) lowers each thread to a ``reg_m × reg_n``
block of output cells, the reduce-loop body replicated per cell with its operand loads deduped
(``A[m,k]`` reused across the ``n`` cells, ``B[k,n]`` across the ``m`` cells) — the
arithmetic-intensity lever for scalar SGEMM. This test pins each TILE variant and checks the
contraction stays accurate vs numpy AND emits the matching lowering structure (register
replication, the ``#pragma unroll``'d inner reduce, the per-CTA thread count), over BOTH a
static M and a SYMBOLIC M (the dynamic-grid tier: the launch sizes from the runtime extent, the
overhang cell clamp-reads + skips its store). All pins are the ``DEPLODOCK_TILE`` codec — the
register-tile analogue of ``test_reduction_combine_coverage``'s ``DEPLODOCK_REDUCE`` matrix; no
legacy ``BN`` / ``BM`` / ``FM`` / ``FN``.

Pure GPU accuracy (no ``-O1`` numerics change), so it runs in the correctness lane.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..conftest import dyn_M, requires_cuda, requires_sm90

# Square base shape; divisible by every variant's parallel·register product so the static
# column is exact-cover (one CTA where the variant asks for it) and the dynamic column runs at
# an off-divisor length to exercise the masked tail.
_M = _K = _N = 64
_DYN_M = 70  # off the 64 base → a partial last register-row when M is register-tiled


def _matmul_graph(mode: str):
    """``(1, M, K) @ (K, N)``; ``mode='dynamic'`` makes the M (row) axis symbolic."""
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp

    Mg = dyn_M(mode, _M)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (1, Mg, _K)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (_K, _N)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (1, Mg, _N)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    return g


def _run(mode: str, tile: str, monkeypatch) -> tuple[np.ndarray, np.ndarray, str]:
    """Compile the matmul under the pinned ``DEPLODOCK_TILE`` codec, run on seeded inputs at the
    mode's runtime M, and return ``(output, reference, kernel_source)``."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    monkeypatch.setenv("DEPLODOCK_TILE", tile)
    m = _DYN_M if mode == "dynamic" else _M
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1, m, _K), dtype=np.float32)
    b = rng.standard_normal((_K, _N), dtype=np.float32)
    be = CudaBackend()
    compiled = be.compile(_matmul_graph(mode))
    got = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["c"])
    src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None))
    return got, (a @ b), src


# (label, TILE codec, expects-register-replication, expected __launch_bounds__ or None).
#   none        ("")            — one thread per cell, no register replication / unroll
#   reg_inner   (f4)            — 4 register cells along N, B-load shared across them
#   reg_2d      (f2xf2)         — full 2×2 register block, both operands reused
#   single_cta  (n32xm16/f2xf4) — par·reg == 64×64 ⇒ one 512-thread CTA (static)
_VARIANTS = {
    "none": ("", False, None),
    "reg_inner": ("f4", True, None),
    "reg_2d": ("f2xf2", True, None),
    "single_cta": ("n32xm16/f2xf4", True, 512),
}
_SHAPES = ("static", "dynamic")


@pytest.mark.parametrize("variant", list(_VARIANTS))
@pytest.mark.parametrize("mode", _SHAPES)
@requires_cuda
def test_matmul_tile_coverage(variant, mode, monkeypatch):
    tile, has_reg, launch_bounds = _VARIANTS[variant]
    got, ref, src = _run(mode, tile, monkeypatch)

    diff = float(np.abs(got - ref.reshape(got.shape)).max())
    assert diff < 1e-3, f"{variant}/{mode}: matmul mismatch (max abs err {diff})"

    has_copy = "__c0_1" in src or "__c1_0" in src  # a replicated register-cell binding
    if has_reg:
        assert has_copy, f"{variant}/{mode}: expected replicated register cells (__c*)"
        assert "#pragma unroll" in src, f"{variant}/{mode}: the small inner reduce must be unrolled"
    else:
        assert not has_copy, f"{variant}/{mode}: per-cell tier must not replicate register cells"
    if launch_bounds is not None:
        assert f"__launch_bounds__({launch_bounds})" in src, f"{variant}/{mode}: expected a {launch_bounds}-thread CTA"

    if mode == "dynamic":
        # The dynamic-grid tier: the launch sizes from the runtime extent (the symbolic ``Dim``
        # threaded as an ``int`` arg), and a register-tiled symbolic axis guards its tail store.
        assert "int seq_len" in src, f"{variant}/dynamic: symbolic grid must carry the runtime extent arg"


# Fused epilogues — a projection ``Map`` over the ``Semiring`` (``project ∘ contract``): the
# pointwise op folds into the contraction kernel's tail, replicated per register cell. Each is a
# distinct tail shape: a broadcast scalar, a per-``n`` bias (shared across the ``m`` cells), a
# pure activation, and a full ``(m, n)`` residual (no sharing). Pinned to a 2×2 register tile so
# the reg-tile tail-replication + load-dedup is exercised by every epilogue.
_EPILOGUE_TILE = "n16xm16/f2xf2"
_EPILOGUES = ("scale", "bias", "relu", "residual")


def _epilogue_graph(mode: str, epilogue: str):
    """``(1, M, K) @ (K, N)`` with a fused pointwise ``epilogue`` on the contraction output."""
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp
    from deplodock.compiler.ir.tensor.ir import ElementwiseOp

    Mg = dyn_M(mode, _M)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (1, Mg, _K)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (_K, _N)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("ab", (1, Mg, _N)), node_id="ab")
    inputs = ["a", "b"]
    if epilogue == "scale":
        g.add_node(InputOp(), [], Tensor("s", (1,)), node_id="s")
        g.add_node(ElementwiseOp("multiply"), ["ab", "s"], Tensor("o", (1, Mg, _N)), node_id="o")
        inputs.append("s")
    elif epilogue == "bias":
        g.add_node(InputOp(), [], Tensor("bias", (_N,)), node_id="bias")
        g.add_node(ElementwiseOp("add"), ["ab", "bias"], Tensor("o", (1, Mg, _N)), node_id="o")
        inputs.append("bias")
    elif epilogue == "relu":
        g.add_node(ElementwiseOp("relu"), ["ab"], Tensor("o", (1, Mg, _N)), node_id="o")
    else:  # residual — a full (1, M, N) add (depends on both cell axes, no load sharing)
        g.add_node(InputOp(), [], Tensor("r", (1, Mg, _N)), node_id="r")
        g.add_node(ElementwiseOp("add"), ["ab", "r"], Tensor("o", (1, Mg, _N)), node_id="o")
        inputs.append("r")
    g.inputs, g.outputs = inputs, ["o"]
    return g


def _epilogue_ref(epilogue: str, feed: dict) -> np.ndarray:
    base = feed["a"] @ feed["b"]
    if epilogue == "scale":
        return base * feed["s"]
    if epilogue == "bias":
        return base + feed["bias"]
    if epilogue == "relu":
        return np.maximum(base, 0.0)
    return base + feed["r"]


@pytest.mark.parametrize("epilogue", _EPILOGUES)
@pytest.mark.parametrize("mode", _SHAPES)
@requires_cuda
def test_matmul_reg_tile_epilogue(epilogue, mode, monkeypatch):
    """A register-tiled contraction with a fused pointwise epilogue stays accurate AND folds the
    epilogue into the ONE contraction kernel (no separate elementwise launch), over static and
    symbolic M. The epilogue is replicated per register cell in the tail (a per-``n`` bias shared
    across the ``m`` cells, a full residual not shared)."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    monkeypatch.setenv("DEPLODOCK_TILE", _EPILOGUE_TILE)
    m = _DYN_M if mode == "dynamic" else _M
    rng = np.random.default_rng(0)
    feed = {"a": rng.standard_normal((1, m, _K), dtype=np.float32), "b": rng.standard_normal((_K, _N), dtype=np.float32)}
    if epilogue == "scale":
        feed["s"] = np.array([1.5], dtype=np.float32)
    elif epilogue == "bias":
        feed["bias"] = rng.standard_normal((_N,), dtype=np.float32)
    elif epilogue == "residual":
        feed["r"] = rng.standard_normal((1, m, _N), dtype=np.float32)

    be = CudaBackend()
    compiled = be.compile(_epilogue_graph(mode, epilogue))
    got = np.asarray(be.run(compiled, input_data=feed)[0].outputs["o"])
    src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None))

    ref = _epilogue_ref(epilogue, feed)
    diff = float(np.abs(got - ref.reshape(got.shape)).max())
    assert diff < 1e-3, f"{epilogue}/{mode}: fused-epilogue mismatch (max abs err {diff})"
    assert src.count("__global__") == 1, f"{epilogue}/{mode}: epilogue must fuse into the one contraction kernel"
    assert "__c0_1" in src or "__c1_0" in src, f"{epilogue}/{mode}: expected the register-tiled tail (__c*)"


# --------------------------------------------------------------------------- #
# Warp-tier (tensor-core mma.sync) coverage — the same accuracy + structure matrix idea, now
# over the ``Warp`` fragment (the ``WARP`` codec) instead of the scalar ``TILE`` codec. The
# contraction tiles onto ``WM·WN`` warps of ``mma_m16n8k16`` atom cells: f16 operands, f32
# accumulate, f16|f32 store, gmem-direct (no smem staging this tier). All pins are the
# ``DEPLODOCK_WARP`` codec — the tensor-core analogue of the ``DEPLODOCK_TILE`` matrix above;
# no legacy ``MMA`` / ``WM`` / ``WN`` / ``FM`` / ``FN`` / ``BK``. Requires sm_90+ (the warp tier
# is pin-only and non-functional below — ldmatrix host fault + ``sm_NNa`` TMA compile).
_WARP_PIN = "a:mma_m16n8k16_f16/w2xw2/f4xf8/k2"  # WM·FM·atom_m = WN·FN·atom_n = 128 tile, 128 threads
_F16 = "f16"


def _dtype(name: str):
    from deplodock.compiler.dtype import F16, F32

    return {"f16": F16, "f32": F32}[name]


def _mma_matmul_graph(mode: str, M: int, N: int, K: int, out: str, trans: bool):
    """A hand-built ``C[i,j] = Σ_k A[i,k]·B[…]`` over f16 operands; ``trans`` makes B ``[j,k]``
    (Q@Kᵀ, K last). ``mode='dynamic'`` makes the M (row) axis symbolic (the dynamic-grid tier)."""
    from deplodock.compiler.dtype import F16
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.elementwise import ElementwiseImpl
    from deplodock.compiler.ir.expr import Var
    from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
    from deplodock.compiler.ir.stmt import Accum, Assign

    Mg = dyn_M(mode, M)
    i, j, k = Axis("i", Mg), Axis("j", N), Axis("k", K)
    b_index = (Var("j"), Var("k")) if trans else (Var("k"), Var("j"))
    op = LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Loop(
                                axis=k,
                                body=(
                                    Load(name="a_v", input="a", index=(Var("i"), Var("k"))),
                                    Load(name="b_v", input="b", index=b_index),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a_v", "b_v")),
                                    Accum(name="acc", value="p"),
                                ),
                            ),
                            Write(output="c", index=(Var("i"), Var("j")), value="acc"),
                        ),
                    ),
                ),
            ),
        ),
    )
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Mg, K), dtype=F16), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (N, K) if trans else (K, N), dtype=F16), node_id="b")
    g.add_node(op, ["a", "b"], Tensor("c", (Mg, N), dtype=_dtype(out)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    return g


def _compile_run_mma(graph, run_m: int, feed_extra: dict) -> tuple[np.ndarray, str]:
    """Compile the (already WARP-pinned) graph and run it at runtime ``run_m`` on seeded f16
    operands; return ``(output, kernel_source)``."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    rng = np.random.default_rng(0)
    feed = dict(feed_extra)
    be = CudaBackend()
    compiled = be.compile(graph)
    got = np.asarray(be.run(compiled, input_data=feed)[0].outputs["c"])
    src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None))
    return got, src


# (M, N, K, out_dtype, transposed-B). 128 / 256 are tile (128) multiples (exact-cover static);
# the dynamic column runs at M+2 to straddle the tile and exercise the masked store / clamp.
_MMA_CASES = [
    (128, 128, 128, "f32", False),
    (256, 256, 128, "f32", False),
    (128, 128, 128, "f16", False),
    (128, 256, 128, "f16", False),
    (128, 128, 128, "f32", True),  # transposed-B (Q@Kᵀ)
    (128, 128, 128, "f16", True),
]


@pytest.mark.parametrize(("M", "N", "K", "out", "trans"), _MMA_CASES)
@pytest.mark.parametrize("mode", _SHAPES)
@requires_sm90
@requires_cuda
def test_matmul_mma_coverage(M, N, K, out, trans, mode, monkeypatch):
    """An f16×f16 matmul under the pinned ``DEPLODOCK_WARP`` codec lowers to ``mma.sync`` and
    agrees with the f32 reference (within fp16 tolerance) — for canonical AND transposed-B
    operands, f16 AND f32 output, over a STATIC M (tile divisor) and a SYMBOLIC M run at a
    straddling length (the dynamic-grid tier + the masked-tile store)."""
    monkeypatch.setenv("DEPLODOCK_WARP", _WARP_PIN)
    run_m = M if mode == "static" else M + 2
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((run_m, K)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((N, K) if trans else (K, N)) * 0.1).astype(np.float16)
    got, src = _compile_run_mma(_mma_matmul_graph(mode, M, N, K, out, trans), run_m, {"a": a, "b": b})

    ref = a.astype(np.float32) @ (b.T if trans else b).astype(np.float32)
    diff = float(np.abs(got.reshape(run_m, N) - ref).max())
    tol = 5e-2 if out == _F16 else 1e-2
    assert diff < tol, f"{M}x{N}x{K} out={out} trans={trans}/{mode}: mma mismatch (max abs err {diff})"

    assert "mma.sync.aligned.m16n8k16" in src, "the s16816 mma.sync instruction must be emitted"
    assert "dpl_mma_load_a_gmem" in src, "operands must load via the gmem-direct fragment helper"
    assert "wmma::" not in src, "the mma.sync path must not mix in legacy wmma intrinsics"
    assert ("__floats2half2_rn" in src) == (out == _F16), "f16 output needs the fp32→fp16 __half2 downconvert"
    if trans:
        assert "dpl_mma_load_b_gmem_trans" in src, "transposed-B must use the gmem-direct trans helper"
    if mode == "dynamic":
        assert "int seq_len" in src, "the symbolic-M grid must carry the runtime extent arg"


# Fused epilogues over the warp tier — a projection ``Map`` (or a causal ``Select``) folds into
# the ``RegStore`` per fragment element. Each is a distinct tail: a per-``n`` bias (shared across
# the m rows), a pure activation, a full ``(m, n)`` residual, and a coord-predicated causal mask.
_MMA_EPILOGUES = ("bias", "relu", "residual", "causal")


def _mma_epilogue_graph(mode: str, epilogue: str):
    """An f16 ``128³`` matmul with a fused ``epilogue`` (causal uses a hand-built coord ``Select``;
    the rest fold a frontend ``ElementwiseOp``)."""
    from deplodock.compiler.dtype import F16, F32
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp

    M = N = K = 128
    Mg = dyn_M(mode, M)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (Mg, K), dtype=F16), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N), dtype=F16), node_id="b")
    if epilogue == "causal":
        from deplodock.compiler.ir.elementwise import ElementwiseImpl
        from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
        from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
        from deplodock.compiler.ir.stmt import Accum, Assign, Select, SelectBranch

        i, j, k = Axis("i", Mg), Axis("j", N), Axis("k", K)
        op = LoopOp(
            body=(
                Loop(
                    axis=i,
                    body=(
                        Loop(
                            axis=j,
                            body=(
                                Load(name="mz", input="mz", index=(Literal(0, "int"),)),
                                Load(name="mf", input="mf", index=(Literal(0, "int"),)),
                                Loop(
                                    axis=k,
                                    body=(
                                        Load(name="a_v", input="a", index=(Var("i"), Var("k"))),
                                        Load(name="b_v", input="b", index=(Var("k"), Var("j"))),
                                        Assign(name="p", op=ElementwiseImpl("multiply"), args=("a_v", "b_v")),
                                        Accum(name="acc", value="p"),
                                    ),
                                ),
                                Select(
                                    name="out",
                                    branches=(
                                        SelectBranch(value="acc", select=BinaryExpr("<=", Var("j"), Var("i"))),
                                        SelectBranch(value="mf", select=BinaryExpr(">", Var("j"), Var("i"))),
                                    ),
                                ),
                                Write(output="c", index=(Var("i"), Var("j")), value="out"),
                            ),
                        ),
                    ),
                ),
            ),
        )
        g.add_node(InputOp(), [], Tensor("mz", (1,), dtype=F32), node_id="mz")
        g.add_node(InputOp(), [], Tensor("mf", (1,), dtype=F32), node_id="mf")
        g.add_node(op, ["a", "b", "mz", "mf"], Tensor("c", (Mg, N), dtype=F32), node_id="c")
        g.inputs, g.outputs = ["a", "b", "mz", "mf"], ["c"]
        return g

    from deplodock.compiler.ir.frontend.ir import MatmulOp
    from deplodock.compiler.ir.tensor.ir import ElementwiseOp

    g.add_node(MatmulOp(), ["a", "b"], Tensor("ab", (Mg, N), dtype=F32), node_id="ab")
    inputs = ["a", "b"]
    if epilogue == "bias":
        g.add_node(InputOp(), [], Tensor("bias", (N,), dtype=F32), node_id="bias")
        g.add_node(ElementwiseOp("add"), ["ab", "bias"], Tensor("c", (Mg, N), dtype=F32), node_id="c")
        inputs.append("bias")
    elif epilogue == "relu":
        g.add_node(ElementwiseOp("relu"), ["ab"], Tensor("c", (Mg, N), dtype=F32), node_id="c")
    else:  # residual
        g.add_node(InputOp(), [], Tensor("r", (Mg, N), dtype=F32), node_id="r")
        g.add_node(ElementwiseOp("add"), ["ab", "r"], Tensor("c", (Mg, N), dtype=F32), node_id="c")
        inputs.append("r")
    g.inputs, g.outputs = inputs, ["c"]
    return g


@pytest.mark.parametrize("epilogue", _MMA_EPILOGUES)
@pytest.mark.parametrize("mode", _SHAPES)
@requires_sm90
@requires_cuda
def test_matmul_mma_epilogue_coverage(epilogue, mode, monkeypatch):
    """A warp-tier matmul with a fused pointwise / causal epilogue stays accurate AND folds the
    epilogue into the ONE mma.sync kernel (the per-element ``RegStore`` chain), over static and
    symbolic M."""
    monkeypatch.setenv("DEPLODOCK_WARP", _WARP_PIN)
    M = N = K = 128
    run_m = M if mode == "static" else M + 2
    rng = np.random.default_rng(1)
    a = (rng.standard_normal((run_m, K)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    base = a.astype(np.float32) @ b.astype(np.float32)
    feed = {"a": a, "b": b}
    if epilogue == "bias":
        bias = rng.standard_normal((N,)).astype(np.float32)
        feed["bias"] = bias
        ref = base + bias
    elif epilogue == "relu":
        ref = np.maximum(base, 0.0)
    elif epilogue == "residual":
        r = rng.standard_normal((run_m, N)).astype(np.float32)
        feed["r"] = r
        ref = base + r
    else:  # causal
        feed["mz"] = np.array([0.0], np.float32)
        feed["mf"] = np.array([-1e30], np.float32)
        keep = np.arange(N)[None, :] <= np.arange(run_m)[:, None]
        ref = np.where(keep, base, -1e30)

    got, src = _compile_run_mma(_mma_epilogue_graph(mode, epilogue), run_m, feed)
    diff = float(np.abs(got.reshape(run_m, N) - ref).max())
    assert diff < 1e-2, f"{epilogue}/{mode}: fused-epilogue mma mismatch (max abs err {diff})"
    assert "mma.sync.aligned.m16n8k16" in src, f"{epilogue}/{mode}: must reach the warp tier"
    assert src.count("__global__") == 1, f"{epilogue}/{mode}: epilogue must fuse into the one mma kernel"
    if epilogue == "causal":
        assert "?" in src, "the causal mask must render as a per-element ternary"
