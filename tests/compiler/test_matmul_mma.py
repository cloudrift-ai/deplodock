"""End-to-end MMA matmul correctness — M8 of
``plans/mma-fragment-factorization.md``.

Verifies the WMMA F16 path produces correct output across realistic
shapes + both output dtypes (F32 accumulator-direct-store, F16 acc
with __half store). Pins ``DEPLODOCK_MMA=1`` and lets the planner pick
the best-scoring warp-tier variant; compiles via ``nvcc --cubin`` (NVRTC
fails to compile WMMA — cupy's bundled cu13 lacks ``crt/mma.h``) and
compares against the f32 matmul reference within fp16 tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.dtype import F16, F32, DataType
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Assign
from deplodock.compiler.pipeline import KERNEL_PASSES, Pipeline

from .conftest import requires_cuda

# Route every test in this module to the single ``cuda`` xdist_group
# (``tests/conftest.py::_is_cuda_item`` detects the ``"CUDA not available"``
# skipif reason) so they run sequentially on one worker — the host has one
# GPU and scattering CUDA tests across workers exhausts the device context.
# The per-test ``_supports_wmma`` skipif still gates the sm_70+ arch
# requirement on top of this.
pytestmark = [requires_cuda]


def _has_cuda() -> bool:
    try:
        import cupy as cp

        return cp.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _supports_wmma() -> bool:
    if not _has_cuda():
        return False
    import cupy as cp

    cap = cp.cuda.Device().compute_capability
    return int(cap) >= 70


def _matmul_loop_op(*, M: int, N: int, K: int) -> LoopOp:
    i = Axis("i", M)
    j = Axis("j", N)
    k = Axis("k", K)
    return LoopOp(
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
                                    Load(name="b_v", input="b", index=(Var("k"), Var("j"))),
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


def _matmul_graph(*, M: int, N: int, K: int, out_dtype: DataType) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N), dtype=F16), node_id="b")
    g.add_node(
        op=_matmul_loop_op(M=M, N=N, K=K),
        inputs=["a", "b"],
        output=Tensor("c", (M, N), dtype=out_dtype),
        node_id="c",
    )
    g.inputs = ["a", "b"]
    g.outputs = ["c"]
    return g


def _np_dtype(dt: DataType):
    return {F16: np.float16, F32: np.float32}[dt]


# Default-on MMA matches `config.mma_enabled()` (per
# `plans/mma-fragment-factorization.md` post-M5). The pin is still set
# in tests so the fixture is robust against env-var clobbering during
# parallel pytest runs (xdist).
@pytest.mark.skipif(not _supports_wmma(), reason="WMMA needs CUDA + sm_70+")
@pytest.mark.parametrize(
    ("M", "N", "K", "out_dtype"),
    [
        (16, 16, 16, F32),
        (64, 64, 64, F32),
        (128, 128, 128, F32),
        # F16-output cases — the v1 store-bug fix lives here. The accumulator
        # type tracks the output dtype so wmma::store_matrix_sync's overload
        # resolution succeeds (no F32 acc → __half* mismatch).
        (16, 16, 16, F16),
        (64, 64, 64, F16),
        (128, 128, 128, F16),
        # Skewed shape catches asymmetric N_b × WN × FN arithmetic — both dtypes.
        (256, 256, 64, F32),
        (256, 256, 64, F16),
    ],
)
def test_mma_matmul_matches_f32_reference(M: int, N: int, K: int, out_dtype: DataType, monkeypatch):
    """An F16×F16 matmul compiled via the MMA path agrees with the f32
    reference within fp16 tolerance — for both F16 and F32 output dtypes."""
    from deplodock.compiler.ir.kernel.render import render_kernelop

    monkeypatch.setenv("DEPLODOCK_MMA", "1")

    g = _matmul_graph(M=M, N=N, K=K, out_dtype=out_dtype)
    g = Pipeline.build(KERNEL_PASSES).run(g)
    kop = g.nodes["c"].op
    assert kop.knobs.get("ATOM_KIND") == "wmma_m16n16k16_f16", "expected warp-tier MMA variant"

    tensors = {nid: n.output for nid, n in g.nodes.items() if hasattr(n.output, "shape")}
    src = render_kernelop(kop, tensors=tensors)
    assert "wmma::mma_sync" in src
    assert "#include <mma.h>" in src
    # Accumulation is ALWAYS fp32 (matches cuBLAS / PyTorch fp16-GEMM
    # precision; fp16 accumulate over a long K loses ~3-4 ulp per step).
    # The ``mma_sync`` target fragment is declared ``float``, then for a
    # narrower output buffer ``MmaStore`` downconverts via an epilogue
    # (``__float2half`` element-wise into a half ``store_matrix_sync``
    # fragment). For F32 output no downconvert is emitted.
    assert "wmma::accumulator, 16, 16, 16, float" in src, "accumulator must be fp32"
    assert ".num_elements" in src if out_dtype == F16 else True
    if out_dtype == F16:
        # Downconvert epilogue present (fp32 acc → __half store fragment).
        assert "__float2half" in src
        assert "wmma::accumulator, 16, 16, 16, half" in src  # the store-side downconvert fragment
    else:
        # F32 output stores the accumulator directly — no half fragment,
        # no downconvert loop.
        assert "__float2half" not in src
        assert "wmma::accumulator, 16, 16, 16, half" not in src

    # Launch through the real backend (``CudaBackend.run`` → ``run_program``)
    # rather than a hand-rolled ``RawModule`` call. The warp-tier picker may
    # land on a TMA-staged variant (signature ``…, const CUtensorMap* a_desc,
    # b_desc``); a 3-arg cubin launch then passes garbage descriptor pointers
    # and segfaults. The backend's ``_prebuild_descriptors`` binds every
    # ``CUtensorMap`` and opts into dynamic smem, so any geometry — TMA or
    # gmem-direct — launches correctly. The render+source asserts above still
    # cover the KERNEL-stage IR; this only fixes the execution path.
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    np.random.seed(42)
    a = (np.random.randn(M, K) * 0.1).astype(np.float16)
    b = (np.random.randn(K, N) * 0.1).astype(np.float16)
    be = CudaBackend()
    g_run = be.compile(_matmul_graph(M=M, N=N, K=K, out_dtype=out_dtype))
    run_result, _ = be.run(g_run, input_data={"a": a, "b": b})

    expected = a.astype(np.float32) @ b.astype(np.float32)
    result = run_result.outputs["c"].astype(np.float32)
    diff = np.abs(result - expected).max()
    # F32 acc: tight tolerance (≤1e-2). F16 acc: looser (~ K · max-prod-mag · f16-eps);
    # for our 0.1-scale inputs and K ≤ 256, ~5e-2 covers the worst-case drift.
    tol = 5e-2 if out_dtype == F16 else 1e-2
    assert diff < tol, f"M={M} N={N} K={K} out={out_dtype.name} max-abs-err {diff}"


@pytest.mark.skipif(not _supports_wmma(), reason="WMMA needs CUDA + sm_70+")
def test_mma_default_on_picks_warp_variant(monkeypatch):
    """With DEPLODOCK_MMA defaulted ON (config.mma_enabled defaults True),
    a no-env run of an MMA-eligible F16 matmul still emits the warp variant.
    Guards against accidental default flips and confirms priority ordering
    (warp variants outrank scalar in the fork tree)."""
    monkeypatch.delenv("DEPLODOCK_MMA", raising=False)

    g = _matmul_graph(M=64, N=64, K=64, out_dtype=F16)
    g = Pipeline.build(KERNEL_PASSES).run(g)
    kop = g.nodes["c"].op
    assert kop.knobs.get("ATOM_KIND") == "wmma_m16n16k16_f16", "MMA should be on by default"


@pytest.mark.skipif(not _supports_wmma(), reason="WMMA needs CUDA + sm_70+")
def test_mma_disabled_falls_back_to_scalar(monkeypatch):
    """When DEPLODOCK_MMA=0 is set explicitly, the planner drops the
    warp-tier branch and emits only scalar register-tile variants."""
    monkeypatch.setenv("DEPLODOCK_MMA", "0")

    g = _matmul_graph(M=64, N=64, K=64, out_dtype=F16)
    g = Pipeline.build(KERNEL_PASSES).run(g)
    kop = g.nodes["c"].op
    assert kop.knobs.get("ATOM_KIND") is None, "MMA variant should not be emitted when DEPLODOCK_MMA=0"
    # Scalar path should stamp BN/BM.
    assert "BN" in kop.knobs and "BM" in kop.knobs
