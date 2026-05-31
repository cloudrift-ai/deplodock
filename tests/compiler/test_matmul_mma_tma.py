"""Tests for the TMA-staged WMMA path — verifies the block-aware TMA gate
landed on a `feature/tma-gate-block-aware` branch.

Pre-fix, ``050_use_tma._stage_eligible``'s inner-extent collapse used the
cache-axis extents alone, ignoring ``AffineAddressing.block``. For the
warp-tier WMMA slabs whose cache axes are warp/cell-granular (``WN`` × ``FN``
typically 4-8 elements), the collapse reported inner_extent of 4-8 elements
when the actual slab inner width was ``WN·FN·atom_n`` (64-128). At fp32-
hardcoded ``BYTES_PER_ELEM = 4``, the alignment gate ``inner_extent · 4 %
128 == 0`` always failed, so the warp tier was structurally TMA-ineligible
at every shape. Three downstream collapses (eligibility, the materializer's
box descriptor, the mbarrier ``expect_tx`` byte count) used the same broken
arithmetic and needed the same block-multiplier fix; the cuTensorMap
encoder also hardcoded ``CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 2`` (actually
``UINT32`` in the real driver enum) and silently mismatched fp16 element
width when the descriptor was used by ``cp.async.bulk.tensor``.

End-to-end bench (sm_120, RTX 5090, 2048² fp16):

    Eager PyTorch (cuBLAS):      ~99 µs   (1.00×)
    Deplodock gmem-direct:      ~108 µs   (0.92×)  ← previous baseline
    Deplodock TMA-staged:        ~92 µs   (1.08×)  ← post-fix, beats cuBLAS

The picker still defaults to gmem-direct (the scoring doesn't account for
the TMA lever yet — Phase C of plans/mma-perf-closures.md). Pinned, the
TMA WMMA path is now load-bearing.
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
# skipif reason) so they run sequentially on one worker — scattering CUDA
# tests across xdist workers exhausts the single-GPU device context. The
# per-test ``_supports_tma`` skipif still gates the sm_90+ arch requirement.
pytestmark = [requires_cuda]


def _has_cuda() -> bool:
    try:
        import cupy as cp

        return cp.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _supports_tma() -> bool:
    """TMA needs sm_90+ (Hopper, Blackwell). Both consumer (sm_120) and
    datacenter (sm_90 / sm_100) variants admit the cp.async.bulk.tensor
    transport."""
    if not _has_cuda():
        return False
    import cupy as cp

    cap = cp.cuda.Device().compute_capability
    return int(cap) >= 90


def _supports_mma_sync() -> bool:
    """The s16816 ``mma.sync.aligned.m16n8k16.f16.f16.f32`` op + ``ldmatrix``
    need sm_80+ (Ampere and later)."""
    if not _has_cuda():
        return False
    import cupy as cp

    cap = cp.cuda.Device().compute_capability
    return int(cap) >= 80


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


@pytest.fixture
def pin_tma_wmma(monkeypatch):
    """Pin the warp-tier knob set + force TMA promotion. Square WMMA atom,
    WM=WN=2 warps, FM=FN=4 register cells, BK=2 K-stage_inner trip count.
    ``DEPLODOCK_TMA=1`` forces ``050_use_tma`` to promote eligible buffered
    bundles (the block-aware eligibility check at ``_stage_eligible``).
    """
    monkeypatch.setenv("DEPLODOCK_MMA", "1")
    monkeypatch.setenv("DEPLODOCK_TMA", "1")
    monkeypatch.setenv("DEPLODOCK_ATOM_KIND", "wmma_m16n16k16_f16")
    monkeypatch.setenv("DEPLODOCK_WM", "2")
    monkeypatch.setenv("DEPLODOCK_WN", "2")
    monkeypatch.setenv("DEPLODOCK_FM", "4")
    monkeypatch.setenv("DEPLODOCK_FN", "4")
    monkeypatch.setenv("DEPLODOCK_BK", "2")


def _compile_and_render(*, M: int, N: int, K: int, out_dtype: DataType):
    from deplodock.compiler.ir.kernel.render import render_kernelop  # noqa: PLC0415

    g = _matmul_graph(M=M, N=N, K=K, out_dtype=out_dtype)
    g = Pipeline.build(KERNEL_PASSES).run(g)
    kop = g.nodes["c"].op
    tensors = {nid: n.output for nid, n in g.nodes.items() if hasattr(n.output, "shape")}
    src = render_kernelop(kop, tensors=tensors)
    return g, kop, src


@pytest.mark.skipif(not _supports_tma(), reason="TMA needs sm_90+ (Hopper / Blackwell)")
def test_default_picker_lands_on_tma_golden_at_2048_fp16(monkeypatch):
    """With ``DEPLODOCK_MMA=1`` defaulted ON and *no* warp-tier pins, the
    picker on sm_90+ must land on the empirical golden tile for 2048² fp16:
    the square ``WM=2 WN=2 FM=4 FN=4 BK=2`` (128×128 tile) with a depth-3
    ring (``BUFFER_COUNT=3``). Measured 115 µs vs the previous WM=1/WN=4
    64×256 / depth-2 pick's 133 µs (≈ 13 % faster, RTX 5090) — the deeper
    pipeline hides the TMA load latency, and the smaller square tile keeps
    2 CTA-blocks resident so the depth-3 ring still fits.

    Two co-operating priors land this: ``score_tile_geometry``'s small-BK
    pipelinability term (favours the square 128×128 whose per-stage smem
    admits a depth-3 ring at 2 blocks/SM) and ``040_use_ring_buffers``'s
    occupancy-ordered BUFFER_COUNT variants (front-loads the deepest depth
    that keeps 2 blocks for the greedy default).
    """
    monkeypatch.setenv("DEPLODOCK_MMA", "1")
    _, kop, _ = _compile_and_render(M=2048, N=2048, K=2048, out_dtype=F16)
    assert kop.knobs.get("ATOM_KIND") == "wmma_m16n16k16_f16"
    assert int(kop.knobs.get("WM", 0)) == 2, f"expected WM=2, got {kop.knobs.get('WM')}"
    assert int(kop.knobs.get("WN", 0)) == 2, f"expected WN=2, got {kop.knobs.get('WN')}"
    assert int(kop.knobs.get("FM", 0)) == 4, f"expected FM=4, got {kop.knobs.get('FM')}"
    assert int(kop.knobs.get("FN", 0)) == 4, f"expected FN=4, got {kop.knobs.get('FN')}"
    assert int(kop.knobs.get("BK", 0)) == 2, f"expected BK=2, got {kop.knobs.get('BK')}"
    assert int(kop.knobs.get("BUFFER_COUNT", 0)) == 3, f"expected BUFFER_COUNT=3, got {kop.knobs.get('BUFFER_COUNT')}"


@pytest.mark.skipif(not _supports_tma(), reason="TMA needs sm_90+ (Hopper / Blackwell)")
def test_tma_wmma_path_emits_bulk_tensor_ptx(pin_tma_wmma):
    """At a shape large enough to clear both the 128-byte alignment gate
    and the ``src_inner ≥ 2 × inner_extent`` size gate, the WMMA staged
    bundle promotes to TMA and the rendered C source contains the
    ``cp.async.bulk.tensor`` PTX path (not cp.async)."""
    _, _, src = _compile_and_render(M=256, N=256, K=128, out_dtype=F32)
    assert "cp.async.bulk.tensor" in src, "TMA bundle must lower to cp.async.bulk.tensor PTX"
    assert "mbarrier.arrive.expect_tx" in src, "TMA mbarrier handshake must be present"
    # The legacy cp.async path's commit/wait_group instructions must NOT
    # appear — a regression where the bundle stays on cp.async despite
    # eligibility passing would silently lose the TMA perf.
    assert "cp.async.commit_group" not in src
    assert "cp.async.wait_group" not in src


@pytest.mark.skipif(not _supports_tma(), reason="TMA needs sm_90+ (Hopper / Blackwell)")
@pytest.mark.parametrize("M,N,K", [(256, 256, 128), (512, 512, 128), (256, 256, 64)])
def test_tma_wmma_matches_f32_reference(pin_tma_wmma, M: int, N: int, K: int):
    """The TMA-staged WMMA path produces output that matches the f32
    reference within fp16 tolerance.

    The fp16 cuTensorMap encoder enum fix (``CU_TENSOR_MAP_DATA_TYPE_FLOAT16
    = 6`` mapped via ``_DTYPE_BY_ITEMSIZE``) is exercised here — pre-fix
    the descriptor hardcoded ``FLOAT32 = 2`` (actually UINT32 in the
    driver enum), and the TMA hardware misinterpreted fp16 byte strides
    so the per-K_o copy delivered the wrong slot's data.

    ``256×256×64`` pins the exact square-warp geometry the default picker
    lands on at that skewed shape (WM=WN=2, FM=FN=4, BK=2 → 128×128 tile,
    K_o=2). A 3-arg ``RawModule`` launch of this TMA-staged kernel passes
    garbage descriptor pointers and segfaults — the regression that motivated
    routing the launch through the backend, which binds every ``CUtensorMap``
    via ``_prebuild_descriptors``."""
    _, kop, src = _compile_and_render(M=M, N=N, K=K, out_dtype=F32)
    assert kop.knobs.get("ATOM_KIND") == "wmma_m16n16k16_f16"
    assert "cp.async.bulk.tensor" in src  # smoke-check the TMA path actually fired

    # Launch through the real backend so ``_prebuild_descriptors`` binds the
    # per-operand ``CUtensorMap`` args (a hand-rolled cubin launch can't) and
    # opts into the dynamic-smem allowance. Recompile from a fresh loop graph
    # via the full ``CUDA_PASSES`` pipeline under the same pinned env.
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    np.random.seed(42)
    a = (np.random.randn(M, K) * 0.1).astype(np.float16)
    b = (np.random.randn(K, N) * 0.1).astype(np.float16)
    be = CudaBackend()
    g_run = be.compile(_matmul_graph(M=M, N=N, K=K, out_dtype=F32))
    run_result, _ = be.run(g_run, input_data={"a": a, "b": b})

    expected = a.astype(np.float32) @ b.astype(np.float32)
    result = run_result.outputs["c"].astype(np.float32)
    diff = np.abs(result - expected).max()
    assert diff < 1e-2, f"M={M} N={N} K={K} max-abs-err {diff}"


# --- mma.sync (s16816) path: ldmatrix + mma.sync.aligned.m16n8k16 -----------


@pytest.fixture
def pin_mma_sync(monkeypatch):
    """Pin the modern warp-level MMA atom + a staged 128×128 warp tile.

    ``DEPLODOCK_ATOM_KIND=mma_m16n8k16_f16`` opts the s16816 path into the
    enumeration (it's appended to ``_ATOM_KINDS_V1`` but not auto-selected
    until perf-promoted). ``WM=2 FM=4`` → M-tile 128 (``2·4·atom_m=16``);
    ``WN=2 FN=8`` → N-tile 128 (``2·8·atom_n=8``); ``BK=2`` → 32-element
    K-stage. ``DEPLODOCK_TMA=1`` lets the staged bundle promote to TMA on
    sm_90+ (ldmatrix still reads from the staged smem slab either way)."""
    monkeypatch.setenv("DEPLODOCK_MMA", "1")
    monkeypatch.setenv("DEPLODOCK_TMA", "1")
    monkeypatch.setenv("DEPLODOCK_ATOM_KIND", "mma_m16n8k16_f16")
    monkeypatch.setenv("DEPLODOCK_WM", "2")
    monkeypatch.setenv("DEPLODOCK_WN", "2")
    monkeypatch.setenv("DEPLODOCK_FM", "4")
    monkeypatch.setenv("DEPLODOCK_FN", "8")
    monkeypatch.setenv("DEPLODOCK_BK", "2")


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs sm_80+ (Ampere+)")
def test_mma_sync_path_emits_ldmatrix_and_mma_ptx(pin_mma_sync):
    """The pinned mma.sync atom lowers to ``ldmatrix`` + ``mma.sync.aligned``
    inline PTX (the s16816 path) and emits zero ``nvcuda::wmma`` intrinsics —
    a clean swap of the tensor-core instruction family, not a mix."""
    _, kop, src = _compile_and_render(M=256, N=256, K=128, out_dtype=F16)
    assert kop.knobs.get("ATOM_KIND") == "mma_m16n8k16_f16"
    assert "ldmatrix.sync.aligned" in src, "mma.sync operands must load via ldmatrix"
    assert "mma.sync.aligned.m16n8k16" in src, "the s16816 mma.sync instruction must be emitted"
    assert "wmma::" not in src, "the mma.sync path must not mix in legacy wmma intrinsics"
    # f32 accumulate → f16 output goes through the per-lane __float2half epilogue.
    assert "__float2half" in src, "f16 output needs the fp32→fp16 downconvert epilogue"


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs sm_80+ (Ampere+)")
@pytest.mark.parametrize("M,N,K,out_dtype", [(256, 256, 128, F32), (128, 256, 128, F32), (256, 256, 128, F16)])
def test_mma_sync_matches_reference(pin_mma_sync, M: int, N: int, K: int, out_dtype: DataType):
    """The s16816 path matches the f32 reference within fp16 tolerance —
    guards the ldmatrix per-lane address map, the B-operand ``.trans``, and
    the per-lane ``RegStore`` epilogue (a wrong lane map / trans flag yields
    a plausible-but-wrong result that this catches). f32 output exercises the
    direct ``RegStore`` path; f16 output exercises the ``__float2half``
    downconvert."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    _, kop, src = _compile_and_render(M=M, N=N, K=K, out_dtype=out_dtype)
    assert kop.knobs.get("ATOM_KIND") == "mma_m16n8k16_f16"
    assert "mma.sync.aligned.m16n8k16" in src

    np.random.seed(7)
    a = (np.random.randn(M, K) * 0.1).astype(np.float16)
    b = (np.random.randn(K, N) * 0.1).astype(np.float16)
    be = CudaBackend()
    g_run = be.compile(_matmul_graph(M=M, N=N, K=K, out_dtype=out_dtype))
    run_result, _ = be.run(g_run, input_data={"a": a, "b": b})

    expected = a.astype(np.float32) @ b.astype(np.float32)
    result = run_result.outputs["c"].astype(np.float32)
    diff = np.abs(result - expected).max()
    assert diff < 1e-2, f"M={M} N={N} K={K} out={out_dtype.name} max-abs-err {diff}"
