"""Static-vs-dynamic parity for the masked-M ``mma.sync`` matmul, across transports.

One matmul op (``a[M,K] @ b[K,N]``) is compiled BOTH shape-specialised (static
``M``) and dynamic (``Dim('seq_len')`` ``M`` — one kernel for every runtime size)
and run at the SAME runtime ``M``, asserting both match the torch reference — across
the cp.async AND TMA transports (pinned). This is the parity guard the
TMA-for-dynamic work (#245) needed: the dynamic TMA path must produce the same
result as the static path on the same shape. Parity is asserted at tile-divisor M
(where all four static/dynamic × cp.async/tma combos fire); the dynamic path's
extra reach to *off-hint* sizes (1, 31, 700 — masked overhang, where static under a
pinned warp tile can't follow) is covered by ``test_matmul_mma_masked.py``.

The ``shape_mode`` × ``transport`` fixtures fan one test body out over the full
{static, dynamic} × {cp.async, tma} matrix — the fixture pattern to lift into
``conftest`` if/when other matmul tests adopt it. The pinned-knob ``transport``
fixture (warp tile + ``TMA=0|1``) is the "pinned knobs" half; the structure test
asserts the pin actually selected the intended transport (and, for dynamic, the
runtime ``seq_len`` arg) before the accuracy test trusts it.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.context import Context
from deplodock.compiler.dtype import BF16, F16
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.pipeline import CUDA_PASSES, Pipeline

from ..conftest import dyn_M, requires_sm90

# Static matmul N / K (only M flips static<->dynamic). K static so the source
# innermost dim stays static — TMA-eligible (a symbolic innermost dim would
# break the 16 B global-stride alignment; that N-masked case stays on cp.async
# and is covered by test_matmul_mma_masked.py).
_N, _K = 1024, 512
# The warp tile + operand-staging codecs (the new knob design): WM=WN=FM=FN=2, BK=2 atoms
# per inner mma step; the transport is the STAGE codec (cp.async vs tma), not a TMA bool.
_WARP_CODEC = "a:mma_m16n8k16_f16/w2xw2/f2xf2/k2"


def _has_cuda() -> bool:
    try:
        import cupy as cp

        return cp.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _supports_mma_sync() -> bool:
    """s16816 ``mma.sync.aligned.m16n8k16`` + ``ldmatrix`` need sm_80+."""
    if not _has_cuda():
        return False
    import cupy as cp

    return int(cp.cuda.Device().compute_capability) >= 80


def _supports_tma() -> bool:
    """``cp.async.bulk.tensor`` needs sm_90+ (Hopper / Blackwell)."""
    if not _has_cuda():
        return False
    import cupy as cp

    return int(cp.cuda.Device().compute_capability) >= 90


def _mma_graph(mode: str, *, M: int):
    """``a @ b`` with the M axis static (``mode='static'``) or symbolic
    (``mode='dynamic'`` → ``Dim('seq_len')``, one kernel for every runtime M)."""
    m_dim = dyn_M(mode, M)
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m_dim, _K), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (_K, _N), dtype=F16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (m_dim, _N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


@pytest.fixture(params=["cp.async", "tma"])
def transport(request, monkeypatch) -> str:
    """Pin the warp tile (``WARP`` codec) + the operand-staging transport (``STAGE`` codec
    — ``d2/cp`` = cp.async, ``d2/tma`` = cp.async.bulk.tensor). The "pinned knobs" fixture."""
    monkeypatch.setenv("DEPLODOCK_WARP", _WARP_CODEC)
    monkeypatch.setenv("DEPLODOCK_STAGE", "d2/tma" if request.param == "tma" else "d2/cp")
    return request.param


def test_pinned_transport_and_shape_fire(shape_mode, transport):
    """CPU render (forced sm_120): the pinned knobs select the intended
    transport and the symbolic M threads a runtime ``seq_len`` arg — so the
    accuracy test below is exercising the path it claims to."""
    lowered = Pipeline.build(CUDA_PASSES).run(_mma_graph(shape_mode, M=512), ctx=Context(compute_capability=(12, 0)))
    src = lowered.nodes["o"].op.kernel_source
    assert "mma.sync.aligned.m16n8k16" in src and "ldmatrix" in src, "must be on the s16816 tensor-core tier"
    if transport == "tma":
        assert "cp.async.bulk.tensor" in src, f"{shape_mode}/tma: TMA must fire"
        assert "CUtensorMap" in src, "TMA kernel must take the descriptor param"
    else:
        assert "cp.async.bulk.tensor" not in src, f"{shape_mode}/cp.async: TMA must NOT fire"
        # Positive staging assertion: the operands ride an smem slab filled by cp.async (not
        # gmem-direct) — otherwise the pin is silently ignored and the test proves nothing.
        assert "cp.async" in src, f"{shape_mode}/cp.async: operands must stage via cp.async"
        assert "__shared__" in src, f"{shape_mode}/cp.async: staged operands need an smem slab"
    if shape_mode == "dynamic":
        assert "int seq_len" in src, "dynamic kernel must carry the runtime extent arg"
    else:
        assert "int seq_len" not in src, "static kernel bakes M — no runtime extent arg"


@requires_sm90
@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize("M", [256, 512])
def test_static_dynamic_mma_parity(shape_mode, transport, M):
    """The SAME matmul, compiled static and dynamic, on cp.async and TMA, is
    accurate vs torch — so the four paths agree. ``M`` is a multiple of the
    WM·FM·16 = 64-row tile, where ALL four combos fire: static-specialised
    *non-divisor* M is degenerate under the pinned warp tile (M=1 doesn't lower)
    and static-masked-M is not TMA-eligible (an asymmetry vs the dynamic path —
    its globalDim isn't runtime-resolved), so strict parity is only well-defined
    at divisor M. Dynamic robustness at off-hint sizes (1, 31, 700 — below / at /
    above the 512 hint, masked overhang) is covered by
    ``test_matmul_mma_masked.py``."""
    if transport == "tma" and not _supports_tma():
        pytest.skip("TMA needs sm_90+ (Hopper / Blackwell)")
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    be = CudaBackend()
    compiled = be.compile(_mma_graph(shape_mode, M=M))
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((M, _K)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((_K, _N)) * 0.1).astype(np.float16)
    result, _ = be.run(compiled, input_data={"a": a, "b": b})
    got = result.outputs["o"].astype(np.float32)
    want = a.astype(np.float32) @ b.astype(np.float32)
    assert got.shape == (M, _N)
    diff = np.abs(got - want).max()
    assert diff < 5e-2, f"{shape_mode}/{transport} M={M}: max abs err {diff}"


# --- cp.async staging invariants (the landed warp-staging path) -------------
# The parity sweep above asserts the staged path is *accurate*; these two pin the
# properties that sweep can't: that staging is a pure perf transform (bit-identical
# to gmem-direct) and that the bf16 atom stages too (the cp.async byte-width path
# must handle the 2-byte operand, not just f16). Both pin the cp.async transport
# only (`STAGE=d2/cp`); TMA / scalar / depth>1 are deferred and out of scope here.


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize("M", [128, 256])
def test_staged_matches_gmem_direct_bit_for_bit(monkeypatch, M):
    """cp.async operand staging is a PURE perf transform: the staged kernel
    (``STAGE=d2/cp``) must produce **bit-identical** output to the gmem-direct
    baseline (same ``WARP`` tile, no ``STAGE``) on the same inputs — and actually
    stage (cp.async + smem slab) where the baseline does not. Guards against a
    staging-fill bug that perturbs the result, and against the pin silently
    no-op'ing to gmem-direct."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    rng = np.random.default_rng(0)
    a = (rng.standard_normal((M, _K)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((_K, _N)) * 0.1).astype(np.float16)

    def _run(stage: str | None) -> tuple[np.ndarray, str]:
        monkeypatch.setenv("DEPLODOCK_WARP", _WARP_CODEC)
        if stage:
            monkeypatch.setenv("DEPLODOCK_STAGE", stage)
        else:
            monkeypatch.delenv("DEPLODOCK_STAGE", raising=False)
        be = CudaBackend()
        compiled = be.compile(_mma_graph("static", M=M))
        src = compiled.nodes["o"].op.kernel_source
        got = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["o"])
        return got, src

    staged, staged_src = _run("d2/cp")
    gmem, gmem_src = _run(None)
    assert "cp.async" in staged_src and "__shared__" in staged_src, "STAGE=d2/cp must stage via a cp.async smem slab"
    assert "cp.async" not in gmem_src, "the gmem-direct baseline must not stage"
    np.testing.assert_array_equal(staged, gmem)  # bit-identical: staging perturbs nothing
    want = a.astype(np.float32) @ b.astype(np.float32)
    assert np.abs(staged.astype(np.float32).reshape(M, _N) - want).max() < 5e-2


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
def test_bf16_operands_stage_via_cp_async(monkeypatch):
    """The bf16 MMA atom (``mma_m16n8k16_bf16``) stages through cp.async and stays
    accurate vs torch — the cp.async byte-width fill must handle the 2-byte bf16
    operand, the same width as f16 but a distinct atom/dtype path. (No native numpy
    bf16: feed the bits as uint16 and reinterpret the uint16 output, per
    ``test_flash_tensorcore_generated``.)"""
    import torch  # noqa: PLC0415

    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    monkeypatch.setenv("DEPLODOCK_WARP", "a:mma_m16n8k16_bf16/w2xw2/f2xf2/k2")
    monkeypatch.setenv("DEPLODOCK_STAGE", "d2/cp")
    M = 256
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, _K), dtype=BF16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (_K, _N), dtype=BF16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (M, _N), dtype=BF16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    be = CudaBackend()
    compiled = be.compile(g)
    src = compiled.nodes["o"].op.kernel_source
    assert "cp.async" in src and "mma.sync.aligned.m16n8k16" in src, "bf16 operands must stage on the mma tier"
    assert "cp.async.bulk.tensor" not in src, "cp.async transport must not emit TMA"
    torch.manual_seed(0)
    qa = (torch.randn(M, _K) * 0.1).to(torch.bfloat16)
    qb = (torch.randn(_K, _N) * 0.1).to(torch.bfloat16)
    data = {"a": qa.view(torch.uint16).numpy(), "b": qb.view(torch.uint16).numpy()}
    got_bits = np.asarray(be.run(compiled, input_data=data)[0].outputs["o"]).astype(np.uint16)
    got = torch.from_numpy(got_bits).view(torch.bfloat16).float().numpy().reshape(M, _N)
    want = (qa.float() @ qb.float()).numpy()
    diff = float(np.abs(got - want).max())
    assert diff < 1e-1, f"bf16 staged mma mismatch (max abs err {diff})"
