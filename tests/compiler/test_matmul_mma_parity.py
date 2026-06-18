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
from deplodock.compiler.dtype import F16
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.pipeline import CUDA_PASSES, Pipeline

from .conftest import dyn_M, requires_sm90

# Static matmul N / K (only M flips static<->dynamic). K static so the source
# innermost dim stays static — TMA-eligible (a symbolic innermost dim would
# break the 16 B global-stride alignment; that N-masked case stays on cp.async
# and is covered by test_matmul_mma_masked.py).
_N, _K = 1024, 512
_WARP_KNOBS = {"MMA": "mma_m16n8k16_f16", "WM": "2", "WN": "2", "FM": "2", "FN": "2", "BK": "2"}


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
    """Pin the warp tile + force the transport (``TMA=1`` = cp.async.bulk.tensor,
    ``TMA=0`` = cp.async). The "pinned knobs" fixture."""
    for k, v in _WARP_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    monkeypatch.setenv("DEPLODOCK_TMA", "1" if request.param == "tma" else "0")
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
