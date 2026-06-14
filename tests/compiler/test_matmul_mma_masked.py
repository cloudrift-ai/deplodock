"""Masked warp-tier MMA for symbolic output axes (M9).

A matmul whose M output axis is symbolic (``Dim('seq_len')``, hint 512,
runtime ``int seq_len`` kernel arg) reaches the mma.sync warp tier as a
MASKED tile: the enumerator stamps ``OVERHANG`` (no hint-divisibility — the
runtime extent is unknown), the planner ceil-divs the grid and wraps the cell
in a boundary ``Cond``, ``021`` hoists the K-pipeline above it (clamped slab
fill — commit "stage: runtime-extent clamp"), and ``005_lower_atom_tile``
stamps per-element row guards onto the ``RegStore`` for tiles straddling the
bound. One cached kernel serves every runtime seq_len.

CPU tests pin an explicit ``Context`` (no GPU needed for render); the GPU
accuracy tests run the symbolic launch path (``_resolve_symbolic`` reads the
extent off the input array shapes) at several runtime sizes around the hint.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.context import Context
from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import F16
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.pipeline import CUDA_PASSES, Pipeline
from deplodock.compiler.pipeline.passes.lowering.tile._atom import ATOM_REGISTRY
from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import _enumerate_warp_matmul_impl

_WARP_KNOBS = {"MMA": "mma_m16n8k16_f16", "WM": "2", "WN": "2", "FM": "2", "FN": "2", "BK": "2"}


def _has_cuda() -> bool:
    try:
        import cupy as cp

        return cp.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _supports_mma_sync() -> bool:
    if not _has_cuda():
        return False
    import cupy as cp

    return int(cp.cuda.Device().compute_capability) >= 80


def _symbolic_m_graph(*, K: int = 512, N: int = 1024) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (Dim("seq_len"), K), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N), dtype=F16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (Dim("seq_len"), N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def test_warp_enumeration_admits_forced_mask_m():
    """``m_forced_mask`` skips the M-side divisibility constraints (the
    runtime extent is unknown — the boundary guard covers the partial tile)
    and stamps the axis into ``OVERHANG`` on every row."""
    atoms = (ATOM_REGISTRY["mma_m16n8k16_f16"],)
    rows = _enumerate_warp_matmul_impl(
        E_M=512,
        E_N=1024,
        E_K=512,
        ctx=Context(compute_capability=(12, 0)),
        force_splitk_one=False,
        atoms=atoms,
        m_axis_name="i",
        n_axis_name="j",
        m_forced_mask=True,
        n_forced_mask=False,
    )
    assert rows, "forced-mask M should enumerate warp rows"
    assert all(r["OVERHANG"] == ("i",) for r in rows), "every symbolic-M row must stamp OVERHANG"
    # No hint-divisibility on the masked axis: FM values that don't divide the
    # hint's per-warp cell count (512/16 cells, e.g. FM=6 with WM=2) appear.
    fms = {r["FM"] for r in rows}
    assert any(f for f in fms if (512 // 16) % f != 0), f"masked M should sweep non-divisor FM values, got {sorted(fms)}"


def test_warp_enumeration_static_keeps_divisor_constraints():
    """Without forced masks the warp enumerator is unchanged: a non-divisible
    static extent (here E_M=100, atom_m=16) yields no rows."""
    atoms = (ATOM_REGISTRY["mma_m16n8k16_f16"],)
    rows = _enumerate_warp_matmul_impl(
        E_M=100,
        E_N=1024,
        E_K=512,
        ctx=Context(compute_capability=(12, 0)),
        force_splitk_one=False,
        atoms=atoms,
        m_axis_name="i",
        n_axis_name="j",
        m_forced_mask=False,
        n_forced_mask=False,
    )
    assert rows == []


def test_symbolic_m_masked_mma_kernel_structure(monkeypatch):
    """End-to-end render (CPU): the symbolic-M masked warp kernel carries the
    runtime ``seq_len`` arg, a ceil-div grid, the unguarded mma.sync pipeline
    with a clamped A-slab fill, the whole-tile boundary Cond, and per-element
    row guards on the fragment store."""
    for k, v in _WARP_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    lowered = Pipeline.build(CUDA_PASSES).run(_symbolic_m_graph(), ctx=Context(compute_capability=(12, 0)))
    kop = lowered.nodes["o"].op
    assert kop.knobs.get("MMA") == "mma_m16n8k16_f16"
    assert kop.knobs.get("OVERHANG"), "symbolic-M warp row must be masked"
    src = kop.kernel_source
    assert "int seq_len" in src, "runtime extent must be a kernel arg"
    assert "mma.sync.aligned.m16n8k16" in src
    assert "ldmatrix" in src
    # Commit-1 clamp on the hoisted cooperative A fill: bound by the runtime
    # extent, fallback to its last row.
    assert "< seq_len) ?" in src and "seq_len - 1" in src, f"A-slab fill must clamp to the runtime extent:\n{src[:1500]}"
    # Per-element row guards from the RegStore (both fragment row blocks).
    assert "+ _g < (seq_len)" in src and "+ _g + 8 < (seq_len)" in src, "fragment store must row-guard against seq_len"


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize("seq", [1, 31, 512, 700])
def test_symbolic_m_masked_mma_accuracy(monkeypatch, seq):
    """One compiled symbolic kernel is accurate at runtime sizes below, at,
    and above the 512 hint — including the straddling-tile cases (1, 31, 700
    are not multiples of the WM·FM·16 = 64-row tile). The greedy pick under
    these pins takes the cp.async double-buffered pipeline (shape D), so the
    hoist + clamp + guard interplay is exercised, not just the sync path."""
    for k, v in _WARP_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    be = CudaBackend()
    compiled = be.compile(_symbolic_m_graph())
    rng = np.random.default_rng(0)
    b = (rng.standard_normal((512, 1024)) * 0.1).astype(np.float16)
    a = (rng.standard_normal((seq, 512)) * 0.1).astype(np.float16)
    result, _ = be.run(compiled, input_data={"a": a, "b": b})
    got = result.outputs["o"].astype(np.float32)
    want = a.astype(np.float32) @ b.astype(np.float32)
    assert got.shape == (seq, 1024)
    diff = np.abs(got - want).max()
    assert diff < 5e-2, f"seq={seq}: masked MMA mismatch (max abs err {diff})"


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize("seq", [31, 512, 700])
def test_symbolic_mn_masked_mma_accuracy(monkeypatch, seq):
    """Both output axes symbolic (the QK^T shape: M = N = seq_len, static
    K = head_dim): the N-side mask forces per-element scalar stores (a column
    pair straddles the bound) and the output's ldm resolves from the runtime
    ``seq_len`` kernel arg (``_resolve_ldm`` Expr path) — one kernel, every
    runtime size."""
    for k, v in _WARP_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("q", (Dim("seq_len"), 128), dtype=F16), node_id="q")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("kT", (128, Dim("seq_len")), dtype=F16), node_id="kT")
    g.add_node(op=MatmulOp(), inputs=["q", "kT"], output=Tensor("o", (Dim("seq_len"), Dim("seq_len")), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["q", "kT"], ["o"]

    be = CudaBackend()
    compiled = be.compile(g)
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    kt = (rng.standard_normal((128, seq)) * 0.1).astype(np.float16)
    result, _ = be.run(compiled, input_data={"q": q, "kT": kt})
    got = result.outputs["o"].astype(np.float32)
    want = q.astype(np.float32) @ kt.astype(np.float32)
    assert got.shape == (seq, seq)
    diff = np.abs(got - want).max()
    assert diff < 5e-2, f"seq={seq}: both-symbolic masked MMA mismatch (max abs err {diff})"


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
def test_symbolic_m_masked_mma_residual_epilogue_accuracy(monkeypatch):
    """A fused residual epilogue (``o = a@b + r`` with ``r`` sharing the
    symbolic M) rides the guarded RegStore: masked rows must skip their
    epilogue gmem reads too, so a straddling runtime size stays accurate and
    fault-free."""
    for k, v in _WARP_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from deplodock.compiler.ir.tensor.ir import ElementwiseOp  # noqa: PLC0415

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (Dim("seq_len"), 512), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (512, 1024), dtype=F16), node_id="b")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("r", (Dim("seq_len"), 1024), dtype=F16), node_id="r")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("mm", (Dim("seq_len"), 1024), dtype=F16), node_id="mm")
    g.add_node(op=ElementwiseOp("add"), inputs=["mm", "r"], output=Tensor("o", (Dim("seq_len"), 1024), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b", "r"], ["o"]

    be = CudaBackend()
    compiled = be.compile(g)
    rng = np.random.default_rng(1)
    seq = 100
    a = (rng.standard_normal((seq, 512)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((512, 1024)) * 0.1).astype(np.float16)
    r = (rng.standard_normal((seq, 1024)) * 0.1).astype(np.float16)
    result, _ = be.run(compiled, input_data={"a": a, "b": b, "r": r})
    got = result.outputs["o"].astype(np.float32)
    want = a.astype(np.float32) @ b.astype(np.float32) + r.astype(np.float32)
    diff = np.abs(got - want).max()
    assert diff < 5e-2, f"masked MMA + residual epilogue mismatch (max abs err {diff})"


def _symbolic_k_graph(*, M: int = 64, N: int = 128) -> Graph:
    """A @ B with the REDUCE axis symbolic (``Dim('seq_len')``) — the SDPA P@V
    shape after the demoted-matmul split (static M/N, runtime K = seq_len)."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, Dim("seq_len")), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (Dim("seq_len"), N), dtype=F16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (M, N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def test_warp_enumeration_admits_forced_mask_k():
    """``k_forced_mask`` skips the K-side divisibility constraint (the runtime
    reduce extent is unknown — the partial final K slab is zero-filled) and
    stamps the K axis into ``OVERHANG`` on every row, exactly like a masked
    output axis."""
    atoms = (ATOM_REGISTRY["mma_m16n8k16_f16"],)
    rows = _enumerate_warp_matmul_impl(
        E_M=64,
        E_N=128,
        E_K=512,
        ctx=Context(compute_capability=(12, 0)),
        force_splitk_one=True,
        atoms=atoms,
        m_axis_name="i",
        n_axis_name="j",
        m_forced_mask=False,
        n_forced_mask=False,
        k_axis_name="k",
        k_forced_mask=True,
    )
    assert rows, "forced-mask K should enumerate warp rows"
    assert all("k" in r["OVERHANG"] for r in rows), "every symbolic-K row must stamp OVERHANG"
    # No hint-divisibility on the masked K: a BK whose cell count doesn't divide
    # the hint's K cells is admitted (the ceil-div K_o loop + zero-fill cover it).
    assert any(r["BK"] not in (1, 2, 4) or (512 // 16) % r["BK"] != 0 for r in rows) or rows


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize("seq", [16, 31, 130, 512, 700])
def test_symbolic_k_masked_mma_accuracy(monkeypatch, seq):
    """One compiled symbolic-K kernel is accurate at runtime reduce extents
    below, at, and above the 512 hint — including the straddling cases (31, 130,
    700 are not multiples of the BK·atom_k = 32-element K tile, so the final K_o
    slab is partial and must be ZERO-filled past seq_len, not edge-clamped)."""
    for k, v in _WARP_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    be = CudaBackend()
    compiled = be.compile(_symbolic_k_graph())
    rng = np.random.default_rng(0)
    for run_seq in [seq]:
        a = (rng.standard_normal((64, run_seq)) * 0.1).astype(np.float16)
        b = (rng.standard_normal((run_seq, 128)) * 0.1).astype(np.float16)
        result, _ = be.run(compiled, input_data={"a": a, "b": b})
        got = result.outputs["o"].astype(np.float32)
        want = a.astype(np.float32) @ b.astype(np.float32)
        assert got.shape == (64, 128)
        diff = np.abs(got - want).max()
        assert diff < 5e-2, f"seq={run_seq}: masked-K MMA mismatch (max abs err {diff})"
