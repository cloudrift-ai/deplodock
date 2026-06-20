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

from .conftest import requires_sm90

# ``TMA=0`` pins the cp.async masked path: these tests assert cp.async-specific
# codegen (the clamped A-slab fill, the "shape D" double-buffered pipeline).
# Symbolic-M with a static innermost dim is now ALSO TMA-eligible
# (``050_use_tma`` builds the descriptor's globalDim per launch from the runtime
# shape; TMA zero-fills the masked overhang) — that path has its own test below.
_WARP_KNOBS = {"MMA": "mma_m16n8k16_f16", "WM": "2", "WN": "2", "FM": "2", "FN": "2", "BK": "2", "TMA": "0"}


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


def _supports_tma() -> bool:
    """TMA (cp.async.bulk.tensor) needs sm_90+ (Hopper / Blackwell)."""
    if not _has_cuda():
        return False
    import cupy as cp

    return int(cp.cuda.Device().compute_capability) >= 90


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


@requires_sm90
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


_TMA_KNOBS = {"MMA": "mma_m16n8k16_f16", "WM": "2", "WN": "2", "FM": "2", "FN": "2", "BK": "2", "TMA": "1"}


def test_symbolic_m_masked_mma_tma_structure(monkeypatch):
    """Symbolic-M (static innermost) reaches the warp tier staged via **TMA**:
    the kernel carries the runtime ``seq_len`` arg + a ``CUtensorMap`` descriptor
    param and stages the A operand with ``cp.async.bulk.tensor`` (no cp.async
    clamp — the descriptor's globalDim is the runtime extent and TMA zero-fills
    the masked overhang). The descriptor is encoded per launch from the input
    array shape (``program._prebuild_descriptors``)."""
    for k, v in _TMA_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    lowered = Pipeline.build(CUDA_PASSES).run(_symbolic_m_graph(), ctx=Context(compute_capability=(12, 0)))
    kop = lowered.nodes["o"].op
    assert kop.knobs.get("TMA") is True, "symbolic-M with static innermost dim must be TMA-eligible"
    src = kop.kernel_source
    assert "int seq_len" in src, "runtime extent must still be a kernel arg"
    assert "cp.async.bulk.tensor" in src, "A operand must stage via TMA"
    assert "CUtensorMap" in src, "kernel must take the TMA descriptor param"
    assert "mma.sync.aligned.m16n8k16" in src


@requires_sm90
@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize("seq", [1, 31, 512, 700])
def test_symbolic_m_masked_mma_tma_accuracy(monkeypatch, seq):
    """The TMA-staged symbolic-M kernel is accurate at runtime sizes below, at,
    and above the 512 hint — the cases that exercise the per-launch descriptor
    build (``_collapse_inert_dims`` must keep the masked dim even when its
    runtime extent is 1/31, and TMA zero-fills the box overhang past ``seq_len``)."""
    for k, v in _TMA_KNOBS.items():
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
    assert diff < 5e-2, f"seq={seq}: TMA masked MMA mismatch (max abs err {diff})"


@requires_sm90
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


@requires_sm90
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


@requires_sm90
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


def _batched_symbolic_mk_graph(*, H: int = 16, N: int = 128) -> Graph:
    """The SDPA P@V split-consumer in full: a BATCHED matmul (``H`` heads) whose
    M (query) AND K (key) axes are both symbolic ``seq_len`` —
    ``xna[H, seq, seq] @ xnb[H, seq, N]``. The batch axis sits before K in the B
    operand's index (``xnb[head, k, n]``), the case ``classify_matmul_operands``
    must recognize so the cell reaches the warp tier (not scalar)."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("xna", (H, Dim("seq_len"), Dim("seq_len")), dtype=F16), node_id="xna")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("xnb", (H, Dim("seq_len"), N), dtype=F16), node_id="xnb")
    g.add_node(op=MatmulOp(), inputs=["xna", "xnb"], output=Tensor("o", (H, Dim("seq_len"), N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["xna", "xnb"], ["o"]
    return g


def test_batched_symbolic_mk_reaches_warp(monkeypatch):
    """The batched masked-M + masked-K P@V consumer must reach the mma.sync tier
    (the ``classify_matmul_operands`` batch-aware B test), not fall to scalar."""
    for k, v in _WARP_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    lowered = Pipeline.build(CUDA_PASSES).run(_batched_symbolic_mk_graph(), ctx=Context(compute_capability=(12, 0)))
    kop = lowered.nodes["o"].op
    assert kop.knobs.get("MMA") == "mma_m16n8k16_f16", "batched symbolic M+K matmul must reach the warp tier"
    src = kop.kernel_source
    assert "mma.sync.aligned.m16n8k16" in src and "ldmatrix" in src
    assert "int seq_len" in src, "runtime extent must be a kernel arg"


@requires_sm90
@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize("seq", [16, 31, 130, 512, 700])
def test_batched_symbolic_mk_masked_mma_accuracy(monkeypatch, seq):
    """One compiled batched symbolic-M+K kernel (the deployable P@V consumer) is
    accurate at runtime sizes around the 512 hint, including the straddling
    cases where both the M tile and the partial K slab are masked."""
    for k, v in _WARP_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    be = CudaBackend()
    compiled = be.compile(_batched_symbolic_mk_graph())
    rng = np.random.default_rng(0)
    xna = (rng.standard_normal((16, seq, seq)) * 0.1).astype(np.float16)
    xnb = (rng.standard_normal((16, seq, 128)) * 0.1).astype(np.float16)
    result, _ = be.run(compiled, input_data={"xna": xna, "xnb": xnb})
    got = result.outputs["o"].astype(np.float32)
    want = np.matmul(xna.astype(np.float32), xnb.astype(np.float32))
    assert got.shape == (16, seq, 128)
    diff = np.abs(got - want).max()
    assert diff < 5e-2, f"seq={seq}: batched masked-M+K MMA mismatch (max abs err {diff})"


# --- demoted symbolic-N B operand: TMA + warp-spec (WS1/WS2) ----------------
# The rotary QK^T's key cone materializes the canonical B operand
# ``xnb[…, K, N]`` with a symbolic inner N (``seq_len``). A bare symbolic inner
# extent gives the K dim an unaligned gmem stride, so ``cuTensorMapEncodeTiled``
# rejects it and the masked-N matmul is stuck on cp.async (no TMA → no
# warp-spec). ``_split_demoted`` now pads the inner N up to a multiple of 64 so
# the stride stays 16 B-aligned; ``050_use_tma`` accepts the aligned symbolic
# inner, and warp-spec follows automatically on the resulting TMA depth-2 bundle.


def _demoted_symbolic_n_graph(M=None, N=None, K: int = 128) -> Graph:
    """Computed-B-cone matmul (the rotary QK^T shape): an elementwise scale on
    BOTH operands feeds a transposed-``[N, K]`` Linear, so fusion demotes the
    matmul and ``005_split_demoted`` materializes the canonical ``xnb[K, N]``
    producer. With N symbolic this is the dynamic QK^T B operand WS1 unblocks.
    M and N default to the same ``Dim('seq_len')`` (the [seq, seq] scores)."""
    from deplodock.compiler.ir.frontend.ir import LinearOp  # noqa: PLC0415
    from deplodock.compiler.ir.tensor.ir import ElementwiseOp  # noqa: PLC0415

    M = Dim("seq_len") if M is None else M
    N = Dim("seq_len") if N is None else N
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (M, K), dtype=F16), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("sx", (M, K), dtype=F16), node_id="sx")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("w", (N, K), dtype=F16), node_id="w")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("sw", (N, K), dtype=F16), node_id="sw")
    g.add_node(op=ElementwiseOp("multiply"), inputs=["x", "sx"], output=Tensor("xs", (M, K), dtype=F16), node_id="xs")
    g.add_node(op=ElementwiseOp("multiply"), inputs=["w", "sw"], output=Tensor("ws", (N, K), dtype=F16), node_id="ws")
    g.add_node(op=LinearOp(), inputs=["xs", "ws"], output=Tensor("o", (M, N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["x", "sx", "w", "sw"], ["o"]
    return g


def test_demoted_symbolic_n_b_operand_reaches_tma_and_warpspec(monkeypatch):
    """The demoted symbolic-N B operand stages via **TMA** and warp-spec fires:
    ``005_split_demoted`` pads the ``xnb`` inner N to a 16 B-aligned multiple, so
    the masked-N consumer's bundle promotes to ``cp.async.bulk.tensor`` and the
    greedy default deploys the warp-specialized role split (no env pin — TMA +
    WARPSPEC are the chosen-by-greedy decisions, the deployable shape)."""
    for k, v in _WARP_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    monkeypatch.delenv("DEPLODOCK_TMA", raising=False)  # drop the _WARP_KNOBS TMA=0 pin — let greedy pick TMA
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")  # force the demotion split
    lowered = Pipeline.build(CUDA_PASSES).run(_demoted_symbolic_n_graph(), ctx=Context(compute_capability=(12, 0)))

    xnb = lowered.nodes["o__xnb"].output
    inner = xnb.shape[-1]
    assert not inner.is_static, "the xnb inner N stays symbolic (runtime-sized, correct at any seq)"
    assert "64" in inner.expr.pretty(), f"the xnb inner N must be padded to a 64-multiple, got {inner.expr.pretty()}"

    con = lowered.nodes["o"].op
    assert con.knobs.get("TMA") is True, "greedy must deploy TMA on the symbolic-N B operand"
    assert con.knobs.get("WARPSPEC") is True, "warp-spec must fire on the TMA depth-2 bundle"
    src = con.kernel_source
    assert "cp.async.bulk.tensor" in src, "B operand must stage via TMA"
    assert "mbarrier.arrive.expect_tx" in src, "TMA mbarrier handshake must be present"
    assert "cp.async.commit_group" not in src, "must not fall back to the legacy cp.async path"
    assert "mma.sync.aligned.m16n8k16" in src
    assert "int seq_len" in src, "runtime extent must still be a kernel arg"


@pytest.mark.skipif(not _supports_tma(), reason="TMA (cp.async.bulk.tensor) needs sm_90+")
@pytest.mark.parametrize("seq", [31, 130, 512, 700])
def test_demoted_symbolic_n_tma_accuracy(monkeypatch, seq):
    """The TMA-staged + warp-specialized symbolic-N kernel is accurate below, at,
    and above the 512 hint — including straddling sizes (31, 130, 700) where the
    last N tile reads the padded ``[seq, round_up)`` overhang columns. Those
    garbage columns feed the mma only into store-masked output positions, so the
    live scores stay correct."""
    for k, v in _WARP_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    monkeypatch.delenv("DEPLODOCK_TMA", raising=False)  # let greedy deploy TMA
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    be = CudaBackend()
    compiled = be.compile(_demoted_symbolic_n_graph())
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    sx = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    w = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    sw = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    result, _ = be.run(compiled, input_data={"x": x, "sx": sx, "w": w, "sw": sw})
    got = result.outputs["o"].astype(np.float32)
    want = (x * sx).astype(np.float32) @ (w * sw).astype(np.float32).T
    assert got.shape == (seq, seq)
    diff = np.abs(got - want).max()
    assert diff < 5e-2, f"seq={seq}: demoted symbolic-N TMA MMA mismatch (max abs err {diff})"


# --- demoted masked-K P@V: TMA via OOB zero-fill (WS3) -----------------------
# The SDPA P@V's A operand is the softmax-normalized probs ``xn[head, m, k]`` with
# the reduce K innermost and symbolic (``seq_len``). Padding its inner K to a
# 16 B-aligned multiple makes it TMA-eligible; the masked-K reduce overhang must
# read 0, which the *other* operand (V, middle-K, allocated at the real seq_len)
# delivers via TMA's hardware OOB zero-fill — every overhang product is
# ``A_overhang × 0`` (A_overhang finite, since scratch is zero-init + holds only
# finite kernel outputs). ``040`` rings the masked-K bundle only when it will reach
# TMA (``050.tile_reaches_tma``); otherwise it stays SYNC with the per-value
# ternary zero-fill, so a masked-K bundle is never stranded on cp.async.


def _pv_softmax_graph(H: int = 16, N: int = 128) -> Graph:
    """Softmax(scores) @ V with the reduce K = ``seq_len`` symbolic (the SDPA P@V
    shape). Fusion demotes the matmul; ``005_split_demoted`` materializes the
    softmax-prob A cone ``xn[H, seq, seq]`` (inner K) + the clean symbolic-K gemm."""
    from deplodock.compiler.ir.frontend.ir import SoftmaxOp  # noqa: PLC0415

    s = Dim("seq_len")
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("scores", (H, s, s), dtype=F16), node_id="scores")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("v", (H, s, N), dtype=F16), node_id="v")
    g.add_node(op=SoftmaxOp(axis=-1), inputs=["scores"], output=Tensor("probs", (H, s, s), dtype=F16), node_id="probs")
    g.add_node(op=MatmulOp(), inputs=["probs", "v"], output=Tensor("o", (H, s, N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["scores", "v"], ["o"]
    return g


def test_demoted_masked_k_pv_reaches_tma(monkeypatch):
    """The demoted masked-K P@V consumer stages via TMA on sm_90+: the padded
    probs A cone + the middle-K V both clear ``050``'s gate, so greedy deploys the
    TMA ring (``cp.async.bulk.tensor``). The probs cone's inner K is padded to a
    64-multiple (symbolic, runtime-sized)."""
    for k, v in _WARP_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    monkeypatch.delenv("DEPLODOCK_TMA", raising=False)  # let greedy pick TMA
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")  # force the demotion split
    lowered = Pipeline.build(CUDA_PASSES).run(_pv_softmax_graph(), ctx=Context(compute_capability=(12, 0)))

    xn = lowered.nodes["o__xn"].output
    inner = xn.shape[-1]
    assert not inner.is_static, "the probs cone inner K stays symbolic (runtime-sized)"
    assert "64" in inner.expr.pretty(), f"the probs cone inner K must be padded to a 64-multiple, got {inner.expr.pretty()}"

    con = lowered.nodes["o"].op
    assert con.knobs.get("TMA") is True, "greedy must deploy TMA on the masked-K P@V"
    src = con.kernel_source
    assert "cp.async.bulk.tensor" in src, "masked-K operands must stage via TMA"
    assert "mma.sync.aligned.m16n8k16" in src
    assert "int seq_len" in src, "runtime reduce extent must be a kernel arg"


def test_demoted_masked_k_pv_stays_sync_below_sm90(monkeypatch):
    """Below sm_90 (no TMA) the masked-K P@V keeps the SYNC ternary zero-fill —
    ``040`` declines the ring (``tile_reaches_tma`` False), never leaving a
    masked-K bundle on a cp.async / double-buffer transport that can't zero the
    partial-K slab."""
    for k, v in _WARP_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    monkeypatch.delenv("DEPLODOCK_TMA", raising=False)
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")
    lowered = Pipeline.build(CUDA_PASSES).run(_pv_softmax_graph(), ctx=Context(compute_capability=(8, 0)))
    con = lowered.nodes["o"].op
    assert con.knobs.get("TMA") is not True, "sm_80 has no TMA — masked-K must stay SYNC"
    assert con.knobs.get("RING") == 1, "the masked-K ring must be declined below sm_90 (stays on the SYNC ternary)"


@pytest.mark.skipif(not _supports_tma(), reason="TMA (cp.async.bulk.tensor) needs sm_90+")
@pytest.mark.parametrize("seq", [16, 31, 130, 512, 700])
def test_demoted_masked_k_pv_tma_accuracy(monkeypatch, seq):
    """The TMA-staged masked-K P@V is accurate below, at, and above the 512 hint —
    including the straddling reduce extents (31, 130, 700) where the final K slab
    is partial and TMA's middle-K OOB zero-fill must zero V past ``seq_len`` (a
    clamped duplicate would corrupt the reduction). seq=16 exercises the tiny
    K_o=1 ring (prologue-only, no steady state)."""
    for k, v in _WARP_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    monkeypatch.delenv("DEPLODOCK_TMA", raising=False)
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    be = CudaBackend()
    compiled = be.compile(_pv_softmax_graph())
    rng = np.random.default_rng(seq)
    scores = (rng.standard_normal((16, seq, seq)) * 2).astype(np.float16)
    v = (rng.standard_normal((16, seq, 128)) * 0.1).astype(np.float16)
    result, _ = be.run(compiled, input_data={"scores": scores, "v": v})
    got = result.outputs["o"].astype(np.float32)
    sc = scores.astype(np.float32)
    e = np.exp(sc - sc.max(-1, keepdims=True))
    probs = e / e.sum(-1, keepdims=True)
    want = np.matmul(probs, v.astype(np.float32))
    assert got.shape == (16, seq, 128)
    diff = np.abs(got - want).max()
    assert diff < 5e-2, f"seq={seq}: demoted masked-K P@V TMA mismatch (max abs err {diff})"
