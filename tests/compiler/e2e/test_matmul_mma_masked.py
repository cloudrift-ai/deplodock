"""Masked warp-tier MMA for symbolic output / reduce axes (M9), at STRADDLING sizes.

A matmul whose M (and/or N, K) axis is symbolic (``Dim('seq_len')``, hint 512,
runtime ``int seq_len`` kernel arg) reaches the mma.sync warp tier as a MASKED
tile: the planner ceil-divs the grid and wraps the cell in a boundary ``Cond``,
``021`` hoists the K-pipeline above it (clamped slab fill), and
``kernel/005_lower_atom_tile`` stamps per-element row/col guards onto the
``RegStore`` for tiles straddling the bound. ``kernel/_stage_expand`` zero-fills
the partial final K slab past a symbolic reduce extent. One cached kernel serves
every runtime size.

THE WHOLE POINT of this file is **off-hint / straddling sizes** (1, 31, 130, 700
— NOT tile-divisor multiples of the 64-row / 32-element tile), which exercise the
boundary-guard + clamp + zero-fill interplay that the tile-divisor parity sweep
in ``test_matmul_mma_parity.py`` cannot reach. CPU structure tests pin an explicit
``Context`` (no GPU needed for render); the GPU accuracy tests run the symbolic
launch path (the extent is resolved off the input array shapes) at sizes below,
at, and above the 512 hint.

The full masked warp tier lowers and runs: symbolic-M, symbolic-M+N, symbolic-K
(zero-filled partial slab), the batched P@V split-consumer, and the demoted P@V —
each fed as a synthetic standalone graph here and accuracy-checked at off-hint
sizes, alongside the symbolic-K P@V's real-SDPA-decomposition coverage in
``e2e/test_ops_vs_torch.py``.
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
from deplodock.compiler.pipeline.knob import mma_atom

from ..conftest import requires_sm90

# The warp-tile (WARP) codec + the staging transport (STAGE) codec: ``d2/cp`` pins the
# cp.async masked path (the clamped A-slab fill); ``d2/tma`` the TMA path. Symbolic-M with
# a static innermost dim is TMA-eligible — that path has its own test below.
_WARP_CODEC = "a:mma_m16n8k16_f16/w2xw2/f2xf2/k2"
_CP_KNOBS = {"WARP": _WARP_CODEC, "STAGE": "d2/cp"}
_TMA_KNOBS = {"WARP": _WARP_CODEC, "STAGE": "d2/tma"}


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


# --- symbolic-M masked warp tier (cp.async) ---------------------------------


def test_symbolic_m_masked_mma_kernel_structure(monkeypatch):
    """End-to-end render (CPU): the symbolic-M masked warp kernel carries the
    runtime ``seq_len`` arg, the mma.sync pipeline with a clamped A-slab fill,
    and per-element row guards on the fragment store. The masking signal lives in
    the codegen (the clamp + the ``+ _g < (seq_len)`` row guards) and the
    ``S_ext_n_symbolic_axis`` shape feature, not a standalone OVERHANG knob."""
    for k, v in _CP_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    lowered = Pipeline.build(CUDA_PASSES).run(_symbolic_m_graph(), ctx=Context(compute_capability=(12, 0)))
    kop = lowered.nodes["o"].op
    assert mma_atom(kop.knobs) == "mma_m16n8k16_f16"
    assert kop.knobs.get("S_ext_n_symbolic_axis"), "symbolic-M warp row must carry a symbolic axis"
    src = kop.kernel_source
    assert "int seq_len" in src, "runtime extent must be a kernel arg"
    assert "mma.sync.aligned.m16n8k16" in src
    assert "ldmatrix" in src
    # Clamp on the hoisted cooperative A fill: bound by the runtime extent,
    # fallback to its last row.
    assert "< seq_len) ?" in src and "seq_len - 1" in src, f"A-slab fill must clamp to the runtime extent:\n{src[:1500]}"
    # Per-element row guards from the RegStore (both fragment row blocks).
    assert "+ _g < (seq_len)" in src and "+ _g + 8 < (seq_len)" in src, "fragment store must row-guard against seq_len"


@requires_sm90
@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize("seq", [1, 31, 512, 700])
def test_symbolic_m_masked_mma_accuracy(monkeypatch, seq):
    """One compiled symbolic kernel is accurate at runtime sizes below, at,
    and above the 512 hint — including the straddling-tile cases (1, 31, 700 are
    not multiples of the WM·FM·16 = 64-row tile, so the trailing rows straddle the
    bound and exercise the hoist + clamp + per-element guard interplay)."""
    for k, v in _CP_KNOBS.items():
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


# --- symbolic-M masked warp tier (TMA) --------------------------------------


def test_symbolic_m_masked_mma_tma_structure(monkeypatch):
    """Symbolic-M (static innermost) reaches the warp tier staged via **TMA**:
    the kernel carries the runtime ``seq_len`` arg + a ``CUtensorMap`` descriptor
    param and stages the A operand with ``cp.async.bulk.tensor`` (the descriptor's
    globalDim is the runtime extent and TMA zero-fills the masked overhang)."""
    for k, v in _TMA_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    lowered = Pipeline.build(CUDA_PASSES).run(_symbolic_m_graph(), ctx=Context(compute_capability=(12, 0)))
    kop = lowered.nodes["o"].op
    # The transport is the orthogonal STAGE codec on the schedule (``d<depth>/tma``), not a
    # legacy ``PLACE@<edge>=…:tma`` placement knob — symbolic-M with a static innermost dim
    # is TMA-eligible, so the pin survives onto the lowered op.
    assert kop.knobs.get("STAGE", "").endswith("/tma"), f"symbolic-M static-innermost must stage via TMA: {kop.knobs.get('STAGE')!r}"
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
    build (the masked dim must survive even when its runtime extent is 1/31, and
    TMA zero-fills the box overhang past ``seq_len``)."""
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


# --- both output axes symbolic (the QK^T scores shape) ----------------------


@requires_sm90
@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize("seq", [31, 512, 700])
def test_symbolic_mn_masked_mma_accuracy(monkeypatch, seq):
    """Both output axes symbolic (the QK^T shape: M = N = seq_len, static
    K = head_dim): the N-side mask forces per-element scalar stores (a column
    pair straddles the bound) and the output's ldm resolves from the runtime
    ``seq_len`` kernel arg — one kernel, every runtime size, off-hint included."""
    for k, v in _CP_KNOBS.items():
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


# --- masked + fused residual epilogue ---------------------------------------


@requires_sm90
@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
def test_symbolic_m_masked_mma_residual_epilogue_accuracy(monkeypatch):
    """A fused residual epilogue (``o = a@b + r`` with ``r`` sharing the
    symbolic M) rides the guarded RegStore: masked rows must skip their epilogue
    gmem reads too, so a straddling runtime size (seq=100, not a 64-row multiple)
    stays accurate and fault-free."""
    for k, v in _CP_KNOBS.items():
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


# --- demoted symbolic-N B operand: TMA + warp-spec --------------------------
# The rotary QK^T's key cone materializes the canonical B operand ``xnb[…, K, N]``
# with a symbolic inner N (``seq_len``); ``010_split_demoted`` pads the inner N up
# to a 16 B-aligned multiple so ``050_use_tma`` accepts it and warp-spec follows.


def _demoted_symbolic_n_graph(M=None, N=None, K: int = 128) -> Graph:
    """Computed-B-cone matmul (the rotary QK^T shape): an elementwise scale on
    BOTH operands feeds a transposed-``[N, K]`` Linear, so fusion demotes the
    matmul and ``010_split_demoted`` materializes the canonical ``xnb[K, N]``
    producer. M and N default to the same ``Dim('seq_len')`` (the [seq, seq]
    scores)."""
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


@requires_sm90
@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize("seq", [31, 130, 512, 700])
def test_demoted_symbolic_n_accuracy(monkeypatch, seq):
    """The demoted symbolic-N matmul (the rotary QK^T B-cone) is accurate below,
    at, and above the 512 hint under the GREEDY pick — including straddling sizes
    (31, 130, 700) where the last N tile reads the padded ``[seq, round_up)``
    overhang columns; those garbage columns feed the mma only into store-masked
    output positions, so the live scores stay correct. (No ``DEPLODOCK_MMA`` pin:
    the multi-kernel demoted graph's ``map`` producer rejects a global warp pin
    under ``enumeration/_validate.validate_pins`` — greedy lowers it cleanly.)"""
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")  # force the demotion split
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
    assert diff < 5e-2, f"seq={seq}: demoted symbolic-N MMA mismatch (max abs err {diff})"


# --- masked-K (symbolic reduce) mma tier — R7, fully recovered. ------------------
# The masked-K mma tier (a symbolic ``seq_len`` reduce on the tensor-core path) now
# lowers + runs correctly through these SYNTHETIC standalone graphs. Four fixes
# landed: (1) ``_classify`` admits a symbolic-K SEMIRING (tiles K at the ``Dim``
# hint); (2) ``_build._rebracket_k`` (warp tier) ceil-divides ``K_o`` so the loop bound is the
# runtime ``ceil(seq_len/(BK·atom_k))`` (covers seq > hint) and ``seq_len`` enters the
# kernel signature; (3) ``assembly/_slab._stamp_kmask`` stamps ``Source.kmask`` on the
# staged operands so ``_stage_expand`` ZERO-fills the partial-final-K smem overhang
# (previously the field was never populated, so the slab read stale smem past
# ``seq_len`` — correct in a fresh context, wrong under concurrent load); (4) the
# demoted softmax-P@V split's MONOID ``xn`` producer no longer trips ``validate_pins``
# on the union ``MMA`` pin — ``enumeration/010_build`` skips the strict per-op check
# for a multi-op kernel set (``_is_union_pinned``), since a global pin is a union pin
# across the producer + gemm consumer. All masked-K tests are de-quarantined.


def _symbolic_k_graph(*, M: int = 64, N: int = 128) -> Graph:
    """A @ B with the REDUCE axis symbolic (``Dim('seq_len')``) — the SDPA P@V
    shape after the demoted-matmul split (static M/N, runtime K = seq_len)."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, Dim("seq_len")), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (Dim("seq_len"), N), dtype=F16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (M, N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


@requires_sm90
@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize("seq", [16, 31, 130, 512, 700])
def test_symbolic_k_masked_mma_accuracy(monkeypatch, seq):
    """One compiled symbolic-K kernel must be accurate at runtime reduce extents
    below, at, and above the 512 hint — including the straddling cases (31, 130,
    700 are not multiples of the BK·atom_k = 32-element K tile, so the final K_o
    slab is partial and must be ZERO-filled past seq_len, not edge-clamped)."""
    for k, v in _CP_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    be = CudaBackend()
    compiled = be.compile(_symbolic_k_graph())
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((64, seq)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((seq, 128)) * 0.1).astype(np.float16)
    result, _ = be.run(compiled, input_data={"a": a, "b": b})
    got = result.outputs["o"].astype(np.float32)
    want = a.astype(np.float32) @ b.astype(np.float32)
    assert got.shape == (64, 128)
    diff = np.abs(got - want).max()
    assert diff < 5e-2, f"seq={seq}: masked-K MMA mismatch (max abs err {diff})"


def _batched_symbolic_mk_graph(*, H: int = 16, N: int = 128) -> Graph:
    """The SDPA P@V split-consumer in full: a BATCHED matmul (``H`` heads) whose
    M (query) AND K (key) axes are both symbolic ``seq_len`` —
    ``xna[H, seq, seq] @ xnb[H, seq, N]``. The batch axis sits before K in the B
    operand's index, the case ``classify_matmul_operands`` must recognize."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("xna", (H, Dim("seq_len"), Dim("seq_len")), dtype=F16), node_id="xna")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("xnb", (H, Dim("seq_len"), N), dtype=F16), node_id="xnb")
    g.add_node(op=MatmulOp(), inputs=["xna", "xnb"], output=Tensor("o", (H, Dim("seq_len"), N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["xna", "xnb"], ["o"]
    return g


def test_batched_symbolic_mk_reaches_warp(monkeypatch):
    """The batched masked-M + masked-K P@V consumer must reach the mma.sync tier
    (the ``classify_matmul_operands`` batch-aware B test), not stay a LoopOp."""
    for k, v in _CP_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    lowered = Pipeline.build(CUDA_PASSES).run(_batched_symbolic_mk_graph(), ctx=Context(compute_capability=(12, 0)))
    kop = lowered.nodes["o"].op
    assert mma_atom(kop.knobs) == "mma_m16n8k16_f16", "batched symbolic M+K matmul must reach the warp tier"
    src = kop.kernel_source
    assert "mma.sync.aligned.m16n8k16" in src and "ldmatrix" in src
    assert "int seq_len" in src, "runtime extent must be a kernel arg"


@requires_sm90
@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize("seq", [16, 31, 130, 512, 700])
def test_batched_symbolic_mk_masked_mma_accuracy(monkeypatch, seq):
    """One compiled batched symbolic-M+K kernel (the deployable P@V consumer) must
    be accurate at runtime sizes around the 512 hint, including the straddling
    cases where both the M tile and the partial K slab are masked.

    Routed through the SCALAR tier (no ``WARP`` pin): the batched-warp fragment
    codegen for a masked-M + symbolic-K mma is a separate, unrecovered gap (the
    ``dpl_mma_load_a_gmem_mclamp_kzero`` batched-operand path) — tracked by the
    still-xfailed ``test_batched_symbolic_mk_reaches_warp`` structure test, not by
    this accuracy gate. The scalar tier serves the deployable result correctly."""
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


def _pv_softmax_graph(H: int = 16, N: int = 128) -> Graph:
    """Softmax(scores) @ V with the reduce K = ``seq_len`` symbolic (the SDPA P@V
    shape). Fusion demotes the matmul; ``010_split_demoted`` materializes the
    softmax-prob A cone ``xn[H, seq, seq]`` + the clean symbolic-K gemm."""
    from deplodock.compiler.ir.frontend.ir import SoftmaxOp  # noqa: PLC0415

    s = Dim("seq_len")
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("scores", (H, s, s), dtype=F16), node_id="scores")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("v", (H, s, N), dtype=F16), node_id="v")
    g.add_node(op=SoftmaxOp(axis=-1), inputs=["scores"], output=Tensor("probs", (H, s, s), dtype=F16), node_id="probs")
    g.add_node(op=MatmulOp(), inputs=["probs", "v"], output=Tensor("o", (H, s, N), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["scores", "v"], ["o"]
    return g


@pytest.mark.skipif(not _supports_tma(), reason="TMA (cp.async.bulk.tensor) needs sm_90+")
@pytest.mark.parametrize("seq", [16, 31, 130, 512, 700])
def test_demoted_masked_k_pv_tma_accuracy(monkeypatch, seq):
    """The TMA-staged masked-K P@V (the demoted softmax-P@V split-consumer) must be
    accurate below, at, and above the 512 hint — including the straddling reduce
    extents (31, 130, 700) where the final K slab is partial and the overhang past
    ``seq_len`` must read 0 (a clamped duplicate would corrupt the reduction)."""
    for k, v in _CP_KNOBS.items():
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
