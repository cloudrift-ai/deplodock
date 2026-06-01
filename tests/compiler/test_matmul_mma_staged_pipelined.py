"""Tests for the staged + pipelined MMA codegen path (M2 of
``plans/mma-perf-closures.md``).

The pre-2026 ``005_lower_atom_tile._atom_body_to_mma`` walked the AtomTile
body looking for one ``(outer_st, reduce_st, enclosing_bundle)`` triple and
rebuilt the body from scratch. The shape ``080_pipeline_stages`` emits —
prologue StageBundle + K_o-1 SerialTile {issue-next bundle, AsyncWait,
K_i reduce, AsyncWait} + drain AsyncWait + epilogue K_i reduce + Write —
has elements the rebuild path couldn't reconstruct (the prologue, the
epilogue, the AsyncWaits). The rewrite raised ``RuleSkipped`` and the
AtomTile flowed through to render, erroring as ``AtomTile must be
consumed by the MMA materializer``.

The M2 fix replaces the rebuild path with a **transform walk** that
preserves every structural Stmt (StageBundle wraps, AsyncWait, K_o
SerialTile, prologue/epilogue) and only rewrites reduce SerialTiles
(body → ldmatrix + mma.sync chain, ``is_reduce`` cleared) and Write
(→ RegStore). Two sub-fixes ride alongside:

- **LdmatrixLoad src_index phase prefix** — buffered slabs are
  ``[phase, *cache_axes]``-shaped; the consumer Load index in the IR
  carries the leading phase. ``_mma_src_index`` splices the prefix back
  in front of the computed cache coords so the fragment reads the right
  slot.
- **A/B classification via Source.cache_dims** — for staged smem loads
  the K axis sits in the *middle* of the index tuple (not the last/first
  dim), so the legacy "K_name in load.index[-1]" heuristic misclassifies.
  We instead read the cache_dim whose ``axis.name == K_name`` and
  branch on ``source_dim`` (1=A, 0=B).

The plan's perf bench gate (≤ 102 µs at 2048² fp16) was verified
end-to-end by ``deplodock run --bench``: the pinned variant runs at
~249 µs vs the post-#182 baseline of ~108 µs, **doesn't beat baseline**,
and is therefore dropped from the Phase B golden search (per the plan's
keep/drop decision rule). The codegen fix stays — the IR was emitting a
pipelined-staged shape that crashed at materialize time; the fix lets it
compile and run correctly so the lever can be measured at all.
"""

from __future__ import annotations

import re

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
# per-test ``_supports_mma_sync`` skipif still gates the sm_80+ arch requirement.
pytestmark = [requires_cuda]


def _has_cuda() -> bool:
    try:
        import cupy as cp

        return cp.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _supports_mma_sync() -> bool:
    """The s16816 ``mma.sync.aligned.m16n8k16`` op + ``ldmatrix`` need
    sm_80+ (Ampere and later)."""
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


def _np_dtype(dt: DataType):
    return {F16: np.float16, F32: np.float32}[dt]


def _cp_dtype(dt: DataType, cp):
    return {F16: cp.float16, F32: cp.float32}[dt]


@pytest.fixture
def pin_staged_pipelined(monkeypatch):
    """Pin the warp-tier knob set that exercises the staged + pipelined
    cp.async MMA path: s16816 mma.sync atom (sm_80+), WM=WN=2 warps,
    FM=4 / FN=8 register cells (FN doubled vs the old WMMA pin since
    atom_n=8, not 16, preserving the 128×128 tile), BK=2 K-stage_inner
    trip count → the picker lands on the buffered-async path that
    ``080_pipeline_stages`` double-buffers via cp.async.

    ``DEPLODOCK_TMA=0`` keeps this test focused on the cp.async lever
    even on sm_90+ where ``050_use_tma`` would otherwise promote the
    bundle (post-block-aware eligibility fix); the TMA path has its own
    test suite.

    ``DEPLODOCK_BUFFER_COUNT=2`` pins the classic double-buffer ring so
    these tests exercise the ``% 2`` modular-phase addressing they were
    written against. Without the pin, ``040_use_ring_buffers``'s
    occupancy-ordered greedy default now front-loads a depth-3 ring for
    this 128×128 tile, whose ``080_pipeline_stages`` unroll emits explicit
    per-slot offsets instead of the ``a7 % 2`` phase Expr."""
    monkeypatch.setenv("DEPLODOCK_MMA", "1")
    monkeypatch.setenv("DEPLODOCK_TMA", "0")
    monkeypatch.setenv("DEPLODOCK_ATOM_KIND", "mma_m16n8k16_f16")
    monkeypatch.setenv("DEPLODOCK_WM", "2")
    monkeypatch.setenv("DEPLODOCK_WN", "2")
    monkeypatch.setenv("DEPLODOCK_FM", "4")
    monkeypatch.setenv("DEPLODOCK_FN", "8")
    monkeypatch.setenv("DEPLODOCK_BK", "2")
    monkeypatch.setenv("DEPLODOCK_BUFFER_COUNT", "2")


def _compile_and_render(*, M: int, N: int, K: int, out_dtype: DataType):
    from deplodock.compiler.ir.kernel.render import render_kernelop  # noqa: PLC0415

    g = _matmul_graph(M=M, N=N, K=K, out_dtype=out_dtype)
    g = Pipeline.build(KERNEL_PASSES).run(g)
    kop = g.nodes["c"].op
    tensors = {nid: n.output for nid, n in g.nodes.items() if hasattr(n.output, "shape")}
    src = render_kernelop(kop, tensors=tensors)
    return g, kop, src


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize("M,N,K", [(128, 128, 128), (256, 256, 128)])
def test_staged_pipelined_matches_f32_reference(pin_staged_pipelined, M: int, N: int, K: int):
    """The staged + pipelined MMA path produces output that matches the
    f32 reference within fp16 tolerance. Pre-M2 the AtomTile lowering
    raised ``RuleSkipped`` on the pipelined shape, so the kernel never
    compiled and the matmul never ran."""
    import cupy as cp

    from deplodock.compiler.backend.cuda.nvcc import compile_to_cubin  # noqa: PLC0415

    g, kop, src = _compile_and_render(M=M, N=N, K=K, out_dtype=F32)
    assert kop.knobs.get("ATOM_KIND") == "mma_m16n8k16_f16"
    # The pinned BK=2 with buffer_count >= 2 must produce the cp.async-staged
    # pipelined MMA path — verified at the C source.
    assert "cp.async.commit_group" in src
    assert "mma.sync.aligned.m16n8k16" in src
    assert "cp.async.wait_group" in src

    cap = cp.cuda.Device().compute_capability
    cubin_path = compile_to_cubin(src, kop.name, arch=f"sm_{cap}")
    mod = cp.RawModule(path=str(cubin_path))
    k = mod.get_function(kop.name)

    np.random.seed(42)
    a = (np.random.randn(M, K) * 0.1).astype(np.float16)
    b = (np.random.randn(K, N) * 0.1).astype(np.float16)
    a_cp = cp.asarray(a)
    b_cp = cp.asarray(b)
    c_cp = cp.zeros((M, N), dtype=cp.float32)

    knobs = kop.knobs
    wm, wn = int(knobs["WM"]), int(knobs["WN"])
    fm, fn = int(knobs["FM"]), int(knobs["FN"])
    splitk = int(knobs.get("SPLITK", 1))
    atom_m, atom_n = 16, 8  # mma_m16n8k16_f16
    m_b = max(1, M // (wm * fm * atom_m))
    n_b = max(1, N // (wn * fn * atom_n))
    grid_x = m_b * n_b * splitk
    threads_per_cta = wm * wn * 32
    k((grid_x,), (threads_per_cta,), (a_cp, b_cp, c_cp))

    expected = a.astype(np.float32) @ b.astype(np.float32)
    result = c_cp.get()
    diff = np.abs(result - expected).max()
    # F32 acc — tight tolerance.
    assert diff < 1e-2, f"M={M} N={N} K={K} max-abs-err {diff}"


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
def test_staged_pipelined_phase_prefix_renders(pin_staged_pipelined):
    """The buffered-slab phase prefix lands on every ``ldmatrix`` load offset
    (``[phase * stride + cache_coords]``). Pre-M2 ``_mma_src_index`` dropped
    the leading phase Expr; the load addressed slot 0 every iter, racing with
    cp.async's writes to slot 1 → garbled output. The A operand loads through
    ``dpl_ldmatrix_x4(in0_frag, &a_smem[...])`` and the B operand through
    ``dpl_ldmatrix_x2_trans(in1_frag, &b_smem[...])``."""
    _, _, src = _compile_and_render(M=128, N=128, K=128, out_dtype=F32)
    # ``a7 % 2`` is the inner-K_o reduce's phase Expr (``a7`` is the K_o
    # axis after axis renumbering). It must appear in the rendered ldmatrix
    # load offsets for both operand slabs.
    a_loads = re.findall(r"dpl_ldmatrix_\w+\(\w+, &a_smem\[([^\]]+)\]", src)
    b_loads = re.findall(r"dpl_ldmatrix_\w+\(\w+, &b_smem\[([^\]]+)\]", src)
    assert a_loads, "expected ≥1 a_smem ldmatrix load"
    assert b_loads, "expected ≥1 b_smem ldmatrix load"
    # At least one inner-loop load addresses the buffered slot via a
    # ``% 2`` phase factor on a_smem.
    assert any("% 2" in o for o in a_loads), f"a_smem loads missing phase prefix: {a_loads}"
    assert any("% 2" in o for o in b_loads), f"b_smem loads missing phase prefix: {b_loads}"


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
def test_staged_pipelined_ab_classification(pin_staged_pipelined):
    """A vs B classification via ``Source.cache_dims`` — the cache axis
    whose ``axis.name == K_name`` has ``source_dim == 1`` for A (K is the
    inner gmem dim) or ``0`` for B (K is the outer gmem dim). Pre-M2 the
    classification used ``K_name in load.index[-1]`` which for staged
    slabs misses (K sits in the middle of the slab-coord tuple).
    Smoke-test: the rendered ``dpl_mma_m16n8k16_f16(acc, a, b, acc)`` call
    has ``a`` referring to the A operand fragment loaded from ``a_smem`` and
    ``b`` to the B operand fragment loaded from ``b_smem`` — i.e. no swap."""
    _, _, src = _compile_and_render(M=128, N=128, K=128, out_dtype=F32)
    # Find the first mma.sync call: dpl_mma_m16n8k16_f16(acc, a, b, acc).
    m = re.search(r"dpl_mma_m16n8k16_f16\((\w+),\s*(\w+),\s*(\w+),\s*\1\)", src)
    assert m is not None, "expected at least one dpl_mma_m16n8k16_f16 call"
    _, a_arg, b_arg = m.groups()
    # The fragment SSA naming is order-dependent (pulled from the Load's
    # ``names[0]``), so we don't pin specific names. Instead verify the
    # round-trip: find the ldmatrix load that defined ``a_arg`` and check it
    # reads from ``a_smem`` (M-side gmem buffer); same for ``b_arg`` and
    # ``b_smem``. Pre-M2 the classification keyed off ``K_name in
    # load.index[-1]`` which for staged slabs misses (K sits in the middle
    # of the slab-coord tuple), swapping A↔B and yielding
    # ``dpl_mma_m16n8k16_f16(acc, b_frag, a_frag, acc)`` — the K↔M↔N matmul
    # indexing collapses to wrong output.
    a_load_m = re.search(rf"dpl_ldmatrix_\w+\({re.escape(a_arg)}, &(\w+)\[", src)
    b_load_m = re.search(rf"dpl_ldmatrix_\w+\({re.escape(b_arg)}, &(\w+)\[", src)
    assert a_load_m is not None and b_load_m is not None
    assert a_load_m.group(1) == "a_smem", f"a operand should load from a_smem, got {a_load_m.group(1)}"
    assert b_load_m.group(1) == "b_smem", f"b operand should load from b_smem, got {b_load_m.group(1)}"


# ---------------------------------------------------------------------------
# REG_PIPELINE — register-tier operand double-buffer (plans/mma-register-pipeline.md)
# ---------------------------------------------------------------------------


def _supports_tma() -> bool:
    """``REG_PIPELINE``'s cross-K_o prefetch only fires on the TMA-pipelined K_o
    loop (the ``080``-pipelined ring with per-slot mbarrier waits), which needs
    sm_90+. On sm_80 the cp.async path has no phase/slot wait to relocate."""
    if not _has_cuda():
        return False
    import cupy as cp

    return int(cp.cuda.Device().compute_capability) >= 90


@pytest.fixture
def pin_reg_pipeline_tma(monkeypatch):
    """Same warp-tier tile as :func:`pin_staged_pipelined`, but TMA left **on**
    (sm_90+) so ``080`` produces the TMA-pipelined K_o loop that
    ``REG_PIPELINE`` prefetches across. (``REG_PIPELINE`` is a no-op on the
    cp.async path that ``DEPLODOCK_TMA=0`` forces.)"""
    monkeypatch.setenv("DEPLODOCK_MMA", "1")
    monkeypatch.setenv("DEPLODOCK_ATOM_KIND", "mma_m16n8k16_f16")
    monkeypatch.setenv("DEPLODOCK_WM", "2")
    monkeypatch.setenv("DEPLODOCK_WN", "2")
    monkeypatch.setenv("DEPLODOCK_FM", "4")
    monkeypatch.setenv("DEPLODOCK_FN", "8")
    monkeypatch.setenv("DEPLODOCK_BK", "2")
    monkeypatch.setenv("DEPLODOCK_BUFFER_COUNT", "4")


@pytest.mark.skipif(not _supports_tma(), reason="REG_PIPELINE cross-K_o prefetch needs the TMA path (sm_90+)")
def test_reg_pipeline_off_no_buffer(pin_reg_pipeline_tma):
    """``REG_PIPELINE`` defaults off — no ``__rp1`` second-operand buffer appears
    (the knob is a measured fork, never forced on, so the greedy / DB-less path
    stays unchanged)."""
    _, _, src_off = _compile_and_render(M=256, N=256, K=256, out_dtype=F32)
    assert "__rp1" not in src_off, "REG_PIPELINE-off kernel must not declare a second operand buffer"


@pytest.mark.skipif(not _supports_tma(), reason="REG_PIPELINE cross-K_o prefetch needs the TMA path (sm_90+)")
def test_reg_pipeline_cross_ko_prefetch_shape(pin_reg_pipeline_tma, monkeypatch):
    """With ``REG_PIPELINE=1`` the TMA-pipelined K_o loop gains the cross-K_o
    prefetch: a ``__rp1`` second operand buffer declared + loaded via ldmatrix,
    an iteration-0 prime guarded by ``== 0``, and the accumulator left single-
    buffered (never aliased)."""
    monkeypatch.setenv("DEPLODOCK_REG_PIPELINE", "1")
    _, kop, src = _compile_and_render(M=256, N=256, K=256, out_dtype=F32)
    assert kop.knobs.get("REG_PIPELINE") is True
    assert "__rp1" in src, "REG_PIPELINE-on kernel must declare a second operand buffer"
    rp1_ldm = re.findall(r"dpl_ldmatrix_\w+\((\w*__rp1)\b", src)
    assert rp1_ldm, "expected ldmatrix loads into the __rp1 prefetch buffer"
    # Iteration-0 prime: a `== 0` guarded block seeds the buffer.
    assert re.search(r"if\s*\([^)]*==\s*0\)", src), "expected the iteration-0 prime guard"
    # The accumulator is never double-buffered — no acc fragment carries the suffix.
    assert not re.search(r"acc\w*__rp1", src), "accumulator must stay single-buffered"
    # End-to-end ``max_diff = 0`` accuracy is validated via ``deplodock run`` on
    # the live device — a raw cubin launch can't supply the host-built TMA
    # descriptor this path needs, so the unit test asserts shape only.
