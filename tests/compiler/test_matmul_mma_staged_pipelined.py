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
(body → Mma chain, ``is_reduce`` cleared) and Write (→ MmaStore). Two
sub-fixes ride alongside:

- **MmaLoad src_index phase prefix** — buffered slabs are
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
# per-test ``_supports_wmma`` skipif still gates the sm_70+ arch requirement.
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


def _cp_dtype(dt: DataType, cp):
    return {F16: cp.float16, F32: cp.float32}[dt]


@pytest.fixture
def pin_staged_pipelined(monkeypatch):
    """Pin the warp-tier knob set that exercises the staged + pipelined
    cp.async MMA path: square WMMA atom (sm_70+), WM=WN=2 warps,
    FM=FN=4 register cells, BK=2 K-stage_inner trip count → the picker
    lands on the buffered-async path that ``080_pipeline_stages``
    double-buffers via cp.async.

    ``DEPLODOCK_TMA=0`` keeps this test focused on the cp.async lever
    even on sm_90+ where ``050_use_tma`` would otherwise promote the
    bundle (post-block-aware eligibility fix); the TMA path has its own
    test suite."""
    monkeypatch.setenv("DEPLODOCK_MMA", "1")
    monkeypatch.setenv("DEPLODOCK_TMA", "0")
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


@pytest.mark.skipif(not _supports_wmma(), reason="WMMA needs CUDA + sm_70+")
@pytest.mark.parametrize("M,N,K", [(128, 128, 128), (256, 256, 128)])
def test_staged_pipelined_matches_f32_reference(pin_staged_pipelined, M: int, N: int, K: int):
    """The staged + pipelined MMA path produces output that matches the
    f32 reference within fp16 tolerance. Pre-M2 the AtomTile lowering
    raised ``RuleSkipped`` on the pipelined shape, so the kernel never
    compiled and the matmul never ran."""
    import cupy as cp

    from deplodock.compiler.backend.cuda.nvcc import compile_to_cubin  # noqa: PLC0415

    g, kop, src = _compile_and_render(M=M, N=N, K=K, out_dtype=F32)
    assert kop.knobs.get("ATOM_KIND") == "wmma_m16n16k16_f16"
    # The pinned BK=2 with buffer_count >= 2 must produce the cp.async-staged
    # pipelined MMA path — verified at the C source.
    assert "cp.async.commit_group" in src
    assert "wmma::mma_sync" in src
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
    atom_m = atom_n = 16  # wmma_m16n16k16_f16
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


@pytest.mark.skipif(not _supports_wmma(), reason="WMMA needs CUDA + sm_70+")
def test_staged_pipelined_phase_prefix_renders(pin_staged_pipelined):
    """The buffered-slab phase prefix lands on every ``wmma::load_matrix_sync``
    offset (``[phase * stride + cache_coords]``). Pre-M2 ``_mma_src_index``
    dropped the leading phase Expr; the load addressed slot 0 every iter,
    racing with cp.async's writes to slot 1 → garbled output."""
    _, _, src = _compile_and_render(M=128, N=128, K=128, out_dtype=F32)
    # ``a7 % 2`` is the inner-K_o reduce's phase Expr (``a7`` is the K_o
    # axis after axis renumbering); ``1`` (literal) is the epilogue's
    # last-slot read. Both must appear in the rendered load offsets.
    a_loads = re.findall(r"wmma::load_matrix_sync\(\w+, &a_smem\[([^\]]+)\]", src)
    b_loads = re.findall(r"wmma::load_matrix_sync\(\w+, &b_smem\[([^\]]+)\]", src)
    assert a_loads, "expected ≥1 a_smem load"
    assert b_loads, "expected ≥1 b_smem load"
    # At least one inner-loop load addresses the buffered slot via a
    # ``% 2`` phase factor on a_smem.
    assert any("% 2" in o for o in a_loads), f"a_smem loads missing phase prefix: {a_loads}"
    assert any("% 2" in o for o in b_loads), f"b_smem loads missing phase prefix: {b_loads}"


@pytest.mark.skipif(not _supports_wmma(), reason="WMMA needs CUDA + sm_70+")
def test_staged_pipelined_ab_classification(pin_staged_pipelined):
    """A vs B classification via ``Source.cache_dims`` — the cache axis
    whose ``axis.name == K_name`` has ``source_dim == 1`` for A (K is the
    inner gmem dim) or ``0`` for B (K is the outer gmem dim). Pre-M2 the
    classification used ``K_name in load.index[-1]`` which for staged
    slabs misses (K sits in the middle of the slab-coord tuple).
    Smoke-test: the rendered ``mma_sync`` arguments end up as
    ``mma_sync(c, a, b, c)`` with ``a`` referring to ``in1_frag`` (the
    A operand loaded from ``a_smem``) and ``b`` to ``in0_frag`` (the B
    operand loaded from ``b_smem``) — i.e. no swap."""
    _, _, src = _compile_and_render(M=128, N=128, K=128, out_dtype=F32)
    # Find the first mma_sync call.
    m = re.search(r"wmma::mma_sync\((\w+),\s*(\w+),\s*(\w+),\s*\1\)", src)
    assert m is not None, "expected at least one mma_sync"
    _, a_arg, b_arg = m.groups()
    # The fragment SSA naming is order-dependent (pulled from the Load's
    # ``names[0]``), so we don't pin specific names. Instead verify the
    # round-trip: find the wmma::load_matrix_sync that defined ``a_arg``
    # and check it reads from ``a_smem`` (M-side gmem buffer); same for
    # ``b_arg`` and ``b_smem``. Pre-M2 the classification keyed off
    # ``K_name in load.index[-1]`` which for staged slabs misses (K sits
    # in the middle of the slab-coord tuple), swapping A↔B and yielding
    # ``mma_sync(c, b_frag, a_frag, c)`` — the K↔M↔N matmul indexing
    # collapses to wrong output.
    a_load_m = re.search(rf"wmma::load_matrix_sync\({re.escape(a_arg)}, &(\w+)\[", src)
    b_load_m = re.search(rf"wmma::load_matrix_sync\({re.escape(b_arg)}, &(\w+)\[", src)
    assert a_load_m is not None and b_load_m is not None
    assert a_load_m.group(1) == "a_smem", f"a operand should load from a_smem, got {a_load_m.group(1)}"
    assert b_load_m.group(1) == "b_smem", f"b operand should load from b_smem, got {b_load_m.group(1)}"
