"""CUDA accuracy regression for specific planner knob tuples.

Where ``test_tune_accuracy`` lets the search wander and checks the
*picked* variant, this file pins each output-fragment / reduce / stage
codec (the ``TILE`` / ``REDUCE`` / ``STAGE`` knobs) to a configuration
that previously emitted a wrong-output kernel and confirms the lowered
kernel matches the numpy-backend reference within fp32 tolerance.

Knob pinning rides on the existing ``DEPLODOCK_KNOBS="K1=V1,..."``
env-var mechanism (see ``deplodock/compiler/pipeline/knob.py`` —
``apply_knobs_env`` splats the aggregate into per-knob
``DEPLODOCK_<K>=V`` vars at import time, and ``Knob.narrow`` overrides
the schedule's candidate codecs with the pinned value in
``lowering/tile/020_schedule`` so only the matching variant is built).

The shared failure mode is the "single-CTA + F-replicated" codegen
class: ``BN·FN = full_N AND BM·FM = full_M`` with ``FM·FN > 1`` (so
``N_b = M_b = 1``). When the extent-1 BLOCK Loops got dropped by
``drop_size_one_free_axes`` the Tile renderer fell through to its
linear-flatten path (grid = ceil(threads/256), block = 256), which
runs the cooperative-smem body across **two CTAs** that each only
loaded half the smem — output garbage.

Each parametrized config below maps 1:1 to a kernel the user surfaced
during the offline knob sweep:

- ``matmul``: 4 broken (BN, BM, FM, FN) tiles. Reference peak ≈ 37;
  pre-fix max_diff 30–42 (essentially noise).
- ``gated_mlp``: 4 broken (BN, BM, FM, FN) tiles. Reference peak
  ≈ 1195; pre-fix max_diff 770–1195.

Skipped without CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..conftest import dyn_M, requires_cuda


def _format_knobs(knobs: dict) -> str:
    """Render a knob dict as ``"K1=V1,K2=V2,..."`` for ``DEPLODOCK_KNOBS``."""
    return ",".join(f"{k}={v}" for k, v in knobs.items())


def _build_norm_linear_graph(dims: dict, mode: str = "static"):
    """``RmsNorm(x) @ W.T`` in fp16 — the fused norm+matmul shape of
    Qwen3-Embedding's ``k_linear_mean_reduce``. ``mode='dynamic'`` makes the
    seq axis symbolic (``Dim('seq_len')``, the deployable masked-tile path).
    The norm reduction stages ``x`` in ``BK``-element inner slabs; at fp16 +
    ``BK=32`` that slab is 64 B (not 128 B-aligned), the TMA-misalignment hazard."""
    from deplodock.compiler.dtype import F16
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import LinearOp, RmsNormOp

    S, H, Inter = dims["S"], dims["H"], dims["I"]
    Sg = dyn_M(mode, S)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, Sg, H), dtype=F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("wn", (H,), dtype=F16), node_id="wn")
    g.add_node(InputOp(), [], Tensor("w", (Inter, H), dtype=F16), node_id="w")
    g.add_node(RmsNormOp(), ["x", "wn"], Tensor("xn", (1, Sg, H), dtype=F16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "w"], Tensor("y", (1, Sg, Inter), dtype=F16), node_id="y")
    g.inputs, g.outputs = ["x", "wn", "w"], ["y"]
    return g, {"x": (1, S, H), "wn": (H,), "w": (Inter, H)}, ("y", (1, S, Inter))


def _run_with_knobs(graph, inputs: dict[str, np.ndarray], out_name: str, knobs: dict, monkeypatch) -> np.ndarray:
    """Set the per-knob ``DEPLODOCK_<K>`` env vars (the same pinning
    mechanism ``DEPLODOCK_KNOBS=...`` uses after ``apply_knobs_env``
    splats it) so the partition planner filters its variant enumeration
    down to the single ``TileParams`` we want to verify, then compile
    + run via the CUDA backend."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    for k, v in knobs.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", str(v))

    be = CudaBackend()
    compiled = be.compile(graph)
    return be.run(compiled, input_data=inputs)[0].outputs[out_name]


def _reference(graph, inputs: dict[str, np.ndarray], out_name: str) -> np.ndarray:
    """Numpy reference — runs the same Graph on the NumpyBackend so the
    comparison stays self-contained (no torch dependency)."""
    from deplodock.compiler.backend.numpy import NumpyBackend

    be = NumpyBackend()
    compiled = be.compile(graph)
    return be.run(compiled, input_data=inputs)[0].outputs[out_name]


def _random_inputs(input_shapes: dict[str, tuple[int, ...]], seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal(shape, dtype=np.float32) for name, shape in input_shapes.items()}


def _assert_match(forced: np.ndarray, ref: np.ndarray) -> None:
    assert forced.shape == ref.shape, f"shape mismatch: {forced.shape} vs {ref.shape}"
    assert np.all(np.isfinite(forced)), "forced-knob output has non-finite values"
    # fp32 reduction-order drift across CTAs vs numpy's pairwise sum
    # can push max_diff to a few percent of peak — same 5%-of-peak
    # tolerance ``deplodock run --bench`` uses.
    peak = float(np.max(np.abs(ref)))
    atol = max(1e-3, 0.05 * peak)
    np.testing.assert_allclose(forced, ref, atol=atol, rtol=0.05)


# The single-CTA + F-replicated matmul / gated-mlp register-tile cases that used to live here
# (pinned via legacy ``BN``/``BM``/``FM``/``FN``) are now covered by the new-schema ``TILE`` codec
# matrix in ``test_matmul_tile_coverage`` (static AND dynamic, accuracy + lowering structure).


# The #244 dynamic-tune wedge: a scalar (no-atom) cooperative norm-reduce whose fp16 per-slot box
# is 64 B — not a 128 B multiple, so under a depth-2 staging ring the second slot lands at a 64 B
# offset. The 128 B slot check sized off the fp32 ``BYTES_PER_ELEM`` constant let the fp16 64 B slab
# through, the materializer left it unpadded, and ``cp.async.bulk.tensor`` faulted with
# ``CUDA_ERROR_MISALIGNED_ADDRESS`` → 1 s watchdog hang → bench_fail. The dtype-aware gate now
# declines TMA for this slab (→ cp.async). The scalar ``TILE`` (no ``a:`` atom) forces the scalar
# tier; the depth-2 ``STAGE`` ring double-buffers so the slot offset matters.
_NORM_REDUCE_WEDGE_KNOBS = {"TILE": "n128", "STAGE": "d2/cp"}
_NORM_DIMS = {"S": 32, "H": 128, "I": 512}


@requires_cuda
def test_norm_linear_fp16_scalar_reduce_tma_alignment(shape_mode, monkeypatch):
    """fp16 ``RmsNorm(x) @ W.T`` forced through the scalar cooperative-reduce tile
    that wedged the #244 dynamic tune. The pinned config must produce correct
    output whether the seq axis is baked (static) or symbolic (dynamic
    ``seq_len``) — STATIC/DYNAMIC PARITY: both paths must make the same TMA
    decision (decline, since the fp16 64 B ring slot isn't 128 B-aligned) and fall
    back to cp.async. Pre-fix the dynamic path emitted a misaligned
    ``cp.async.bulk.tensor`` store and hung (``HungKernelError``); this test would
    time out. The numpy reference is the static graph; the forced CUDA run uses
    the mode's graph."""
    # This pins a scalar cooperative-reduce config for the demoted matmul's CUT producer
    # kernel (the #244 TMA wedge); SPLIT_CONE=1 forces that GMEM cut (the default is now the
    # keep(SMEM) fused edge, which has no separate norm-reduce producer to wedge).
    monkeypatch.setenv("DEPLODOCK_SPLIT_CONE", "1")
    static_graph, input_shapes, (out_name, _) = _build_norm_linear_graph(_NORM_DIMS)
    forced_graph, _, _ = _build_norm_linear_graph(_NORM_DIMS, mode=shape_mode)
    rng = np.random.default_rng(0)
    # fp16 inputs scaled down so the H=128 sum-of-squares stays well inside fp16 range.
    inputs = {name: (rng.standard_normal(shape, dtype=np.float32) * 0.1).astype(np.float16) for name, shape in input_shapes.items()}
    ref = _reference(static_graph, inputs, out_name)
    forced = _run_with_knobs(forced_graph, inputs, out_name, _NORM_REDUCE_WEDGE_KNOBS, monkeypatch)
    # fp16 reduction drift over H=128 — same looser bound the other fp16 tests use.
    peak = float(np.max(np.abs(ref.astype(np.float32))))
    atol = max(5e-1, 0.1 * peak)
    np.testing.assert_allclose(forced.astype(np.float32), ref.astype(np.float32), atol=atol, rtol=0.1)


# Staged MMA regression. These
# shapes pick a K-split (K_o > 1) under the planner's natural priors,
# placing a SerialTile(K_o) between the AtomTile and the StageBundle
# (which wraps SerialTile(K_i) > loads). The buggy pre-fix shape
# emitted the lowered Mma chain with the bundle hoisted ABOVE K_o, but
# the bundle's ``Source.origin`` references the K_o coord (e.g.
# ``a[a0*16, a5*1024]``), so the kernel failed to compile with
# ``identifier 'a5' is undefined``.
#
# The warp-tier 16/64/128/256 squares all naturally pick K_o = 1
# (full K fits in one stage); we need an asymmetric M/N/K shape
# with large K to force the K-split.
_MMA_K_SPLIT_SHAPES: tuple[tuple[int, int, int], ...] = (
    # (M, N, K). 2048³ matches the original reproducer from the
    # ``deplodock run --bench`` investigation; smaller shapes fit
    # K fully in one stage and the regression doesn't surface.
    (2048, 2048, 2048),
)


@requires_cuda
@pytest.mark.parametrize(("M", "N", "K"), _MMA_K_SPLIT_SHAPES, ids=lambda v: f"{v[0]}x{v[1]}x{v[2]}" if isinstance(v, tuple) else str(v))
def test_mma_matmul_k_split_staged(M: int, N: int, K: int, monkeypatch):
    """MMA-staged matmul with K_o > 1 — surfaces the
    ``005_lower_atom_tile`` body-walk regression where the StageBundle
    sits inside a K_o SerialTile inside the AtomTile. The lowered Mma
    chain re-wraps with the bundle BETWEEN K_o and K_i so the producer
    ``Source.origin`` (which references the K_o coord) renders in scope.
    """
    import numpy as np

    from deplodock.compiler.dtype import F16
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp

    monkeypatch.setenv("DEPLODOCK_TILE", "a:mma_m16n8k16_f16/w2x2/f2x2/k2")  # force the warp (mma) tier
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N), dtype=F16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("c", (M, N), dtype=F16), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]

    rng = np.random.default_rng(0)
    inputs = {
        "a": (rng.standard_normal((M, K), dtype=np.float32) * 0.1).astype(np.float16),
        "b": (rng.standard_normal((K, N), dtype=np.float32) * 0.1).astype(np.float16),
    }
    ref = (inputs["a"].astype(np.float32) @ inputs["b"].astype(np.float32)).astype(np.float16)
    forced = _run_with_knobs(g, inputs, "c", {}, monkeypatch)
    # Looser tolerance for fp16 acc on K=2048: the drift bound is
    # ~K · max · f16-eps, and the 0.05-of-peak default isn't enough.
    peak = float(np.max(np.abs(ref.astype(np.float32))))
    atol = max(5e-1, 0.1 * peak)
    np.testing.assert_allclose(forced.astype(np.float32), ref.astype(np.float32), atol=atol, rtol=0.1)


# Scalar-FMA fp16 regression (prerequisite for the split-pipe GEMM). On cc>=9.0 the F16 atom is
# eligible whenever the K-loads are F16, so at >=512^3 the greedy compile
# *prefers* the tensor-core variant; a scalar ``TILE`` codec (no ``a:`` atom) is what forces the
# scalar register-tile FMA path (fp16 in -> fp32 accumulate -> fp16 out). The
# 512^3 shape is the smallest where the scalar pin is load-bearing (<=256^3 greedy picks
# scalar on its own, so a small shape wouldn't prove the knob does anything).
def _build_f16_matmul_graph(M: int, N: int, K: int):
    from deplodock.compiler.dtype import F16
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N), dtype=F16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("c", (M, N), dtype=F16), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    rng = np.random.default_rng(0)
    inputs = {
        "a": (rng.standard_normal((M, K), dtype=np.float32) * 0.1).astype(np.float16),
        "b": (rng.standard_normal((K, N), dtype=np.float32) * 0.1).astype(np.float16),
    }
    ref = (inputs["a"].astype(np.float32) @ inputs["b"].astype(np.float32)).astype(np.float16)
    return g, inputs, ref


# Scalar-tile knob: a scalar ``TILE`` codec (no ``a:`` atom) forces the scalar register-tile FMA
# path — the ``n32x8/f4x26`` (BN=32 BM=8 FN=4 FM=26) hero tile, gmem-direct.
_SCALAR_F16_KNOBS = {"TILE": "n32x8/f4x26"}


@requires_cuda
def test_scalar_matmul_f16(monkeypatch):
    """fp16 matmul forced through the scalar register-tile FMA path (a scalar ``TILE`` codec, no
    ``a:`` atom); fp32 accumulate, verified vs the numpy backend. Guards against fp16 being
    re-routed to the tensor-core atom path and against the fp16 scalar render (``__half2float``
    multiply, ``__float2half`` store) regressing."""
    g, inputs, ref = _build_f16_matmul_graph(512, 512, 512)
    forced = _run_with_knobs(g, inputs, "c", _SCALAR_F16_KNOBS, monkeypatch)
    peak = float(np.max(np.abs(ref.astype(np.float32))))
    atol = max(5e-1, 0.1 * peak)
    np.testing.assert_allclose(forced.astype(np.float32), ref.astype(np.float32), atol=atol, rtol=0.1)


def test_sgemm_inner_reduce_is_unrolled(monkeypatch):
    """``assembly/030_mark_unroll`` flags the small FMA inner reduce (the ``BK=32`` K
    loop, ≤ 64 trips) for ``#pragma unroll``, giving ptxas the register-resident
    operand reuse + ILP the hand-tuned SGEMM relies on (the ``TM=26`` hero tile runs
    at 255 regs / ~293 µs with the unroll, ~126 regs / ~384 µs without — the lever for
    the article's ~96 %-of-cuBLAS number). The K-outer pipeline loop (> 64 unrolled
    trips) stays rolled. Compile-only (inspects the kernel source). No CUDA needed."""
    from deplodock.compiler.context import Context
    from deplodock.compiler.ir.cuda.ir import CudaOp
    from deplodock.compiler.pipeline import KERNEL_PASSES, Pipeline

    g, _, _ = _build_2d_matmul_graph(_ARTICLE_DIMS)
    for k, v in {"TILE": "n32x8/f4x26", "STAGE": "d2/tma"}.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", str(v))
    res = Pipeline.build([*KERNEL_PASSES, "lowering/cuda"]).run(g, ctx=Context.from_target((12, 0)))
    src = "\n".join(n.op.kernel_source for n in res.nodes.values() if isinstance(n.op, CudaOp))
    assert "#pragma unroll" in src, "the small FMA inner reduce must be marked for #pragma unroll"


@requires_cuda
def test_unstaged_atom_lowers_gmem_direct(monkeypatch):
    """When the greedy compile picks the tensor-core atom variant but its operands
    aren't staged for ``ldmatrix`` (``TMA=0`` + a deliberately-large warp register
    tile whose slabs don't fit the smem budget), ``005_lower_atom_tile`` lowers them
    to a **gmem-direct fragment load** (``dpl_mma_load_{a,b}_gmem``) instead of
    raising — ldmatrix is smem-only, so the gmem path lets an unstageable MMA tile
    compile rather than crash. Compile-only (inspects the kernel source).

    Two facts make this fire: (1) the over-ceiling ``FM=26`` warp register pin is
    authoritative (``warp_reg_offers`` bypasses the ``_MAX_WARP_CELLS`` search
    ceiling for a full pin), so the warp build proceeds; (2) with **no** ``STAGE``
    pin the budget-aware ``120_stage`` filter prunes every over-budget staging subset
    to the empty one (``FM=26`` slabs blow the smem cap), so greedy's option-0 stages
    nothing and the operands lower gmem-direct."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.ir.cuda.ir import CudaOp

    g, _, _ = _build_f16_matmul_graph(512, 512, 512)
    # Pin the WARP-tier geometry via a warp ``TILE`` codec (the over-ceiling ``f26x4`` register
    # tile + atom-K chunk) and leave STAGE unpinned — an explicit STAGE pin is authoritative (no
    # budget filter), but here we want the budget-aware filter to decline the over-budget staging so
    # the operands fall to the gmem-direct path. The ``a:<atom>`` token forces the warp (mma) tier.
    monkeypatch.setenv("DEPLODOCK_TILE", "a:mma_m16n8k16_f16/w1x1/f26x4/k2")
    compiled = CudaBackend().compile(g)  # no longer raises
    src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if isinstance(n.op, CudaOp))
    assert "dpl_mma_load_a_gmem" in src and "dpl_mma_load_b_gmem" in src, "unstaged operands not loaded gmem-direct"
    assert "mma.sync.aligned.m16n8k16" in src, "tensor-core path not taken (scalar fallback?)"


@requires_cuda
def test_unstaged_atom_mma_accuracy(monkeypatch):
    """The gmem-direct mma fragment load must produce CORRECT results — a wrong
    lane→element map silently corrupts the matmul. Force an unstaged tensor-core
    matmul (a warp ``TILE`` pin, STAGE unpinned ⇒ gmem-direct) and verify it matches
    the numpy reference. Guards the m16n8k16 fragment layout in the gmem helpers."""
    g, inputs, ref = _build_f16_matmul_graph(128, 128, 128)
    knobs = {"TILE": "a:mma_m16n8k16_f16/w2x2/f2x2/k1"}
    forced = _run_with_knobs(g, inputs, "c", knobs, monkeypatch)
    _assert_match(forced.astype(np.float32), ref.astype(np.float32))


# Accuracy regressions for the fp32 SGEMM tile shapes from the matmul-optimization
# blog posts ("Modern GPU Matmul Optimization" and "Surfacing a 60% performance bug
# in cuBLAS" — both a scalar SIMT fp32 SGEMM on the RTX 5090, the ``TM=26`` register
# tile at 2048³ reaching ~106 % of cuBLAS). Each row pins a tile geometry that once
# emitted a wrong-output kernel and asserts the lowered kernel still matches the numpy
# reference:
#
# - ``BM=8``: an out-of-hint THREAD width. ``_TUNE_AXIS_CHOICES`` excludes 8, so a
#   non-authoritative ``Knob.narrow`` would drop the pin and re-enumerate the default
#   geometry. The pin must be honored regardless of hint membership.
# - ``FM=26`` on ``BM=8, M=2048``: a non-divisor register tile → a 208-row masked-M
#   tile (ceil-div grid + per-row boundary ``Cond`` on the Write). The article's hero
#   tile (``TM=26``); ``FN=26`` is the symmetric masked-N case.
# - ``FN=4`` / ``FM=4``: a multi-axis composite register tile (``BN_thread ×
#   FN_register`` on one source dim) — its strided addressing must round-trip through
#   the smem stage.
# - ``INTERLEAVE_LOADS=0``: the ``kernel/095_interleave_loads`` opt-out (flat-LDS
#   layout) must still produce correct output.
#
# NOTE: TMA (``cp.async.bulk.tensor``) on the scalar SGEMM tile is covered by the
# ``*_tma`` rows below (which pin ``STAGE=d2/tma``): ``130_transport`` promotes the scalar
# fp32 tier as well as the warp-tier MMA atom, depositing the slab unswizzled for the
# plain-``Load`` consumer.

# 2048×2048 fp32 matmul = the blogs' hero shape. The configs are benched against a
# numpy reference at this size so the masked-tile (FM=26) overhang math + the
# composite-stride staging are actually exercised on the full tile.
_ARTICLE_DIMS = {"M": 2048, "K": 2048, "N": 2048}


def _build_2d_matmul_graph(dims: dict):
    """2D matmul ``a (M, K) @ b (K, N)`` — the canonical SGEMM shape
    the article kernel targets, no leading batch dim."""
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp

    M, K, N = dims["M"], dims["K"], dims["N"]
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (M, K)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (M, N)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    return g, {"a": (M, K), "b": (K, N)}, ("c", (M, N))


# (label, dims, knobs, env extras). ``dims`` is the M/K/N; ``knobs`` is the per-knob
# ``DEPLODOCK_<K>`` pin set; ``env`` carries non-knob ``DEPLODOCK_*`` vars (today only
# the ``INTERLEAVE_LOADS`` opt-out). Masked-N (FN=26) uses a smaller 512³ shape — the
# per-launch watchdog in ``program.py`` is 1 s and the masked kernels are scalar
# (SYNC) here, so the small shape keeps comfortably inside it under parallel xdist.
#
# Every row is the SCALAR fp32 tier (no MMA atom is eligible for fp32). Most rows stage
# via plain SYNC cooperative loads; the ``*_tma`` rows pin ``STAGE=d2/tma`` so the staged
# operands ride the ``cp.async.bulk.tensor`` double-buffer ring — ``130_transport`` now
# promotes the scalar tier, depositing the slab unswizzled for the plain-``Load`` consumer.
# The dead ``ASYNC_COPY`` / ``PIPELINE_STAGES`` knobs (whose
# passes were removed in the block-DAG rewrite) were dropped here.
_SMALL_DIMS = {"M": 512, "K": 512, "N": 512}
_MASKED_TILE_CONFIGS: tuple[tuple[str, dict, dict, dict], ...] = (
    # A scalar TILE pin (n32x8 = BN=32 BM=8) the planner must honor authoritatively.
    ("bm8_pin_outside_hints", _ARTICLE_DIMS, {"TILE": "n32x8"}, {}),
    # reg_m=26 (FM) non-divisor of M → a 208-row masked-M tile (ceil-div grid + per-row
    # boundary Cond on the Write).
    ("fm26_masked_m", _ARTICLE_DIMS, {"TILE": "n32x8/f1x26"}, {}),
    # reg_n=4 (FN) multi-axis composite cache on N (BN_thread × FN_register on one source dim) —
    # the composite stride must round-trip through smem stage → revert-to-gmem.
    ("multi_axis_fn4", _ARTICLE_DIMS, {"TILE": "n32x8/f4"}, {}),
    # Multi-axis composite on BOTH axes: reg_m=4 and reg_n=4 (the FM×FN > 1 register tile).
    ("multi_axis_fm4_fn4", _ARTICLE_DIMS, {"TILE": "n32x8/f4x4"}, {}),
    # kernel/095_interleave_loads opt-out (flat-LDS layout) must still be correct.
    (
        "interleave_loads_disabled",
        _ARTICLE_DIMS,
        {"TILE": "n32x8/f4x4"},
        {"DEPLODOCK_INTERLEAVE_LOADS": "0"},
    ),
    # Symmetric masked-N: reg_n=26 reg_m=4 on 512³ → a 320-col overhang (boundary Cond on N).
    ("fn26_masked_n", _SMALL_DIMS, {"TILE": "n32x8/f26x4"}, {}),
    # Scalar-tile TMA: the staged operands ride the cp.async.bulk.tensor ring (unswizzled
    # deposit, plain-Load consumer). A clean (f4x4) tile exercises the box copy.
    ("tma_clean_fm4_fn4", _ARTICLE_DIMS, {"TILE": "n32x8/f4x4", "STAGE": "d2/tma"}, {}),
)


def _run_with_knobs_and_env(graph, inputs, out_name: str, knobs: dict, env: dict, monkeypatch) -> np.ndarray:
    """Variant of ``_run_with_knobs`` that also stamps extra env vars
    (used for opt-out flags like ``DEPLODOCK_INTERLEAVE_LOADS``)."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return _run_with_knobs(graph, inputs, out_name, knobs, monkeypatch)


@requires_cuda
@pytest.mark.parametrize(
    ("label", "dims", "knobs", "env"),
    _MASKED_TILE_CONFIGS,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_masked_tile_accuracy_configs(label: str, dims: dict, knobs: dict, env: dict, monkeypatch):  # noqa: ARG001 — ``label`` is the test id
    """End-to-end accuracy regression for the scalar fp32 SGEMM tile geometries from
    the matmul-optimization blog posts. Each row pins a tile that once emitted a
    wrong-output kernel; a failure flags a regression in (a) ``Knob.narrow``
    authoritative pin semantics, (b) the masked-tile (FM=26 / FN=26) ceil-div grid +
    boundary-Cond codegen, (c) the multi-axis composite-stride staging round-trip,
    (d) the ``095_interleave_loads`` opt-out, or (e) the scalar-tile TMA transport
    (``*_tma`` rows — the unswizzled ``cp.async.bulk.tensor`` deposit + plain-``Load``
    consumer)."""
    graph, input_shapes, (out_name, _) = _build_2d_matmul_graph(dims)
    inputs = _random_inputs(input_shapes, seed=42)
    ref = _reference(graph, inputs, out_name)
    forced = _run_with_knobs_and_env(graph, inputs, out_name, knobs, env, monkeypatch)
    _assert_match(forced, ref)
