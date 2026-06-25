"""CUDA accuracy regression for specific planner knob tuples.

Where ``test_tune_accuracy`` lets the search wander and checks the
*picked* variant, this file pins each (BN, BM, FM, FN, BK, SPLITK, BR)
to a configuration that previously emitted a wrong-output kernel and
confirms the lowered kernel matches the numpy-backend reference within
fp32 tolerance.

Knob pinning rides on the existing ``DEPLODOCK_KNOBS="K1=V1,..."``
env-var mechanism (see ``deplodock/compiler/pipeline/knob.py`` —
``apply_knobs_env`` splats the aggregate into per-knob
``DEPLODOCK_<K>=V`` vars at import time, and ``Knob.narrow`` intersects
the planner's candidate lists with the pinned values inside
``010_partition_loops._enumerate_cartesian`` so only matching
``TileParams`` are enumerated).

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

# Shapes match the user's offline scan.
_MATMUL_DIMS = {"M": 32, "K": 128, "N": 64}
_GATED_DIMS = {"S": 32, "H": 128, "I": 256}

# Broken (BN, BM, FM, FN) tiles + a representative BK. The full
# matrix in the user's offline scan crossed each tile with 6 BKs × 3
# STAGE codes; BK varies the K-loop chunking and STAGE varies the
# producer-consumer pipeline encoding, neither changes the underlying
# single-CTA-vs-multi-CTA dispatch that was wrong. One BK is enough
# to lock the regression.
_BROKEN_MATMUL: tuple[dict, ...] = (
    {"BN": 16, "BM": 32, "FM": 1, "FN": 4, "BK": 64, "SPLITK": 1, "BR": 1},
    {"BN": 32, "BM": 16, "FM": 2, "FN": 2, "BK": 64, "SPLITK": 1, "BR": 1},
    {"BN": 32, "BM": 32, "FM": 1, "FN": 2, "BK": 64, "SPLITK": 1, "BR": 1},
    {"BN": 64, "BM": 16, "FM": 2, "FN": 1, "BK": 64, "SPLITK": 1, "BR": 1},
)
_BROKEN_GATED: tuple[dict, ...] = (
    {"BN": 16, "BM": 32, "FM": 1, "FN": 16, "BK": 64, "SPLITK": 1, "BR": 1},
    {"BN": 32, "BM": 16, "FM": 2, "FN": 8, "BK": 64, "SPLITK": 1, "BR": 1},
    {"BN": 32, "BM": 32, "FM": 1, "FN": 8, "BK": 64, "SPLITK": 1, "BR": 1},
    {"BN": 64, "BM": 16, "FM": 2, "FN": 4, "BK": 64, "SPLITK": 1, "BR": 1},
)


def _format_knobs(knobs: dict) -> str:
    """Render a knob dict as ``"K1=V1,K2=V2,..."`` for ``DEPLODOCK_KNOBS``."""
    return ",".join(f"{k}={v}" for k, v in knobs.items())


def _build_matmul_graph(dims: dict, mode: str = "static"):
    """``(1, M, K) @ (K, N)``. ``mode='dynamic'`` makes the M axis symbolic
    (``Dim('seq_len')``); the returned ``input_shapes`` stay concrete so the run
    feeds a real (1, M, K) array the symbolic kernel resolves ``seq_len`` from."""
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp

    M, K, N = dims["M"], dims["K"], dims["N"]
    Mg = dyn_M(mode, M)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (1, Mg, K)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (1, Mg, N)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    return g, {"a": (1, M, K), "b": (K, N)}, ("c", (1, M, N))


def _build_gated_graph(dims: dict, mode: str = "static"):
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp
    from deplodock.compiler.ir.tensor.ir import ElementwiseOp

    S, H, Inter = dims["S"], dims["H"], dims["I"]
    Sg = dyn_M(mode, S)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, Sg, H)), node_id="x")
    g.add_node(InputOp(), [], Tensor("wg", (H, Inter)), node_id="wg")
    g.add_node(InputOp(), [], Tensor("wu", (H, Inter)), node_id="wu")
    g.add_node(MatmulOp(), ["x", "wg"], Tensor("mg", (1, Sg, Inter)), node_id="mg")
    g.add_node(MatmulOp(), ["x", "wu"], Tensor("mu", (1, Sg, Inter)), node_id="mu")
    g.add_node(ElementwiseOp("silu"), ["mg"], Tensor("sg", (1, Sg, Inter)), node_id="sg")
    g.add_node(ElementwiseOp("multiply"), ["sg", "mu"], Tensor("y", (1, Sg, Inter)), node_id="y")
    g.inputs, g.outputs = ["x", "wg", "wu"], ["y"]
    return g, {"x": (1, S, H), "wg": (H, Inter), "wu": (H, Inter)}, ("y", (1, S, Inter))


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


@requires_cuda
@pytest.mark.parametrize("knobs", _BROKEN_MATMUL, ids=lambda k: f"BN{k['BN']}_BM{k['BM']}_FM{k['FM']}_FN{k['FN']}")
def test_matmul_single_cta_f_replicated(knobs: dict, shape_mode, monkeypatch):
    """matmul (1, 32, 128) @ (128, 64) — single-CTA + F-replicated tile. The
    pinned config must produce the same correct output whether M is baked
    (static) or symbolic (dynamic ``seq_len``, masked path) — the numpy
    reference is the static graph; the forced CUDA run uses the mode's graph."""
    static_graph, input_shapes, (out_name, _) = _build_matmul_graph(_MATMUL_DIMS)
    forced_graph, _, _ = _build_matmul_graph(_MATMUL_DIMS, mode=shape_mode)
    inputs = _random_inputs(input_shapes)
    ref = _reference(static_graph, inputs, out_name)
    forced = _run_with_knobs(forced_graph, inputs, out_name, knobs, monkeypatch)
    _assert_match(forced, ref)


@requires_cuda
@pytest.mark.parametrize("knobs", _BROKEN_GATED, ids=lambda k: f"BN{k['BN']}_BM{k['BM']}_FM{k['FM']}_FN{k['FN']}")
def test_gated_mlp_single_cta_f_replicated(knobs: dict, shape_mode, monkeypatch):
    """gated_mlp (1, 32, 128) → (1, 32, 256) — single-CTA + F-replicated tile,
    static and dynamic (symbolic ``seq_len``). Reference is the static graph."""
    static_graph, input_shapes, (out_name, _) = _build_gated_graph(_GATED_DIMS)
    forced_graph, _, _ = _build_gated_graph(_GATED_DIMS, mode=shape_mode)
    inputs = _random_inputs(input_shapes)
    ref = _reference(static_graph, inputs, out_name)
    forced = _run_with_knobs(forced_graph, inputs, out_name, knobs, monkeypatch)
    _assert_match(forced, ref)


# The #244 dynamic-tune wedge: a scalar (``MMA=0``) cooperative norm-reduce whose
# fp16 per-slot box is a single ``BK=32`` axis = 64 B — not a 128 B multiple, so
# under ``RING=2`` the second TMA ring slot lands at a 64 B offset. The 128 B slot
# check sized off the fp32 ``BYTES_PER_ELEM`` constant let the fp16 64 B slab
# through, the materializer left it unpadded, and ``cp.async.bulk.tensor`` faulted
# with ``CUDA_ERROR_MISALIGNED_ADDRESS`` → 1 s watchdog hang → bench_fail. The
# dtype-aware gate now declines TMA for this slab (→ cp.async). ``MMA=0`` forces
# the scalar tier; ``RING=2`` double-buffers so the slot offset matters.
_NORM_REDUCE_WEDGE_KNOBS = {"BM": 1, "BN": 128, "FM": 1, "FN": 1, "BK": 32, "SPLITK": 1, "BR": 1, "MMA": 0, "RING": 2}
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


# Staged MMA regression for ``plans/mma-smem-staging.md`` M5. These
# shapes pick a K-split (K_o > 1) under the planner's natural priors,
# placing a SerialTile(K_o) between the AtomTile and the StageBundle
# (which wraps SerialTile(K_i) > loads). The buggy pre-fix shape
# emitted the lowered Mma chain with the bundle hoisted ABOVE K_o, but
# the bundle's ``Source.origin`` references the K_o coord (e.g.
# ``a[a0*16, a5*1024]``), so the kernel failed to compile with
# ``identifier 'a5' is undefined``.
#
# ``test_matmul_mma``'s 16/64/128/256 squares all naturally pick
# K_o = 1 (full K fits in one stage); we need an asymmetric M/N/K
# shape with large K to force the K-split.
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

    monkeypatch.setenv("DEPLODOCK_MMA", "1")
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


# Scalar-FMA fp16 regression (prerequisite for the split-pipe GEMM, see
# plans/make-a-plan-for-sparkling-hamster.md). On cc>=9.0 the F16 atom is
# eligible whenever the K-loads are F16, so at >=512^3 the greedy compile
# *prefers* the tensor-core variant; ``DEPLODOCK_MMA=0`` is what forces the
# scalar register-tile FMA path (fp16 in -> fp32 accumulate -> fp16 out). The
# 512^3 shape is the smallest where MMA=0 is load-bearing (<=256^3 greedy picks
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


# Scalar-only knobs (no ATOM_KIND / WARP_SPECIALIZE). With DEPLODOCK_MMA unset
# these make the greedy compile pick the (unstaged-with-TMA=0) atom variant.
_SCALAR_F16_KNOBS = {"BM": 8, "BN": 32, "FM": 26, "FN": 4, "BK": 32, "SPLITK": 1, "TMA": 0, "STAGE": 11}


@requires_cuda
def test_scalar_matmul_f16(monkeypatch):
    """fp16 matmul forced through the scalar register-tile FMA path
    (``DEPLODOCK_MMA=0``); fp32 accumulate, verified vs the numpy backend.
    Guards against fp16 being re-routed to the tensor-core atom path and
    against the fp16 scalar render (``__half2float`` multiply, ``__float2half``
    store) regressing."""
    monkeypatch.setenv("DEPLODOCK_MMA", "0")
    g, inputs, ref = _build_f16_matmul_graph(512, 512, 512)
    forced = _run_with_knobs(g, inputs, "c", _SCALAR_F16_KNOBS, monkeypatch)
    peak = float(np.max(np.abs(ref.astype(np.float32))))
    atol = max(5e-1, 0.1 * peak)
    np.testing.assert_allclose(forced.astype(np.float32), ref.astype(np.float32), atol=atol, rtol=0.1)


def test_article_tma_sgemm_reproduction(monkeypatch):
    """The matmul-optimization blogs' hero kernel is a SCALAR fp32 SGEMM whose staged
    operands ride the **TMA** transport (``cp.async.bulk.tensor.2d`` double buffer) —
    the ``TM=26`` tile (``BM=8 BN=32 FM=26 FN=4 BK=32``) reaching ~106 % of cuBLAS at
    2048³. ``130_transport`` promotes any staged matmul with a ringable K loop, scalar
    as well as warp-tier; the scalar tier's plain-``Load`` consumer reads an unswizzled
    (``SwizzleMode.NONE``) deposit (``_slab._make_bundle``, keyed on ``Block.atom``)
    where the warp tier's ``ldmatrix`` reads a swizzled one. This asserts both that the
    staged bundles flip to ``StagePolicy.TMA`` and that the lowered kernel emits the
    ``cp.async.bulk.tensor`` box copy. Tile-level (inspects the staging policy + source).
    No CUDA needed."""
    from deplodock.compiler.context import Context
    from deplodock.compiler.ir.cuda.ir import CudaOp
    from deplodock.compiler.ir.tile.ir import StageBundle, StagePolicy, TileOp
    from deplodock.compiler.pipeline import KERNEL_PASSES, TILE_PASSES, Pipeline

    g, _, _ = _build_2d_matmul_graph(_ARTICLE_DIMS)
    for k, v in {"MMA": 0, "BM": 8, "BN": 32, "FM": 26, "FN": 4, "BK": 32, "SPLITK": 1, "STAGE": 11, "TMA": 1}.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", str(v))
    res = Pipeline.build(TILE_PASSES).run(g, ctx=Context.from_target((12, 0)))
    bundles = [s for n in res.nodes.values() if isinstance(n.op, TileOp) for s in n.op.body.iter() if isinstance(s, StageBundle)]
    assert bundles, "the staged scalar tile must synthesize at least one StageBundle"
    assert any(b.policy is StagePolicy.TMA for b in bundles), "scalar-tile TMA transport engaged"

    g2, _, _ = _build_2d_matmul_graph(_ARTICLE_DIMS)
    cuda = Pipeline.build([*KERNEL_PASSES, "lowering/cuda"]).run(g2, ctx=Context.from_target((12, 0)))
    src = "\n".join(n.op.kernel_source for n in cuda.nodes.values() if isinstance(n.op, CudaOp))
    assert "cp.async.bulk.tensor" in src, "scalar-tile TMA box copy emitted"


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
    for k, v in {"MMA": 0, "BM": 8, "BN": 32, "FM": 26, "FN": 4, "BK": 32, "SPLITK": 1, "STAGE": 11, "TMA": 1}.items():
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
    # Pin only the WARP-tier geometry (the over-ceiling register tile + atom-K chunk)
    # and leave STAGE unpinned — an explicit STAGE pin is authoritative (no budget
    # filter), but here we want the budget-aware filter to decline the over-budget
    # staging so the operands fall to the gmem-direct path. Scalar-tier knobs (BN/BM)
    # are deliberately NOT pinned: they are foreign to the warp tier and the strict
    # knob-pin validator (``_validate``) rejects them alongside a ``MMA=<kind>`` pin.
    for k, v in {"FM": 26, "FN": 4, "BK": 32, "SPLITK": 1, "TMA": 0}.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", str(v))
    # A DEPLODOCK_MMA=<kind> pin is authoritative (the planner drops the scalar
    # tier), so greedy MUST take the atom variant.
    monkeypatch.setenv("DEPLODOCK_MMA", "mma_m16n8k16_f16")
    compiled = CudaBackend().compile(g)  # no longer raises
    src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if isinstance(n.op, CudaOp))
    assert "dpl_mma_load_a_gmem" in src and "dpl_mma_load_b_gmem" in src, "unstaged operands not loaded gmem-direct"
    assert "mma.sync.aligned.m16n8k16" in src, "tensor-core path not taken (scalar fallback?)"


@requires_cuda
def test_unstaged_atom_mma_accuracy(monkeypatch):
    """The gmem-direct mma fragment load must produce CORRECT results — a wrong
    lane→element map silently corrupts the matmul. Force an unstaged tensor-core
    matmul (a full warp pin + ``STAGE=00`` stages nothing) and verify it matches
    the numpy reference. Guards the m16n8k16 fragment layout in the gmem helpers."""
    g, inputs, ref = _build_f16_matmul_graph(128, 128, 128)
    knobs = {"MMA": "mma_m16n8k16_f16", "WN": 2, "WM": 2, "FM": 2, "FN": 2, "BK": 2, "SPLITK": 1, "STAGE": "00"}
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
# NOTE: the article's defining optimization — **TMA** (``cp.async.bulk.tensor``) on
# the scalar SGEMM tile — is covered separately by ``test_article_tma_sgemm_reproduction``
# (the ``*_tma`` rows below also pin ``TMA=1``): ``130_transport`` promotes the scalar
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
# via plain SYNC cooperative loads; the ``*_tma`` rows pin ``TMA=1`` so the staged
# operands ride the ``cp.async.bulk.tensor`` double-buffer ring (the article's hero
# transport — ``130_transport`` now promotes the scalar tier, depositing the slab
# unswizzled for the plain-``Load`` consumer; see ``test_article_tma_sgemm_reproduction``
# for the tile-level check). The dead ``ASYNC_COPY`` / ``PIPELINE_STAGES`` knobs (whose
# passes were removed in the block-DAG rewrite) were dropped here.
_SMALL_DIMS = {"M": 512, "K": 512, "N": 512}
_MASKED_TILE_CONFIGS: tuple[tuple[str, dict, dict, dict], ...] = (
    # BM=8 outside _TUNE_AXIS_CHOICES — the pin must be honored authoritatively.
    ("bm8_pin_outside_hints", _ARTICLE_DIMS, {"BM": 8, "BN": 32, "FM": 1, "FN": 1, "BK": 32, "SPLITK": 1, "BR": 1}, {}),
    # FM=26 non-divisor of M/BM → a 208-row masked-M tile (ceil-div grid + per-row
    # boundary Cond on the Write).
    ("fm26_masked_m", _ARTICLE_DIMS, {"BM": 8, "BN": 32, "FM": 26, "FN": 1, "BK": 32, "SPLITK": 1, "BR": 1}, {}),
    # FN=4 multi-axis composite cache on N (BN_thread × FN_register on one source dim) —
    # the composite stride must round-trip through smem stage → revert-to-gmem.
    ("multi_axis_fn4", _ARTICLE_DIMS, {"BM": 8, "BN": 32, "FM": 1, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1}, {}),
    # Multi-axis composite on BOTH axes: FM=4 and FN=4 (the FM×FN > 1 register tile).
    ("multi_axis_fm4_fn4", _ARTICLE_DIMS, {"BM": 8, "BN": 32, "FM": 4, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1}, {}),
    # kernel/095_interleave_loads opt-out (flat-LDS layout) must still be correct.
    (
        "interleave_loads_disabled",
        _ARTICLE_DIMS,
        {"BM": 8, "BN": 32, "FM": 4, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1},
        {"DEPLODOCK_INTERLEAVE_LOADS": "0"},
    ),
    # The blogs' hero register tile: BM=8 BN=32 FM=26 FN=4 BK=32 → a 208×128 masked-M
    # tile (TM=26 ≈ 106 % of cuBLAS at 2048³ with the TMA transport, see *_tma rows below).
    ("article_tile_fm26_fn4", _ARTICLE_DIMS, {"BM": 8, "BN": 32, "FM": 26, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1}, {}),
    # Symmetric masked-N: FN=26 FM=4 on 512³ → a 320-col overhang (boundary Cond on N).
    ("fn26_masked_n", _SMALL_DIMS, {"BM": 8, "BN": 32, "FM": 4, "FN": 26, "BK": 32, "SPLITK": 1, "BR": 1}, {}),
    # Scalar-tile TMA: the staged operands ride the cp.async.bulk.tensor ring (unswizzled
    # deposit, plain-Load consumer). A clean (FM=4 FN=4) tile and the masked-M hero tile
    # (FM=26 FN=4 — the K pipeline hoisted above the boundary Cond, TMA OOB zero-fill).
    ("tma_clean_fm4_fn4", _ARTICLE_DIMS, {"BM": 8, "BN": 32, "FM": 4, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1, "TMA": 1}, {}),
    ("tma_article_fm26_fn4", _ARTICLE_DIMS, {"BM": 8, "BN": 32, "FM": 26, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1, "TMA": 1}, {}),
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
    consumer). ``test_article_tma_sgemm_reproduction`` covers the same TMA tile at the
    tile/kernel level (no CUDA)."""
    graph, input_shapes, (out_name, _) = _build_2d_matmul_graph(dims)
    inputs = _random_inputs(input_shapes, seed=42)
    ref = _reference(graph, inputs, out_name)
    forced = _run_with_knobs_and_env(graph, inputs, out_name, knobs, env, monkeypatch)
    _assert_match(forced, ref)
