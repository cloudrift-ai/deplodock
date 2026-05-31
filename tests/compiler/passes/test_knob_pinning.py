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

from ..conftest import requires_cuda

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


def _build_matmul_graph(dims: dict):
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp

    M, K, N = dims["M"], dims["K"], dims["N"]
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (1, M, K)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (1, M, N)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    return g, {"a": (1, M, K), "b": (K, N)}, ("c", (1, M, N))


def _build_gated_graph(dims: dict):
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp
    from deplodock.compiler.ir.tensor.ir import ElementwiseOp

    S, H, Inter = dims["S"], dims["H"], dims["I"]
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, S, H)), node_id="x")
    g.add_node(InputOp(), [], Tensor("wg", (H, Inter)), node_id="wg")
    g.add_node(InputOp(), [], Tensor("wu", (H, Inter)), node_id="wu")
    g.add_node(MatmulOp(), ["x", "wg"], Tensor("mg", (1, S, Inter)), node_id="mg")
    g.add_node(MatmulOp(), ["x", "wu"], Tensor("mu", (1, S, Inter)), node_id="mu")
    g.add_node(ElementwiseOp("silu"), ["mg"], Tensor("sg", (1, S, Inter)), node_id="sg")
    g.add_node(ElementwiseOp("multiply"), ["sg", "mu"], Tensor("y", (1, S, Inter)), node_id="y")
    g.inputs, g.outputs = ["x", "wg", "wu"], ["y"]
    return g, {"x": (1, S, H), "wg": (H, Inter), "wu": (H, Inter)}, ("y", (1, S, Inter))


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
def test_matmul_single_cta_f_replicated(knobs: dict, monkeypatch):
    """matmul (1, 32, 128) @ (128, 64) — single-CTA + F-replicated tile."""
    graph, input_shapes, (out_name, _) = _build_matmul_graph(_MATMUL_DIMS)
    inputs = _random_inputs(input_shapes)
    ref = _reference(graph, inputs, out_name)
    forced = _run_with_knobs(graph, inputs, out_name, knobs, monkeypatch)
    _assert_match(forced, ref)


@requires_cuda
@pytest.mark.parametrize("knobs", _BROKEN_GATED, ids=lambda k: f"BN{k['BN']}_BM{k['BM']}_FM{k['FM']}_FN{k['FN']}")
def test_gated_mlp_single_cta_f_replicated(knobs: dict, monkeypatch):
    """gated_mlp (1, 32, 128) → (1, 32, 256) — single-CTA + F-replicated tile."""
    graph, input_shapes, (out_name, _) = _build_gated_graph(_GATED_DIMS)
    inputs = _random_inputs(input_shapes)
    ref = _reference(graph, inputs, out_name)
    forced = _run_with_knobs(graph, inputs, out_name, knobs, monkeypatch)
    _assert_match(forced, ref)


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


# Regressions for the article-reproduction work (see plans + git log
# for ``cab3c83e``, ``8940dc25``, the affine-collapse commit). Each row
# pins a configuration that used to fail before a specific fix:
#
# - ``BM=8``: out-of-set BM. ``_TUNE_AXIS_CHOICES = (1, 16, 32, 64,
#   128, 256)`` excludes 8, so the legacy ``Knob.narrow`` silently
#   dropped the pin and re-enumerated with defaults (``BM=1, BN=256``).
#   Fixed in ``cab3c83e``: pin authoritative regardless of hint
#   membership.
# - ``FM=26`` on ``BM=8, M=2048`` (208-row overhang): non-divisor FM
#   with overhang. ``_enumerate_cartesian`` skipped non-divisor FM
#   when ``m_overhang`` was off; same commit added per-(fm,fn)
#   overhang flip + masked-tile codegen.
# - ``USE_TMA=1`` + multi-source A+B (matmul): ``050_use_tma``
#   rejected multi-source bundles. Fixed in ``8940dc25``: pre-
#   eligibility split of multi-source Stages into N single-source
#   Stages so the materializer's N-stages-per-bundle TMA emit path
#   handles them.
# - ``FN=4`` (multi-axis composite): the cache decomposition
#   ``(BN_thread × FN_register)`` used to mark B's addressing
#   ``TemplateAddressing``, ineligible for TMA's affine box copy.
#   Fixed via the composite-stride check in
#   ``020_stage_inputs._derive_slab`` (admits multi-axis-per-
#   source-dim as ``AffineAddressing``) + the shared
#   ``affine_decode_per_dim`` helper used by every post-staging
#   consumer (``_stage_expand``, ``025_unify_sibling_stages``,
#   ``_source_decl_line``) so the composite stride round-trips
#   through smem-stage → revert-to-gmem → cuda emission.

# 2048×2048 matmul = the article's hero shape. We compare against a
# numpy reference at this size rather than the smaller (32, 128, 64)
# above so the per-tile-shape configurations actually exercise the
# multi-stage TMA/cp.async pipeline and the 128 B alignment math.
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


# (label, dims, knobs, env extras). The label is what shows up in
# the test id; ``dims`` is the M/K/N for the matmul; ``knobs`` is the
# per-knob ``DEPLODOCK_<K>`` pin set; ``env`` carries non-knob
# ``DEPLODOCK_*`` vars (today: the ``INTERLEAVE_LOADS`` opt-out).
#
# **Default dims.** ``_ARTICLE_DIMS`` (2048³) is the article's hero
# shape. Variants that exercise the cp.async + masked path use a
# smaller ``_CPASYNC_DIMS`` (512³) — the per-launch wall-clock
# watchdog in ``program.py`` is 1 s, and the 2048³ cp.async masked-M
# kernel hovers around that threshold under parallel-xdist
# contention. 512³ still triggers masked-M with FM=26 (208 × 2 cells
# leaves 96-row overhang) and is well inside the watchdog.
#
# **Staging-mode coverage.** Each masked-tile config (FM=26 → 208-row
# overhang, FN=26 → 832-col overhang) runs through every staging
# transport × pipelining combination that the article-tile knobs can
# select, so a regression in any of them trips a test:
#
#   - TMA pipelined (``USE_TMA=1 PIPELINE_STAGES=1``)
#   - TMA depth-1 (``USE_TMA=1 PIPELINE_STAGES=0``)
#   - cp.async pipelined (``USE_TMA=0 USE_ASYNC_COPY=1 PIPELINE_STAGES=1``)
#   - cp.async depth-1 (``USE_TMA=0 USE_ASYNC_COPY=1 PIPELINE_STAGES=0``)
#   - sync double-buffered (``USE_TMA=0 USE_ASYNC_COPY=0 PIPELINE_STAGES=1``)
#   - sync depth-1 (``USE_TMA=0 USE_ASYNC_COPY=0 PIPELINE_STAGES=0``)
#
# The B2 per-Load guard (``021_hoist_staged_loads_above_mask._guard_unsafe_loads``)
# handles the masked-tile branch on the TMA + cp.async + sync paths.
# The depth-1 / sync variants additionally exercise:
#
#   - ``050_vectorize_loads``' padded-stride alignment check (using
#     ``Source.alloc_extents[-1]`` so ``070_pad_smem``'s +1 pad
#     doesn't slip a misaligned ``float4`` reinterpret_cast past).
#   - ``100_materialize_tile.emit_bundle_producer``'s single
#     trailing wait on unpipelined multi-stage TMA bundles (mbarrier
#     ``arrive_count = len(stages)`` requires waiting once after ALL
#     stages arrive, not once per stage).
_CPASYNC_DIMS = {"M": 512, "K": 512, "N": 512}
_ARTICLE_REPRODUCTION: tuple[tuple[str, dict, dict, dict], ...] = (
    # cab3c83e: BM=8 outside _TUNE_AXIS_CHOICES — pin must be honored.
    ("bm8_pin_outside_hints", _ARTICLE_DIMS, {"BM": 8, "BN": 32, "FM": 1, "FN": 1, "BK": 32, "SPLITK": 1, "BR": 1}, {}),
    # cab3c83e: FM=26 non-divisor of E_M/BM=256 — overhang/masked tile.
    ("fm26_overhang_masked", _ARTICLE_DIMS, {"BM": 8, "BN": 32, "FM": 26, "FN": 1, "BK": 32, "SPLITK": 1, "BR": 1}, {}),
    # 8940dc25: USE_TMA=1 forces TMA on the matmul A+B bundle that the
    # eligibility check used to reject as multi-source. Tile sized so
    # ``KernelOp.validate``'s smem cap (~99 KB on sm_120) is honored.
    ("multisrc_tma_fm1_fn1", _ARTICLE_DIMS, {"BM": 8, "BN": 32, "FM": 1, "FN": 1, "BK": 32, "SPLITK": 1, "BR": 1, "USE_TMA": 1}, {}),
    # FN=4 multi-axis cache on N. Multi-axis-per-source-dim is now
    # admitted as ``AffineAddressing`` unconditionally (the old
    # ``DEPLODOCK_AFFINE_COLLAPSE`` opt-in was retired once the unify-
    # pass revert path was fixed to round-trip composite strides).
    (
        "multi_axis_affine_fn4",
        _ARTICLE_DIMS,
        {"BM": 8, "BN": 32, "FM": 1, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1, "USE_TMA": 1},
        {},
    ),
    # Multi-axis collapse on BOTH axes: FM=4 and FN=4 — the article's
    # FM×FN > 1 register tile, with multi-source TMA on A and B.
    (
        "multi_axis_affine_fm4_fn4",
        _ARTICLE_DIMS,
        {"BM": 8, "BN": 32, "FM": 4, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1, "USE_TMA": 1},
        {},
    ),
    # 095_interleave_loads opt-out — flat-LDS layout (every Load at
    # the top of the cluster). Locks in that disabling the pass via
    # env still produces correct output, so a future re-enabling of
    # the legacy path stays safe.
    (
        "interleave_loads_disabled",
        _ARTICLE_DIMS,
        {"BM": 8, "BN": 32, "FM": 4, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1, "USE_TMA": 1},
        {"DEPLODOCK_INTERLEAVE_LOADS": "0"},
    ),
    # Article's EXACT tile: ``BM=8 BN=32 FM=26 FN=4 BK=32`` produces a
    # 208×128 masked-M tile (M=2048 not a multiple of 208 → ceil-div
    # grid + per-row boundary Cond on the Write). Exercises the
    # masked-tile boundary Cond hoist in ``021_hoist_staged_loads_above_mask``:
    # every masked cell's K-pipeline runs (TMA elects a single issuer
    # thread; cp.async needs all threads to fetch their lane) and the
    # B2 per-Load guard (``_guard_unsafe_loads``) wraps the still-
    # un-staged gmem Loads that depend on the gated coord so masked
    # threads skip the OOB read. With multi-source TMA + multi-axis
    # composite (default on), this config lowers to a kernel
    # structurally byte-equivalent to ``_lower_matmul_tma_db``'s
    # TM=26 emit (104 cells/thread, 255 registers, 17 % occupancy)
    # and benches at ~97 % of cuBLAS on RTX 5090.
    (
        "article_exact_fm26_masked_tma",
        _ARTICLE_DIMS,
        {"BM": 8, "BN": 32, "FM": 26, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1, "USE_TMA": 1},
        {},
    ),
    # Same masked-M tile (FM=26 overhang) but routed through the
    # cp.async path (``USE_TMA=0`` skips 050_use_tma; ``USE_ASYNC_COPY=1``
    # promotes the BUFFERED bundle to ASYNC; ``PIPELINE_STAGES=1``
    # software-pipelines into prologue/main/epilogue). Verifies the
    # hoist's per-Load guard interacts correctly with cp.async's
    # cooperative-load semantics (all threads issue gmem reads via
    # cp.async.ca, then mbarrier waits on commit) — TMA's
    # OOB_FILL_NAN_REQUEST_ZERO_FMA fills OOB rows with zeros, while
    # cp.async has no equivalent hardware OOB handling, so the
    # per-Load Cond is the only thing keeping masked threads from
    # faulting at the overhang. 512³ shape: smaller dims keep the
    # kernel comfortably inside ``program.py``'s 1 s per-launch
    # watchdog under parallel-xdist contention; 208 × 2 cells +
    # 96-row overhang still exercises the masked-M code path.
    (
        "fm26_masked_cpasync",
        _CPASYNC_DIMS,
        {"BM": 8, "BN": 32, "FM": 26, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1, "USE_TMA": 0, "USE_ASYNC_COPY": 1, "PIPELINE_STAGES": 1},
        {},
    ),
    # Symmetric coverage: ``FN=26 FM=4`` → masked-N overhang on
    # 512³ (832-col tile, N=512 < 832 → 1 cell with 320-col
    # overhang). Boundary Cond gates the N coord. cp.async path
    # only — TMA is not eligible for masked-N at this shape
    # (multi-axis cache on N collides with the TMA box-copy gating).
    (
        "fn26_masked_cpasync",
        _CPASYNC_DIMS,
        {"BM": 8, "BN": 32, "FM": 4, "FN": 26, "BK": 32, "SPLITK": 1, "BR": 1, "USE_TMA": 0, "USE_ASYNC_COPY": 1, "PIPELINE_STAGES": 1},
        {},
    ),
    # Non-masked baseline on the cp.async path (FM=4 FN=4 divides
    # both 2048 and 512). Anchors that the cp.async lowering
    # produces correct results on the un-gated path too, so a
    # regression isolated to masked-tile codegen versus a regression
    # in cp.async in general can be told apart.
    (
        "fm4_fn4_cpasync",
        _CPASYNC_DIMS,
        {"BM": 8, "BN": 32, "FM": 4, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1, "USE_TMA": 0, "USE_ASYNC_COPY": 1, "PIPELINE_STAGES": 1},
        {},
    ),
    # Depth-1 staging coverage (``PIPELINE_STAGES=0`` keeps the bundle
    # at ``pipeline_depth=1``, skipping ``080_pipeline_stages``'s
    # prologue/main/epilogue peel). These three configs previously hung
    # with ``CUDA_ERROR_MISALIGNED_ADDRESS`` on the article tile due to
    # two bugs:
    #
    # 1. ``050_vectorize_loads`` collapsed 4 consecutive smem reads into
    #    a ``float4 reinterpret_cast`` based on the unpadded inner
    #    extent (FN=4 → 4-aligned), missing that ``070_pad_smem``'s
    #    bank-conflict ``pad=1`` makes the per-thread stride 5 floats =
    #    20 bytes — misaligned for any thread with ``a3 % 4 != 0``.
    #    Fix: check ``Source.alloc_extents[-1]`` (padded), not
    #    ``cache_axes[-1].extent``.
    #
    # 2. The unpipelined TMA emit (``100_materialize_tile.emit_tma_stage``)
    #    placed a trailing ``MbarrierWait`` after EVERY stage in a
    #    multi-source bundle. Mbarrier ``arrive_count`` is ``len(stages)
    #    = 2`` so the wait between stage 1 (arrived) and stage 2 (not
    #    yet arrived) blocks forever. Fix: move the wait+sync out of
    #    ``emit_tma_stage`` and emit it ONCE in ``emit_bundle_producer``
    #    after every member has issued its arrive.
    #
    # The cp.async/sync depth-1 paths hit bug 1; TMA depth-1 hits both.
    # Smaller dims (``_CPASYNC_DIMS``) match the cp.async tests' watchdog
    # margin.
    (
        "fm26_masked_tma_no_pipe",
        _CPASYNC_DIMS,
        {"BM": 8, "BN": 32, "FM": 26, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1, "USE_TMA": 1, "USE_ASYNC_COPY": 1, "PIPELINE_STAGES": 0},
        {},
    ),
    (
        "fm26_masked_cpasync_no_pipe",
        _CPASYNC_DIMS,
        {"BM": 8, "BN": 32, "FM": 26, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1, "USE_TMA": 0, "USE_ASYNC_COPY": 1, "PIPELINE_STAGES": 0},
        {},
    ),
    (
        "fm26_masked_sync_db",
        _CPASYNC_DIMS,
        {"BM": 8, "BN": 32, "FM": 26, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1, "USE_TMA": 0, "USE_ASYNC_COPY": 0, "PIPELINE_STAGES": 1},
        {},
    ),
    (
        "fm26_masked_sync_no_pipe",
        _CPASYNC_DIMS,
        {"BM": 8, "BN": 32, "FM": 26, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1, "USE_TMA": 0, "USE_ASYNC_COPY": 0, "PIPELINE_STAGES": 0},
        {},
    ),
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
    _ARTICLE_REPRODUCTION,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_article_reproduction_configs(label: str, dims: dict, knobs: dict, env: dict, monkeypatch):  # noqa: ARG001 — ``label`` is the test id
    """End-to-end accuracy regression for the configurations surfaced
    while reproducing the article's TMA SGEMM kernel via knob pinning.
    Each row exercises one previously-broken code path; a failure here
    indicates a regression in: (a) ``Knob.narrow`` authoritative pin
    semantics, (b) ``_enumerate_cartesian``'s per-(fm,fn) overhang
    handling, (c) ``050_use_tma`` multi-source-split + 128 B inner
    alignment, (d) ``020_stage_inputs._derive_slab`` composite-stride
    affine collapse + ``_stage_expand`` decode, or (e) the masked-tile
    boundary-Cond hoist (``021_hoist_staged_loads_above_mask``) + B2
    per-Load guard (``_guard_unsafe_loads``) interaction with the
    selected transport (TMA / cp.async)."""
    graph, input_shapes, (out_name, _) = _build_2d_matmul_graph(dims)
    inputs = _random_inputs(input_shapes, seed=42)
    ref = _reference(graph, inputs, out_name)
    forced = _run_with_knobs_and_env(graph, inputs, out_name, knobs, env, monkeypatch)
    _assert_match(forced, ref)
