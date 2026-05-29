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
# - ``FN=4`` (multi-axis collapse): the cache decomposition
#   ``(BN_thread × FN_register)`` makes B's addressing
#   ``TemplateAddressing``, ineligible for TMA's affine box copy.
#   Fixed via the new ``DEPLODOCK_AFFINE_COLLAPSE=1`` opt-in in
#   ``020_stage_inputs._derive_slab``: composite-stride detection
#   admits multi-axis-per-source-dim as ``AffineAddressing`` so the
#   materializer's existing ``box_per_dim`` collapse can emit a
#   2D TMA box of ``(BK, BN_total)``.

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


# (label, knobs, env extras). The label is what shows up in the test
# id; ``env`` carries non-knob DEPLODOCK_* vars (today only the
# ``AFFINE_COLLAPSE`` opt-in for the multi-axis-per-dim cases).
_ARTICLE_REPRODUCTION: tuple[tuple[str, dict, dict], ...] = (
    # cab3c83e: BM=8 outside _TUNE_AXIS_CHOICES — pin must be honored.
    ("bm8_pin_outside_hints", {"BM": 8, "BN": 32, "FM": 1, "FN": 1, "BK": 32, "SPLITK": 1, "BR": 1}, {}),
    # cab3c83e: FM=26 non-divisor of E_M/BM=256 — overhang/masked tile.
    ("fm26_overhang_masked", {"BM": 8, "BN": 32, "FM": 26, "FN": 1, "BK": 32, "SPLITK": 1, "BR": 1}, {}),
    # 8940dc25: USE_TMA=1 forces TMA on the matmul A+B bundle that the
    # eligibility check used to reject as multi-source. Tile sized so
    # ``KernelOp.validate``'s smem cap (~99 KB on sm_120) is honored.
    ("multisrc_tma_fm1_fn1", {"BM": 8, "BN": 32, "FM": 1, "FN": 1, "BK": 32, "SPLITK": 1, "BR": 1, "USE_TMA": 1}, {}),
    # Affine-collapse opt-in: FN=4 multi-axis cache on N. With the flag
    # off this falls back to cp.async (FN>1 → TemplateAddressing).
    (
        "affine_collapse_fn4",
        {"BM": 8, "BN": 32, "FM": 1, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1, "USE_TMA": 1},
        {"DEPLODOCK_AFFINE_COLLAPSE": "1"},
    ),
    # Affine-collapse on BOTH axes: FM=4 and FN=4 — the article's
    # FM×FN > 1 register tile, with multi-source TMA on A and B.
    (
        "affine_collapse_fm4_fn4",
        {"BM": 8, "BN": 32, "FM": 4, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1, "USE_TMA": 1},
        {"DEPLODOCK_AFFINE_COLLAPSE": "1"},
    ),
    # 095_interleave_loads opt-out — flat-LDS layout (every Load at
    # the top of the cluster). Locks in that disabling the pass via
    # env still produces correct output, so a future re-enabling of
    # the legacy path stays safe.
    (
        "interleave_loads_disabled",
        {"BM": 8, "BN": 32, "FM": 4, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1, "USE_TMA": 1},
        {"DEPLODOCK_AFFINE_COLLAPSE": "1", "DEPLODOCK_INTERLEAVE_LOADS": "0"},
    ),
)


def _run_with_knobs_and_env(graph, inputs, out_name: str, knobs: dict, env: dict, monkeypatch) -> np.ndarray:
    """Variant of ``_run_with_knobs`` that also stamps extra env vars
    (used here for ``DEPLODOCK_AFFINE_COLLAPSE``)."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return _run_with_knobs(graph, inputs, out_name, knobs, monkeypatch)


@requires_cuda
@pytest.mark.parametrize(
    ("label", "knobs", "env"),
    _ARTICLE_REPRODUCTION,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_article_reproduction_configs(label: str, knobs: dict, env: dict, monkeypatch):  # noqa: ARG001 — ``label`` is the test id
    """End-to-end accuracy regression for the configurations surfaced
    while reproducing the article's TMA SGEMM kernel via knob pinning.
    Each row exercises one previously-broken code path; a failure here
    indicates a regression in: (a) ``Knob.narrow`` authoritative pin
    semantics, (b) ``_enumerate_cartesian``'s per-(fm,fn) overhang
    handling, (c) ``050_use_tma`` multi-source-split + 128 B inner
    alignment, or (d) ``020_stage_inputs._derive_slab`` composite-stride
    affine collapse + ``_stage_expand`` decode."""
    graph, input_shapes, (out_name, _) = _build_2d_matmul_graph(_ARTICLE_DIMS)
    inputs = _random_inputs(input_shapes, seed=42)
    ref = _reference(graph, inputs, out_name)
    forced = _run_with_knobs_and_env(graph, inputs, out_name, knobs, env, monkeypatch)
    _assert_match(forced, ref)
