"""End-to-end CUDA accuracy regressions for the four wrap-body lowering passes.

One module per pass would be four near-identical files: same matmul-graph
builder, same numpy reference, same allclose tolerances. We keep one test
function per pass (each with its own pin-pass-fires assertion) and share
the scaffolding.

Passes covered:

- ``040_use_ring_buffers`` — promotes SYNC ``StageBundle`` →
  ``buffer_count=2`` ring inside ``SerialTile``. Assertion:
  ``buffer_count >= 2`` (so 060 swapping the policy to ASYNC on sm_80+
  doesn't hide the regression).
- ``060_use_async_copy`` — promotes BUFFERED → ``policy=ASYNC``
  (cp.async transport). Assertions: ``StagePolicy.ASYNC`` bundle present
  AND rendered source contains ``cp.async.ca.shared.global`` +
  ``commit_group`` + ``wait_group 0`` (no silent fallback).
- ``070_pad_smem`` — BOOL ``PAD_SMEM`` fork; both polarities must run
  cleanly. Assertions: under ``PAD_SMEM=true`` at least one Source has
  non-zero pad; under ``PAD_SMEM=false`` no Source carries pad.
- TMA transport — cross-checks ``--target sm_80`` (cp.async) vs
  ``sm_90`` (cp.async.bulk.tensor with ``swizzle=NONE``). Skipped if the
  live device is below the requested cap.

All tests are skipped without CUDA. The shared ``matmul_graph`` builder
lives in ``tests/compiler/conftest.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.tile.ir import StageBundle, StagePolicy, TileOp

from ..conftest import matmul_graph, requires_cuda

# ---------- shared scaffolding ----------


def _random_inputs(m: int, k: int, n: int, *, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "a": rng.standard_normal((m, k), dtype=np.float32),
        "b": rng.standard_normal((k, n), dtype=np.float32),
    }


def _reference(graph: Graph, inputs: dict[str, np.ndarray]) -> np.ndarray:
    from deplodock.compiler.backend.numpy import NumpyBackend

    be = NumpyBackend()
    return be.run(be.compile(graph), input_data=inputs)[0].outputs["o"]


def _run_cuda(graph: Graph, inputs: dict[str, np.ndarray]) -> np.ndarray:
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    be = CudaBackend()
    return be.run(be.compile(graph), input_data=inputs)[0].outputs["o"]


def _assert_close(out: np.ndarray, ref: np.ndarray, *, rtol: float = 0.05, atol_floor: float = 1e-3) -> None:
    """Allclose with a peak-relative atol floor — matmul reductions over hundreds
    of elements need a few % atol to survive fp32 reduction-order noise."""
    assert out.shape == ref.shape
    assert np.all(np.isfinite(out)), "kernel output has non-finite values"
    peak = float(np.max(np.abs(ref)))
    atol = max(atol_floor, rtol * peak)
    np.testing.assert_allclose(out, ref, atol=atol, rtol=rtol)


def _tile_op(graph: Graph) -> TileOp | None:
    """Recompile through TILE_PASSES and return the lowered TileOp for ``o`` —
    the TileOp / KernelOp boundary is the only place where Stage policies are
    still visible; later passes lower them into CudaOp source."""
    from deplodock.compiler.pipeline import TILE_PASSES, Pipeline

    g2 = Pipeline.build(TILE_PASSES).run(graph)
    op = g2.nodes["o"].op
    return op if isinstance(op, TileOp) else None


def _kernel_source(graph: Graph) -> str:
    from deplodock.compiler.pipeline import CUDA_PASSES, Pipeline

    g2 = Pipeline.build(CUDA_PASSES).run(graph)
    return g2.nodes["o"].op.kernel_source


def _id(case):
    (m, k, n), knobs = case
    return f"M{m}_K{k}_N{n}_BK{knobs['BK']}"


# Knob pins that force the lowering-pass eligibility check to fire while
# keeping the resulting tile shape coherent (BN*FN | N, BM*FM | M).
_BASE_KNOBS = {"BN": 16, "BM": 16, "FM": 1, "FN": 1, "BK": 64, "SPLITK": 1, "BR": 1}


def _set_knobs(monkeypatch, knobs: dict[str, int]) -> None:
    for key, value in knobs.items():
        monkeypatch.setenv(f"DEPLODOCK_{key}", str(value))


# ---------- 040_use_ring_buffers ----------


_DOUBLE_BUFFER_CASES = (
    # K_o = 256/64 = 4 — standard double-buffered ring.
    ((128, 256, 128), _BASE_KNOBS),
    # K_o = 128/64 = 2 — minimum ping-pong (one alternation).
    ((64, 128, 64), _BASE_KNOBS),
    # K_o = 512/64 = 8 — deep ping-pong, surfaces phase-prepend bugs where the
    # rewritten Load index miscomputes for K_o > 2 iterations.
    ((64, 512, 64), _BASE_KNOBS),
)


@requires_cuda
@pytest.mark.parametrize("case", _DOUBLE_BUFFER_CASES, ids=_id)
def test_double_buffer_matmul_accuracy(case, monkeypatch):
    """Pin knobs so 040 fires (K_o >= 2); assert ``buffer_count >= 2`` on the
    lowered TileOp (covers BUFFERED + the 060-promoted ASYNC case)."""
    (m, k, n), knobs = case
    _set_knobs(monkeypatch, knobs)

    op = _tile_op(matmul_graph(m, k, n))
    assert op is not None, "graph did not lower to a TileOp"
    has_ring = any((isinstance(s, StageBundle) and s.buffer_count >= 2) for s in op.body.iter())
    assert has_ring, f"040_use_ring_buffers did not produce a double-buffered ring for M={m}, K={k}, N={n}"

    inputs = _random_inputs(m, k, n)
    ref = _reference(matmul_graph(m, k, n), inputs)
    out = _run_cuda(matmul_graph(m, k, n), inputs)
    _assert_close(out, ref)


# ---------- 060_use_async_copy ----------


_ASYNC_CASES = (
    # K_o = 256/64 = 4 — standard async ring.
    ((128, 256, 128), _BASE_KNOBS),
    # K_o = 128/64 = 2 — minimum async ping-pong.
    ((64, 128, 64), _BASE_KNOBS),
    # K_o = 192/64 = 3 — smallest K_o that wraps the ring buffer back to slot 0.
    # First broken case for the slot-aliasing race fixed by the trailing
    # AsyncWait sync in 080_pipeline_stages.
    ((64, 192, 64), _BASE_KNOBS),
    # K_o = 512/64 = 8 — deeper ring exercises repeated commit/wait pairs.
    ((64, 512, 64), _BASE_KNOBS),
)


@requires_cuda
@pytest.mark.parametrize("case", _ASYNC_CASES, ids=_id)
def test_async_copy_matmul_accuracy(case, monkeypatch):
    """Pin knobs so 060 fires; assert ``StagePolicy.ASYNC`` AND rendered source
    carries the cp.async issue / commit / wait triple (no silent fallback)."""
    (m, k, n), knobs = case
    _set_knobs(monkeypatch, knobs)

    op = _tile_op(matmul_graph(m, k, n))
    assert op is not None
    has_async = any((isinstance(s, StageBundle) and s.policy == StagePolicy.ASYNC) for s in op.body.iter())
    assert has_async, f"060_use_async_copy did not produce an AsyncBufferedStage for M={m}, K={k}, N={n}"

    src = _kernel_source(matmul_graph(m, k, n))
    assert "cp.async.ca.shared.global" in src, "kernel source missing cp.async issue"
    assert "cp.async.commit_group" in src, "kernel source missing cp.async commit"
    assert "cp.async.wait_group 0" in src, "kernel source missing cp.async wait"

    inputs = _random_inputs(m, k, n)
    ref = _reference(matmul_graph(m, k, n), inputs)
    out = _run_cuda(matmul_graph(m, k, n), inputs)
    _assert_close(out, ref)


# ---------- 070_pad_smem ----------


_PAD_SHAPES: tuple[tuple[int, int, int], ...] = ((128, 256, 128), (64, 128, 64))
_PAD_CASES = [(shape, pol) for shape in _PAD_SHAPES for pol in ("true", "false")]


def _pad_id(case):
    (m, k, n), polarity = case
    return f"M{m}_K{k}_N{n}_pad{polarity}"


def _pad_summary(op: TileOp) -> dict[str, tuple[int, ...]]:
    """Return ``{source_name: pad_tuple}`` for every BUFFERED / ASYNC source."""
    out: dict[str, tuple[int, ...]] = {}
    for s in op.body.iter():
        # 070 pads BUFFERED / ASYNC bundles (SYNC needs no rotation break, TMA
        # forbids pad). On sm_80+ 040's ring is realized as ASYNC.
        if isinstance(s, StageBundle) and s.policy in (StagePolicy.BUFFERED, StagePolicy.ASYNC):
            for stage in s.stages:
                for src in stage.sources:
                    out[src.name] = src.pad
    return out


@requires_cuda
@pytest.mark.parametrize("case", _PAD_CASES, ids=_pad_id)
def test_pad_smem_matmul_accuracy(case, monkeypatch):
    """Both ``PAD_SMEM`` polarities must match numpy — padding is a perf knob,
    never a correctness one. Also asserts the fork actually flipped the IR."""
    (m, k, n), polarity = case
    _set_knobs(monkeypatch, _BASE_KNOBS)
    monkeypatch.setenv("DEPLODOCK_PAD_SMEM", polarity)

    op = _tile_op(matmul_graph(m, k, n))
    assert op is not None
    pads = _pad_summary(op)
    if polarity == "true":
        assert any(p and any(p) for p in pads.values()), f"PAD_SMEM=true produced no padded source: {pads}"
    else:
        assert all(not p or not any(p) for p in pads.values()), f"PAD_SMEM=false leaked pad: {pads}"

    inputs = _random_inputs(m, k, n)
    ref = _reference(matmul_graph(m, k, n), inputs)
    out = _run_cuda(matmul_graph(m, k, n), inputs)
    _assert_close(out, ref)


# ---------- TMA transport (sm_80 cp.async vs sm_90+ cp.async.bulk.tensor) ----------


_TMA_M = _TMA_N = _TMA_K = 384
_TMA_TOL = 1e-2


def _live_capability() -> tuple[int, int]:
    from deplodock.compiler.target import compute_capability

    return compute_capability()


def _run_tma_matmul() -> np.ndarray:
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    rng = np.random.default_rng(seed=1234)
    a = rng.standard_normal((_TMA_M, _TMA_K), dtype=np.float32)
    b = rng.standard_normal((_TMA_K, _TMA_N), dtype=np.float32)
    backend = CudaBackend()
    compiled = backend.compile(matmul_graph(_TMA_M, _TMA_K, _TMA_N))
    out, _ = backend.run(compiled, input_data={"a": a.flatten().tolist(), "b": b.flatten().tolist()})
    return a, b, out.outputs["o"].reshape(_TMA_M, _TMA_N)


@requires_cuda
@pytest.mark.parametrize(
    ("target", "bn", "bm"),
    [
        ((8, 0), None, None),  # cp.async + pad_smem (default geometry)
        ((9, 0), None, None),  # TMA, swizzle=NONE (default geometry)
        ((9, 0), "32", "32"),  # TMA, swizzle=NONE, small slab (no perf gain, sanity)
    ],
    ids=["cpasync", "tma_no_swizzle", "tma_no_swizzle_bn32"],
)
def test_matmul_accuracy_across_tma_modes(monkeypatch, target, bn, bm):
    """Force the codegen path via ``set_target``. Tests requesting a cap above
    the live device are skipped — launch-time smem opt-in (derived from
    ``Context.max_dynamic_smem``) would exceed the actual device cap."""
    from deplodock.compiler import target as target_mod

    live = _live_capability()
    if live < target:
        pytest.skip(f"live device sm_{live[0]}{live[1]} < requested sm_{target[0]}{target[1]}")

    target_mod.set_target(target)
    if bn is not None:
        monkeypatch.setenv("DEPLODOCK_BN", bn)
        monkeypatch.setenv("DEPLODOCK_BM", bm)
    try:
        a, b, actual = _run_tma_matmul()
        expected = a @ b
        peak = float(np.max(np.abs(expected)))
        atol = max(1e-3, _TMA_TOL * peak)
        max_diff = float(np.max(np.abs(actual - expected)))
        assert max_diff < atol, f"max_diff={max_diff:.4f} >= tol={atol:.4f} (peak_eager={peak:.4f})"
    finally:
        target_mod.set_target(None)
