"""CUDA accuracy regression tests for the tuner.

For a handful of small fused-launch patterns (matmul, rmsnorm, gated_mlp,
sdpa — same shapes as the perf suite at smaller dimensions), populate a
tmp :class:`SearchDB` via :func:`Pipeline.tune`, then re-execute the same
graph through :class:`CudaBackend` with the tuned DB and check the
output matches the rule-default reference within fp32 tolerance.

The tuner's own ``_bench_terminal`` only measures latency, so any
variant in the search space that produces wrong output gets cached as
"fast" and later loaded by ``deplodock run``. Each case below
exercises a code path that previously had a quietly-broken variant:

- ``matmul``: F-replicated body with SPLITK > 1 — atomic-add must
  fire on every per-cell Write. ``_compute_axis_dep_set`` had to
  descend into nested reduce Loops to see the Accum names.
- ``rmsnorm``: cooperative-K with F-replicated output cells emits two
  Accums in one reduce → two sibling Combines, which the materializer
  refused. Also: the cooperative SPLITK > 1 path had no cross-CTA
  aggregation so the planner must keep SPLITK=1 for that branch.
- ``gated_mlp``: matmul-style reduce with > 1 Accum feeding a
  non-linear post-reduce combine — atomic-add over partials doesn't
  hold here so the planner must keep SPLITK=1.
- ``sdpa``: multiple top-level stmts in the Tile body (Init / max-
  reduce / sum-reduce / output-Loop siblings) skipped the single-stmt
  wrapper descent in ``_rewrite_for_atomic_lift``, so atomic-add never
  fired for the AV matmul Write under SPLITK > 1.

Skipped without CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import requires_cuda

# Small-shape variants of the perf suite. Dimensions are picked so each
# planner branch (matmul, cooperative-K, multi-Accum matmul, multi-stmt
# kernel) still gets exercised but the tune sweep finishes in seconds.
# Sizes below ~32 trip a separate planner edge case (Tile with no
# BIND_THREAD axis when BN=BM=BR=1) that's out of scope here.
_CASES: tuple[tuple[str, dict], ...] = (
    # Pure matmul. K=128 + small M/N admits SPLITK > 1 + F-replication.
    ("matmul", {"M": 32, "K": 128, "N": 64}),
    # Cooperative-K reduce (K >= warp_size).
    ("rmsnorm", {"S": 64, "H": 256}),
    # Multi-Accum matmul (gate · up share K-reduce, silu+mul fuse into
    # the epilogue) — guards the multi-Accum SPLITK=1 gate.
    ("gated_mlp", {"S": 32, "H": 128, "I": 256}),
    # Two-kernel SDPA exercises the multi-stmt-scope atomic descent in
    # kernel 1. Small heads + seq keeps the second kernel manageable.
    ("sdpa", {"S": 64, "DH": 32, "HEADS": 4, "KV_HEADS": 2}),
)


def _build_graph(op: str, dims: dict):
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp, RmsNormOp, SdpaOp
    from deplodock.compiler.ir.tensor.ir import ElementwiseOp

    g = Graph()

    if op == "matmul":
        M, K, N = dims["M"], dims["K"], dims["N"]
        g.add_node(InputOp(), [], Tensor("a", (1, M, K)), node_id="a")
        g.add_node(InputOp(), [], Tensor("b", (K, N)), node_id="b")
        g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (1, M, N)), node_id="c")
        g.inputs, g.outputs = ["a", "b"], ["c"]
        return g, {"a": (1, M, K), "b": (K, N)}, ("c", (1, M, N))

    if op == "rmsnorm":
        S, H = dims["S"], dims["H"]
        g.add_node(InputOp(), [], Tensor("x", (1, S, H)), node_id="x")
        g.add_node(InputOp(), [], Tensor("w", (H,)), node_id="w")
        g.add_node(RmsNormOp(eps=1e-6), ["x", "w"], Tensor("y", (1, S, H)), node_id="y")
        g.inputs, g.outputs = ["x", "w"], ["y"]
        return g, {"x": (1, S, H), "w": (H,)}, ("y", (1, S, H))

    if op == "gated_mlp":
        S, H, Inter = dims["S"], dims["H"], dims["I"]
        g.add_node(InputOp(), [], Tensor("x", (1, S, H)), node_id="x")
        g.add_node(InputOp(), [], Tensor("wg", (H, Inter)), node_id="wg")
        g.add_node(InputOp(), [], Tensor("wu", (H, Inter)), node_id="wu")
        g.add_node(MatmulOp(), ["x", "wg"], Tensor("mg", (1, S, Inter)), node_id="mg")
        g.add_node(MatmulOp(), ["x", "wu"], Tensor("mu", (1, S, Inter)), node_id="mu")
        g.add_node(ElementwiseOp("silu"), ["mg"], Tensor("sg", (1, S, Inter)), node_id="sg")
        g.add_node(ElementwiseOp("multiply"), ["sg", "mu"], Tensor("y", (1, S, Inter)), node_id="y")
        g.inputs, g.outputs = ["x", "wg", "wu"], ["y"]
        return g, {"x": (1, S, H), "wg": (H, Inter), "wu": (H, Inter)}, ("y", (1, S, Inter))

    if op == "sdpa":
        S, DH, HEADS, KV = dims["S"], dims["DH"], dims["HEADS"], dims["KV_HEADS"]
        q_shape, k_shape, v_shape = (1, HEADS, S, DH), (1, KV, S, DH), (1, KV, S, DH)
        g.add_node(InputOp(), [], Tensor("q", q_shape), node_id="q")
        g.add_node(InputOp(), [], Tensor("k", k_shape), node_id="k")
        g.add_node(InputOp(), [], Tensor("v", v_shape), node_id="v")
        g.add_node(SdpaOp(is_causal=True), ["q", "k", "v"], Tensor("y", q_shape), node_id="y")
        g.inputs, g.outputs = ["q", "k", "v"], ["y"]
        return g, {"q": q_shape, "k": k_shape, "v": v_shape}, ("y", q_shape)

    raise ValueError(f"unknown op: {op}")


def _random_inputs(input_shapes: dict[str, tuple[int, ...]], seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal(shape, dtype=np.float32) for name, shape in input_shapes.items()}


@requires_cuda
@pytest.mark.parametrize("op,dims", _CASES, ids=[c[0] for c in _CASES])
def test_tuned_variant_matches_reference(op: str, dims: dict, tmp_path):
    """Reference output (rule defaults) vs tuned output must agree.

    The tuner records latencies but no accuracy gate, so a wrong-output
    variant that runs faster than the rule default would be silently
    picked. Re-running through ``CudaBackend(tune_db=...)`` after the
    tune sweep and comparing to the rule-default reference catches that
    drift before the bench-kernels-tuned suite would.
    """
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.pipeline import CUDA_PASSES, Pipeline
    from deplodock.compiler.pipeline.search import SearchDB, TuningSearch

    graph, input_shapes, (out_name, out_shape) = _build_graph(op, dims)
    inputs = _random_inputs(input_shapes)

    # Reference: rule defaults, no tune DB.
    ref_backend = CudaBackend()
    ref_compiled = ref_backend.compile(graph)
    ref_out = ref_backend.run(ref_compiled, input_data=inputs).outputs[out_name]
    assert np.all(np.isfinite(ref_out)), "reference output has non-finite values"

    # Tune: low patience to keep wall time bounded; backend records perf
    # so the DB has measurable rows, and ``_pick_best_candidate`` later
    # picks an entry with real timings, not the 1.0us stub.
    db_path = tmp_path / "tune.db"
    db = SearchDB(path=db_path)
    tune_backend = CudaBackend(bench_wall_timeout_s=30.0)
    # Drain the iterator. ``patience=5`` is plenty for these small
    # shapes — the search exhausts the queue or stops on stagnation
    # well before that for a 16x64 matmul.
    candidates = list(Pipeline.build(CUDA_PASSES).tune(graph, search=TuningSearch(patience=5), backend=tune_backend, db=db))
    assert candidates, "tune produced no terminal candidates"

    # Tuned re-run: same graph + inputs, but compile reads from the DB
    # so the picked knobs survive into the lowered kernels.
    tuned_backend = CudaBackend(tune_db=db_path)
    tuned_compiled = tuned_backend.compile(graph)
    tuned_out = tuned_backend.run(tuned_compiled, input_data=inputs).outputs[out_name]

    assert tuned_out.shape == ref_out.shape, f"shape mismatch: tuned {tuned_out.shape} vs ref {ref_out.shape}"
    assert np.all(np.isfinite(tuned_out)), "tuned output has non-finite values"

    # fp32 tolerance scaled to the reference peak. The split-K +
    # cooperative paths sum partial products in a non-deterministic
    # order vs the rule-default's single-CTA pairwise sum, so the
    # max-element drift can be a few percent of peak — match the same
    # 5%-of-peak budget ``deplodock run --bench`` uses.
    peak = float(np.max(np.abs(ref_out)))
    atol = max(1e-3, 0.05 * peak)
    np.testing.assert_allclose(tuned_out, ref_out, atol=atol, rtol=0.05)
