"""CUDA accuracy regressions for matmul shapes with FN > 1.

Each test pins the autotune knob bundle that used to fault on the
register-blocked GEMM builder's path (deleted) and confirms the
per-cell shape + Kernel-IR replicator + ``dedup_replicated`` pipeline
reproduces the same accuracy. The pinned shapes exercise the smem
vectorize / fp16 pack / fused-prologue intersections that produced CUDA
runtime hangs and silently-wrong vector reads on specific autotune
variants.

Skipped without CUDA.
"""

from __future__ import annotations

import numpy as np

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp

from ..conftest import requires_cuda


def _random(shape: tuple[int, ...], *, seed: int = 0, scale: float = 1.0, dtype=np.float32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(dtype)


def _reference(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from deplodock.compiler.backend.numpy import NumpyBackend

    be = NumpyBackend()
    return be.run(be.compile(graph), input_data=inputs)[0].outputs


def _run_cuda(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    be = CudaBackend()
    return be.run(be.compile(graph), input_data=inputs)[0].outputs


def _assert_close(out: np.ndarray, ref: np.ndarray, *, atol_rel: float = 0.05, atol_min: float = 1e-3) -> None:
    """Tolerance scales with the reference peak — matmul reductions over
    K elements drift by ~K·eps on f32, the blocked-vs-per-cell difference
    is well below that floor."""
    assert out.shape == ref.shape, f"shape mismatch {out.shape} vs {ref.shape}"
    assert np.all(np.isfinite(out)), "output has non-finite values"
    peak = float(np.max(np.abs(ref)))
    atol = max(atol_min, atol_rel * peak)
    np.testing.assert_allclose(out, ref, atol=atol, rtol=atol_rel)


def _pin_knobs(monkeypatch, **knobs) -> None:
    for key, value in knobs.items():
        monkeypatch.setenv(f"DEPLODOCK_{key}", str(value))


# ---------------------------------------------------------------------------
# Multi-dim staged-smem vectorize alignment (FN=3 hang)
# ---------------------------------------------------------------------------


@requires_cuda
def test_blocked_matmul_postmul_fn3_no_misalign(monkeypatch):
    """``BK=64 BM=32 BN=32 BR=1 FM=1 FN=3 SPLITK=1 STAGE=111`` on a matmul
    chained into a post-multiply previously hung the kernel ("did not
    complete within 1000 ms"). Root cause: the per-cell + replicator
    pipeline produces multi-dim staged smem ``[K, N_t, N_r]`` with FN=3
    innermost. ``050_vectorize_loads`` packed two cells into one
    ``float2`` reinterpret_cast on a last-dim constant-anchor alignment
    check that didn't see the stride-3 base address ``a3 · FN`` — half
    the threads read at byte 12, not 8-byte aligned for float2, and the
    device stalled on the misaligned access (no exception, just stuck).

    The fix walks back into the Source's innermost cache-axis extent
    and refuses to vectorize when it isn't a multiple of the pack count.

    Shape: ``matmul(a@b) * scalar`` with M=32, K=64, N=96 (clean-divisor
    96 = BN·FN under BN=32 FN=3); the chained ``mul`` is what gives the
    planner a STAGE=111 enumeration with all three buffers stage-able.
    """
    _pin_knobs(monkeypatch, BK=64, BM=32, BN=32, BR=1, FM=1, FN=3, SPLITK=1, STAGE=111)

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (32, 64)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (64, 96)), node_id="b")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("s", (1,)), node_id="s")  # broadcast scalar
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("ab", (32, 96)), node_id="ab")
    g.add_node(op=ElementwiseOp("multiply"), inputs=["ab", "s"], output=Tensor("o", (32, 96)), node_id="o")
    g.inputs = ["a", "b", "s"]
    g.outputs = ["o"]

    inputs = {
        "a": _random((32, 64), seed=1),
        "b": _random((64, 96), seed=2),
        "s": np.array([1.5], dtype=np.float32),
    }
    ref = _reference(g, inputs)["o"]
    out = _run_cuda(g, inputs)["o"]
    _assert_close(out, ref)


# ---------------------------------------------------------------------------
# fp16 half2 alignment on odd-stride N
# ---------------------------------------------------------------------------


@requires_cuda
def test_blocked_matmul_fp16_odd_stride_n_no_misalign(monkeypatch):
    """fp16 matmul whose staged smem slab has an odd innermost cache-axis
    extent (FN=3 here) used to fault with ``CUDA_ERROR_MISALIGNED_ADDRESS``
    because ``050_vectorize_loads`` treated ``n=2 fp16`` (``__half2``) as
    a freebie: the TYPE is 4-byte aligned, but reinterpreting an fp16
    pointer at an odd-element offset still misses the alignment. Two
    cells off an FN=3 base land at consecutive odd offsets — half the
    threads read at byte 6, not 4-byte-aligned for ``__half2``. The fix
    walks back into the Source's innermost cache-axis extent and refuses
    to vectorize when it isn't a multiple of n.

    Shape: M=32, K=64, N=96 (clean divisor 96 = BN·FN under BN=32 FN=3).
    The pinned (BK=64 BM=32 BN=32 FM=1 FN=3 SPLITK=1 BR=1 STAGE=111) is
    the variant from the FN=3 hang reproducer in ``370e6090`` — only the
    matmul-chained-into-mul gives the planner a STAGE=111 enumeration,
    so the test multiplies the matmul output by a scalar broadcast.
    """
    from deplodock.compiler import dtype as _dt  # noqa: PLC0415

    f16_dt = _dt.get("f16")
    f16 = np.dtype(np.float16)
    _pin_knobs(monkeypatch, BK=64, BM=32, BN=32, BR=1, FM=1, FN=3, SPLITK=1, STAGE=111)

    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (32, 64), f16_dt), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (64, 96), f16_dt), node_id="b")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("s", (1,), f16_dt), node_id="s")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("ab", (32, 96), f16_dt), node_id="ab")
    g.add_node(op=ElementwiseOp("multiply"), inputs=["ab", "s"], output=Tensor("o", (32, 96), f16_dt), node_id="o")
    g.inputs = ["a", "b", "s"]
    g.outputs = ["o"]

    inputs = {
        "a": _random((32, 64), seed=3, scale=0.1, dtype=f16),
        "b": _random((64, 96), seed=4, scale=0.1, dtype=f16),
        "s": np.array([1.0], dtype=f16),
    }
    ref = _reference(g, inputs)["o"]
    out = _run_cuda(g, inputs)["o"]
    _assert_close(out, ref, atol_rel=0.05, atol_min=5e-3)


# ---------------------------------------------------------------------------
# Fused RMSNorm + linear: blocked-prologue path (M5)
# ---------------------------------------------------------------------------


@requires_cuda
def test_fused_rmsnorm_linear_blocked_prologue(monkeypatch):
    """Fused RMSNorm + linear at lm_head-style knobs (FN=32, BK=64,
    SPLITK=1, BR=1, FM=1) used to hit the 2 s NVRTC compile budget on
    the legacy per-cell path, because the old blocked-GEMM builder
    excluded fused-prologue shapes. Without it, each of the 32 register
    cells re-emitted the per-K-element normalization
    ``v_k = x[m,k] · inv_rms · norm_weight[k]`` inside its own unrolled
    K loop — ~3.7 s to compile a ~530-line function with 32 live
    accumulators.

    The current per-cell + replicator + ``dedup_replicated`` pipeline
    folds the N-invariant prologue chain back into one body-level copy
    (mean reduce + rsqrt + ``v6 = norm_weight · v5`` computed once per K
    iter), with per-cell weight Load + multiply + Accum into persistent
    acc_i registers — compiles in budget and runs correctly.

    Shape: M=2 (tiny batch — fp32 keeps the accumulator drift small),
    K=1024 (RMSNorm normalization range), N=4096 (BN=128 · FN=32 clean
    divisor — matches the failing tune log without needing the full
    lm_head vocab=151669, which would also work but compiles slower in
    this CI-friendly test).
    """
    _pin_knobs(monkeypatch, BK=64, BM=1, BN=128, BR=1, FM=1, FN=32, SPLITK=1)

    M, K, N = 2, 1024, 4096
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (M, K)), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("wn", (K,)), node_id="wn")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("wl", (N, K)), node_id="wl")
    g.add_node(op=RmsNormOp(eps=1e-6), inputs=["x", "wn"], output=Tensor("xn", (M, K)), node_id="xn")
    g.add_node(op=LinearOp(), inputs=["xn", "wl"], output=Tensor("o", (M, N)), node_id="o")
    g.inputs = ["x", "wn", "wl"]
    g.outputs = ["o"]

    inputs = {
        "x": _random((M, K), seed=5),
        "wn": _random((K,), seed=6, scale=0.1),
        "wl": _random((N, K), seed=7, scale=0.02),  # scaled so output stays bounded
    }
    # Compute reference BEFORE backend.compile (which mutates ops in place).
    ref = _reference(g, inputs)["o"]

    # Structural check: the per-cell + replicator + dedup pipeline emits
    # ONE smem allocation for the RMSNorm input (``x_smem``) — both the
    # mean reduce and the matmul body share it. A regression that wraps
    # the matmul body in its own RegisterTile(N_r) at a different staging
    # context would force a SECOND smem allocation (``x_smem_1``). Detect
    # by counting distinct smem decls for the x buffer.
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    backend = CudaBackend()
    compiled = backend.compile(g)
    cuda_ops = [n.op for n in compiled.nodes.values() if isinstance(n.op, CudaOp)]
    assert cuda_ops, "expected a CudaOp in the lowered graph"
    cuda_src = "\n".join(op.kernel_source for op in cuda_ops)
    # Match ``__shared__`` plus the buffer name, with or without an
    # ``__align__(N)`` directive in between — the materializer stamps
    # ``__align__(128)`` on TMA-targeted slabs so the regex tolerates either form.
    import re as _re  # noqa: PLC0415

    x_smem_decls = len(_re.findall(r"__shared__\s+(?:__align__\([0-9]+\)\s+)?float\s+x_smem\b", cuda_src))
    assert x_smem_decls == 1, (
        f"expected 1 ``__shared__ float x_smem`` decl (per-cell shares the staging); "
        f"got {x_smem_decls} — a regression opening its own smem context inside RegisterTile(N_r) "
        f"would force a second allocation."
    )

    # Kernel-size check: a regression that duplicates the prologue chain
    # inside each register cell's scope inflates the body. With FN=32 and
    # a ~5-stmt v_k chain (Load x, mul by inv_rms, Load norm_weight,
    # mul, ...), the duplicated body lands at ~330 lines; the deduped
    # body at ~210. The renderer prepends the TMA prelude (~75 lines of
    # mbarrier / cp_async_bulk_tensor helpers) when TMA descriptors are
    # present, so the deduped+TMA total is ~286 and a regression would
    # land ~406. Threshold at 360 catches the regression with margin.
    cu_lines = cuda_src.count("\n")
    assert cu_lines < 360, (
        f"rendered kernel is {cu_lines} lines — should produce ~286 with TMA prelude; "
        f"a regression that fails to dedup the N-invariant prologue chain inflates it to ~400+."
    )

    out = backend.run(compiled, input_data=inputs)[0].outputs["o"]
    _assert_close(out, ref, atol_rel=0.05, atol_min=1e-3)


# The FN > 1 blocked-matmul accuracy matrix that used to live here (pinned via legacy
# ``BN``/``BM``/``FM``/``FN``) is now the new-schema ``TILE`` codec matrix in
# ``test_matmul_tile_coverage`` (register replication accuracy + structure, static AND dynamic).
