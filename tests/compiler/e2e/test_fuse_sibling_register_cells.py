"""End-to-end backend-accuracy test for ``012_fuse_sibling_register_cells``.

Pins the autotune knob bundle from the Qwen3-Embedding-0.6B
``k_linear_mean_reduce`` variant whose original 6+ s ``cicc -O1`` compile
(``BK=64, BM=1, BN=64, FM=1, FN=64``) timed out the autotune watchdog, runs it
through ``CudaBackend``, and confirms the rendered kernel is now small enough to
compile within budget while still matching the numpy reference.
"""

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import LinearOp, RmsNormOp

from ..conftest import requires_cuda

# ---------------------------------------------------------------------------
# End-to-end on the Qwen3-Embedding-0.6B linear+mean-reduce failing variant
# ---------------------------------------------------------------------------


@requires_cuda
def test_qwen_lmhead_variant_compiles_within_budget(monkeypatch):
    """The previously-failing ``BK=64, BM=1, BN=64, FM=1, FN=64`` variant of
    ``k_linear_mean_reduce`` on Qwen3-Embedding-0.6B used to take 5–6 s
    under ``cicc -O1`` because the RMSNorm prologue duplicated 64×. With
    the new sibling-Cond fuser landed, the same source folds to a single
    body-level RMSNorm chain + 64 short per-cell guarded multiplies — the
    rendered kernel should comfortably fit in the autotune's 2 s compile
    budget. We pin a smaller M+N here so the test stays CI-friendly; the
    structural pattern is identical."""
    for key, value in {"BK": "64", "BM": "1", "BN": "64", "BR": "1", "FM": "1", "FN": "64", "SPLITK": "1", "STAGE": "1"}.items():
        monkeypatch.setenv(f"DEPLODOCK_{key}", value)

    # M=2 (tiny batch), K=1024 (RMSNorm range), N=64 × 64 + 3 = 4099 — N is
    # deliberately a non-multiple of BN·FN to trigger the masked-overhang
    # Conds whose duplicates this pass folds.
    M, K, N = 2, 1024, 4099
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (M, K)), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("wn", (K,)), node_id="wn")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("wl", (N, K)), node_id="wl")
    g.add_node(op=RmsNormOp(eps=1e-6), inputs=["x", "wn"], output=Tensor("xn", (M, K)), node_id="xn")
    g.add_node(op=LinearOp(), inputs=["xn", "wl"], output=Tensor("o", (M, N)), node_id="o")
    g.inputs = ["x", "wn", "wl"]
    g.outputs = ["o"]

    rng = np.random.default_rng(seed=11)
    inputs = {
        "x": rng.standard_normal((M, K), dtype=np.float32).astype(np.float32),
        "wn": (rng.standard_normal((K,), dtype=np.float32) * 0.1).astype(np.float32),
        "wl": (rng.standard_normal((N, K), dtype=np.float32) * 0.02).astype(np.float32),
    }

    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from deplodock.compiler.backend.numpy import NumpyBackend  # noqa: PLC0415
    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    ref = NumpyBackend().run(NumpyBackend().compile(g), input_data=inputs)[0].outputs["o"]

    backend = CudaBackend()
    compiled = backend.compile(g)
    cuda_ops = [n.op for n in compiled.nodes.values() if isinstance(n.op, CudaOp)]
    cuda_src = "\n".join(op.kernel_source for op in cuda_ops)
    n_lines = cuda_src.count("\n")
    # Pre-fix this was ~1014 lines on the layered MLP kernel (compile 5–6 s).
    # Post-fix the same shape lands at ~720 lines or less. Threshold 850
    # catches a regression with margin but doesn't crack down on small
    # post-fix codegen drift.
    assert n_lines < 850, f"rendered kernel is {n_lines} lines — regression: invariant prefix is no longer being hoisted"

    out = backend.run(compiled, input_data=inputs)[0].outputs["o"]
    np.testing.assert_allclose(out, ref, rtol=5e-2, atol=5e-3)
