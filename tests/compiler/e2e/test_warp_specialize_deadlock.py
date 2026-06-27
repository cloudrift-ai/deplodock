"""Backend-accuracy regression for the WS=1 stranded-TMA deadlock (Qwen3 ``k_linear_mean_reduce``).

The fused RMSNorm + gate/up-linear + SwiGLU kernel of a Qwen3-class decoder MLP wraps its whole body in a per-thread
M-fragment loop. A historical deadlock pinned WS=1 onto this shape and stranded every TMA ``StageBundle`` in the
consumer branch, hanging the device. This e2e test asserts the shape compiles and runs to completion under the default
greedy pick (a WS=1 regression would trip the per-launch watchdog's HungKernelError) and matches the numpy reference.

The slice is the (1, 32, 1024) → (1, 32, 3072) fp16 MLP that hung Qwen3-Embedding-0.6B layer 0.
"""

from __future__ import annotations

import numpy as np

from deplodock.compiler import dtype as _dt
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import LinearOp, RmsNormOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp

from ..conftest import requires_cuda, requires_sm90

_S, _H, _I = 32, 1024, 3072  # seq, hidden, intermediate — the Qwen3-Embedding-0.6B MLP slice


def _build_mlp_slice() -> Graph:
    """RMSNorm → gate/up linears → SwiGLU, fp16 — the ``k_linear_mean_reduce`` fusion."""
    f16 = _dt.get("f16")
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (1, _S, _H), f16), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("nw", (_H,), f16), node_id="nw")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("wg", (_I, _H), f16), node_id="wg")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("wu", (_I, _H), f16), node_id="wu")
    g.add_node(op=RmsNormOp(eps=1e-6), inputs=["x", "nw"], output=Tensor("xn", (1, _S, _H), f16), node_id="xn")
    g.add_node(op=LinearOp(), inputs=["xn", "wg"], output=Tensor("mg", (1, _S, _I), f16), node_id="mg")
    g.add_node(op=LinearOp(), inputs=["xn", "wu"], output=Tensor("mu", (1, _S, _I), f16), node_id="mu")
    g.add_node(op=ElementwiseOp("silu"), inputs=["mg"], output=Tensor("sg", (1, _S, _I), f16), node_id="sg")
    g.add_node(op=ElementwiseOp("multiply"), inputs=["sg", "mu"], output=Tensor("o", (1, _S, _I), f16), node_id="o")
    g.inputs = ["x", "nw", "wg", "wu"]
    g.outputs = ["o"]
    return g


# ---------------------------------------------------------------------------
# End-to-end: the shape compiles and runs to completion (no device hang).
# ---------------------------------------------------------------------------


@requires_cuda
@requires_sm90
def test_mlp_slice_completes_and_matches():
    """The fused linear+mean shape compiles and runs to completion under the default greedy
    pick (a WS=1 regression would trip the per-launch watchdog's HungKernelError) and matches
    the numpy reference."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.numpy import NumpyBackend

    g = _build_mlp_slice()
    rng = np.random.default_rng(0)
    f16 = np.dtype(np.float16)
    inputs = {
        "x": rng.standard_normal((1, _S, _H), dtype=np.float32).astype(f16),
        "nw": rng.standard_normal((_H,), dtype=np.float32).astype(f16),
        "wg": (rng.standard_normal((_I, _H), dtype=np.float32) * 0.05).astype(f16),
        "wu": (rng.standard_normal((_I, _H), dtype=np.float32) * 0.05).astype(f16),
    }

    ref_be = NumpyBackend()
    ref = ref_be.run(ref_be.compile(_build_mlp_slice()), input_data=inputs)[0].outputs["o"]

    be = CudaBackend()
    out = be.run(be.compile(g), input_data=inputs)[0].outputs["o"]

    assert out.shape == ref.shape
    assert np.all(np.isfinite(out.astype(np.float32))), "output has non-finite values"
    peak = float(np.max(np.abs(ref.astype(np.float32))))
    atol = max(1e-2, 0.05 * peak)
    np.testing.assert_allclose(out.astype(np.float32), ref.astype(np.float32), atol=atol, rtol=0.05)
