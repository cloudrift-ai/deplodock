"""The fused-edge e2e coverage — a producer fused into its matmul consumer, ONE kernel.

Black-box (the rebuild's recovery contract): build the graph, compile through ``CudaBackend``,
assert the kernel count + accuracy vs a numpy reference. One combinatorial test covers the MAP
producer × tier matrix (``f(x, …) @ w`` — the demoted-cone shapes a real model emits before a
linear); the MONOID producer (RMSNorm → Linear) is the one special case (its staged-shared-row
structural pin lives with ``test_fused_prologue_compiles_in_budget``).

The **warp tier** engages under a warp ``TILE`` pin: the demoted cone nodifies to a computed-A
``Contraction`` (``_schedule._demoted_warp_option``) and the producer COMPUTE-FILLS the A slab the
``ldmatrix`` drain reads (the mma tier's ``sync`` transport). Two cells stay xfailed via the
registry: the broadcast producer recognizes as a flat un-annotated ``Map`` (a recognition gap), and
the MONOID (rmsnorm) cone carries a reduce — not compute-fillable per cell.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler import dtype as _dt
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from tests.compiler.conftest import requires_cuda

F16 = _dt.get("f16")
_M, _K, _N = 32, 64, 32  # M != K so the row / col broadcasts are unambiguous

_WARP_TILE = "a:mma_m16n8k16_f16/w1x1/f2x2/k2"  # tile 32x16, bk 32 — exact cover of the 32x64x32 shape


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _producer_graph(kind: str) -> tuple[Graph, tuple[str, ...]]:
    """The producer subgraph writing the ``xn`` intermediate — its op(s) + extra inputs."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (_M, _K), F16), node_id="x")
    if kind == "relu":
        g.add_node(ElementwiseOp("relu"), ["x"], Tensor("xn", (_M, _K), F16), node_id="xn")
        return g, ("x",)
    if kind == "sigmoid":
        g.add_node(ElementwiseOp("sigmoid"), ["x"], Tensor("xn", (_M, _K), F16), node_id="xn")
        return g, ("x",)
    if kind == "multiply":
        g.add_node(InputOp(), [], Tensor("y", (_M, _K), F16), node_id="y")
        g.add_node(ElementwiseOp("multiply"), ["x", "y"], Tensor("xn", (_M, _K), F16), node_id="xn")
        return g, ("x", "y")
    if kind == "broadcast":
        # ``xn[m,k] = x[m,k] · rs[m] · cs[k]`` — a row and a col broadcast, the rmsnorm
        # scale-application shape.
        g.add_node(InputOp(), [], Tensor("rs", (_M, 1), F16), node_id="rs")
        g.add_node(InputOp(), [], Tensor("cs", (1, _K), F16), node_id="cs")
        g.add_node(ElementwiseOp("multiply"), ["x", "rs"], Tensor("t", (_M, _K), F16), node_id="t")
        g.add_node(ElementwiseOp("multiply"), ["t", "cs"], Tensor("xn", (_M, _K), F16), node_id="xn")
        return g, ("x", "rs", "cs")
    raise ValueError(kind)


_PRODUCER_REFS = {
    "relu": lambda i: np.maximum(i["x"], 0),
    "sigmoid": lambda i: _sigmoid(i["x"]),
    "multiply": lambda i: i["x"] * i["y"],
    "broadcast": lambda i: i["x"] * i["rs"] * i["cs"],
}


def _compile_run(g: Graph, ins: dict) -> tuple[np.ndarray, list[str]]:
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    be = CudaBackend()
    compiled = be.compile(g)
    srcs = [n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None)]
    got = list(be.run(compiled, input_data=ins)[0].outputs.values())[0]
    return np.asarray(got), srcs


@requires_cuda
@pytest.mark.parametrize("producer", list(_PRODUCER_REFS))
@pytest.mark.parametrize("tier", ["scalar", "warp"])
def test_fused_map_matmul(tier, producer, monkeypatch):
    """``f(x, …) @ w`` computes in **one** kernel matching numpy — the MAP producer fused into the
    matmul, no gmem round-trip for the ``xn`` intermediate. Covers unary (relu / sigmoid),
    multi-input (multiply), and broadcast-operand (``x·rs[m]·cs[k]``) producers on the scalar tier;
    the ``warp`` cells additionally demand the ``mma.sync`` tier (the compute-filled A slab;
    the broadcast cell is xfailed — its producer recognizes as a flat un-annotated ``Map``)."""
    if tier == "warp":
        monkeypatch.setenv("DEPLODOCK_TILE", _WARP_TILE)
    g, extra = _producer_graph(producer)
    g.add_node(InputOp(), [], Tensor("w", (_K, _N), F16), node_id="w")
    g.add_node(MatmulOp(), ["xn", "w"], Tensor("o", (_M, _N), F16), node_id="o")
    g.inputs, g.outputs = [*extra, "w"], ["o"]

    rng = np.random.default_rng(0)
    ins = {nid: (rng.standard_normal(tuple(d.as_static() for d in g.nodes[nid].output.shape)) * 0.3).astype(np.float16) for nid in g.inputs}
    got, srcs = _compile_run(g, ins)
    assert len(srcs) == 1, f"{producer}/{tier}: the fused edge must be ONE kernel, got {len(srcs)}"
    if tier == "warp":
        assert "dpl_mma" in srcs[0], f"{producer}/warp: the pinned warp tier must engage on the fused matmul"
    f32 = {k: v.astype(np.float32) for k, v in ins.items()}
    ref = _PRODUCER_REFS[producer](f32) @ f32["w"]
    np.testing.assert_allclose(got.reshape(_M, _N).astype(np.float32), ref, atol=0.1, rtol=2e-2)


@requires_cuda
@pytest.mark.parametrize("tier", ["scalar", "warp"])
def test_fused_rmsnorm_linear(tier, monkeypatch):
    """The **MONOID** producer fuses: ``rmsnorm(x)·nw @ wg`` computes in one kernel matching a
    numpy reference (whether the linear's N axis rides the grid or a tail sweep is the schedule's
    choice — the staged-shared-row structural pin lives with `test_fused_prologue_compiles_in_budget`,
    whose shape keeps the sweep in-tail). The ``warp`` cell demands the mma tier on the fused
    matmul (xfailed — the MONOID-producer warp fused edge is not rebuilt)."""
    if tier == "warp":
        monkeypatch.setenv("DEPLODOCK_TILE", _WARP_TILE)
    S, H, inter = 32, 1024, 3072
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, S, H), F16), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (H,), F16), node_id="nw")
    g.add_node(InputOp(), [], Tensor("wg", (inter, H), F16), node_id="wg")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("xn", (1, S, H), F16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "wg"], Tensor("o", (1, S, inter), F16), node_id="o")
    g.inputs, g.outputs = ["x", "nw", "wg"], ["o"]

    rng = np.random.default_rng(0)
    ins = {
        "x": (rng.standard_normal((1, S, H)) * 0.3).astype(np.float16),
        "nw": (rng.standard_normal((H,)) * 0.3).astype(np.float16),
        "wg": (rng.standard_normal((inter, H)) * 0.1).astype(np.float16),
    }
    got, srcs = _compile_run(g, ins)
    assert len(srcs) == 1, f"the fused norm→linear must be ONE kernel, got {len(srcs)}"
    if tier == "warp":
        assert "dpl_mma" in srcs[0], "warp: the pinned mma tier must engage on the fused matmul"
    x, nw, wg = (ins[k].astype(np.float32) for k in ("x", "nw", "wg"))
    rms = x[0] * (1.0 / np.sqrt((x[0] ** 2).mean(axis=-1, keepdims=True) + 1e-6)) * nw
    # The fp16 fused-prologue path carries a few % relative error on large elements (cooperative
    # rms-scale reduce + fp16 matmul accumulate at K=1024) — rtol=0.1 catches a regression without
    # flaking; atol absorbs near-zero elements where relative error is meaningless.
    np.testing.assert_allclose(got.reshape(S, inter).astype(np.float32), rms @ wg.T, atol=0.5, rtol=0.1)
