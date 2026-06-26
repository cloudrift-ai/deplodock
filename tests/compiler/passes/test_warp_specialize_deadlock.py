"""Regression tests for the WS=1 stranded-TMA deadlock (Qwen3 ``k_linear_mean_reduce``).

The fused RMSNorm + gate/up-linear + SwiGLU kernel of a Qwen3-class decoder MLP wraps its
whole body in a per-thread M-fragment loop. The original deadlock came from the
``tile/085_warp_specialize`` producer pass: its shallow producer/consumer split stranded
every TMA ``StageBundle`` in the **consumer** branch (the producer branch came out empty),
so the TMA issues sat behind a ``threadIdx.x == 0`` guard no consumer-branch thread can
satisfy, and every consumer ``mbarrier.wait`` spun forever — a deterministic device hang.
The fix made the producer pass's ``_eligible`` run the same split the transform used and
raise ``"not reachable by the producer split"`` when no TMA depth-2 bundle landed
producer-side.

The block-DAG rewrite **removed** the ``085_warp_specialize`` producer pass and its
``DEPLODOCK_WARP_SPECIALIZE`` knob entirely, so that producer-side guard is no longer
reachable (the regression of pinning WS=1 onto this shape simply cannot be expressed). The
old WS=1-pinning test is gone (no code path left to exercise); ``test_mlp_slice_never_offers_ws1``
below is its present-day replacement — see the note there.

What is still live is (a) the present-day invariant that this shape never lowers to a
``WarpSpecialize`` node at all — no path can deploy WS=1 here — and (b) the materializer's
own stranding guard: ``emit_warp_specialize`` rejects a ``WarpSpecialize`` with empty
``consumer_thread_axes`` (the shape that would emit a consumer branch with no thread-axis
decode). Those two are tested directly.

The slice is the (1, 32, 1024) → (1, 32, 3072) fp16 MLP that hung Qwen3-Embedding-0.6B
layer 0; it is lowered under the default greedy pick (the brittle knob pins that reproduced
the old 085 tile family are no longer expressible under the block-DAG tier validation).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from deplodock.compiler import dtype as _dt
from deplodock.compiler import target as target_mod
from deplodock.compiler.context import Context
from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import LinearOp, RmsNormOp
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.ir.tile.ir import SerialTile, TileOp, WarpSpecialize, WarpTile

from ..conftest import requires_cuda, requires_sm90

_mat = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.kernel.100_materialize_tile")

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


@pytest.fixture
def _sm120_target():
    """Pin the sm_120 codegen path so the TMA-gated passes fire without a live device."""
    target_mod.set_target((12, 0))
    try:
        yield
    finally:
        target_mod.set_target(None)


# ---------------------------------------------------------------------------
# Present-day invariant: the shape never lowers to a WarpSpecialize node, so
# WS=1 can never deploy on it (neither the tuner nor the learned prior can
# reach a code path that emits the producer/consumer split).
# ---------------------------------------------------------------------------


def test_mlp_slice_never_offers_ws1(_sm120_target):
    """The fused linear+mean shape lowers with no ``WarpSpecialize`` node and no
    ``WARPSPEC=True`` knob — WS=1 is not in the enumeration, so the deadlock can never
    deploy. Run unpinned: the partition planner's free greedy pick must not surface WS."""
    from deplodock.compiler.pipeline import TILE_PASSES, Pipeline

    g2 = Pipeline.build(TILE_PASSES).run(_build_mlp_slice())
    tile_ops = [n.op for n in g2.nodes.values() if isinstance(n.op, TileOp)]
    assert tile_ops, "expected at least one lowered TileOp"
    for op in tile_ops:
        assert op.knobs.get("WARPSPEC") is not True, f"{op.name} stamped WARPSPEC=True"
        assert not any(isinstance(s, WarpSpecialize) for s in op.body.iter()), f"{op.name} emitted a WarpSpecialize node"


# ---------------------------------------------------------------------------
# Live stranding guard: emit_warp_specialize rejects an empty consumer-axis WS
# (the materializer-side equivalent of the removed producer-split guard — a
# consumer branch with no thread-axis decode is the stranded shape).
# ---------------------------------------------------------------------------


def test_materializer_rejects_empty_consumer_axes():
    """A ``WarpSpecialize`` with empty ``consumer_thread_axes`` raises the loud guard
    rather than emitting a consumer branch that no thread can decode (the present-day
    stranding guard, replacing the removed 085 producer-split check)."""
    k = Axis("k_outer", 8)
    ws = WarpSpecialize(
        producer_body=Body((SerialTile(axis=k, body=Body(()), kind="serial_outer"),)),
        consumer_body=Body((SerialTile(axis=k, body=Body(()), kind="serial_outer"),)),
        ring_depth=2,
        n_producer_threads=32,
        consumer_thread_axes=(),
    )
    op = TileOp(body=Body((WarpTile(axes=(Axis("ws_role", Dim(2)),), body=Body((ws,))),)), name="k_strand")
    g = Graph()
    g.add_node(op=op, inputs=[], output=Tensor(op.name, ()), node_id="op")
    ctx = Context(compute_capability=(9, 0))
    with pytest.raises(ValueError, match="consumer_thread_axes must be non-empty"):
        _mat.rewrite(ctx, g.nodes["op"])


# ---------------------------------------------------------------------------
# Deleted producer-pass guard.
# ---------------------------------------------------------------------------
#
# The original suite had a ``test_pinned_ws1_on_mlp_slice_raises`` that pinned
# ``DEPLODOCK_WARP_SPECIALIZE=1`` and asserted the 085 producer pass raised
# "not reachable by the producer split". Both the pass and the knob were removed in the
# block-DAG rewrite, so there is no code path left to exercise — pinning WS=1 onto this
# shape can no longer be expressed at all. ``test_mlp_slice_never_offers_ws1`` above is the
# present-day replacement (the shape simply never produces a WarpSpecialize node). No xfail
# stub is kept here because the only exception the old invocation now raises is an unrelated
# tier-pin error, which would make a strict xfail "pass" for the wrong reason.


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
