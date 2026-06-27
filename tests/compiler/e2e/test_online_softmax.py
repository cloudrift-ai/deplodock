"""Online-softmax fusion (``lowering/tile/010_recognize``, always on).

The standalone two-pass softmax (row-max reduce + ``Σ exp(x − max)`` reduce + normalize) fuses into a
single streaming online-softmax ``(m, d)`` ``Monoid`` pass (3 reads of ``x`` → 2). CPU test pins the
recognition (3 loops → 2 + the monoid); GPU tests pin numerics vs torch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop.ir import Accum, Assign, Body, Load, Loop, Monoid
from deplodock.compiler.pipeline.passes.lowering.tile._flash import online_softmax_combine
from deplodock.compiler.trace.torch import trace_module

from ..conftest import requires_cuda

_fuse = __import__(
    "deplodock.compiler.pipeline.passes.lowering.tile.010_recognize",
    fromlist=["_fuse"],
)._fuse


class _Softmax(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


def test_online_softmax_combine_builds_asymmetric_monoid() -> None:
    # state (m, d), partial (s); the asymmetric LSE monoid must author combine_states (the
    # cross-partition combine can't derive it from merge).
    mono = online_softmax_combine("m", "d", "s", axis="kv")
    assert mono.state == ("m", "d") and mono.partial == ("s",)
    assert mono.twist.combine_states, "combine_states must be authored for the asymmetric LSE monoid"
    assert mono.commutative


def _softmax_body() -> Body:
    # The decomposed two-pass softmax over reduce axis a1: a row-max reduce then a Σ exp(x − max) reduce.
    idx = (Var("a0"), Var("a1"))
    rowmax = Loop(
        axis=Axis(name="a1", extent=Dim(128)),
        body=Body.coerce((Load(name="in0", input="x", index=idx), Accum(name="acc0", value="in0", op=ElementwiseImpl("maximum")))),
    )
    sumexp = Loop(
        axis=Axis(name="a1", extent=Dim(128)),
        body=Body.coerce(
            (
                Load(name="in1", input="x", index=idx),
                Assign(name="v0", op="subtract", args=("in1", "acc0")),
                Assign(name="v1", op="exp", args=("v0",)),
                Accum(name="acc1", value="v1", op=ElementwiseImpl("add")),
            )
        ),
    )
    return Body.coerce((rowmax, sumexp))


def test_fuse_collapses_the_two_reduces_into_one_monoid() -> None:
    fused, changed = _fuse(_softmax_body())
    assert changed
    monoids = [s for s in fused.iter() if isinstance(s, Monoid)]
    loops = [s for s in fused if isinstance(s, Loop)]
    assert len(loops) == 1, "the two reduce loops fuse into one online-softmax loop"
    assert len(monoids) == 1 and monoids[0].state == ("acc0", "acc1"), "carrier keeps the original acc names"


def test_fuse_is_a_noop_on_an_unrelated_reduce_pair() -> None:
    # A row-max followed by a plain sum (no exp(x − max)) must NOT fuse.
    idx = (Var("a0"), Var("a1"))
    rowmax = Loop(
        axis=Axis(name="a1", extent=Dim(128)),
        body=Body.coerce((Load(name="in0", input="x", index=idx), Accum(name="acc0", value="in0", op=ElementwiseImpl("maximum")))),
    )
    plainsum = Loop(
        axis=Axis(name="a1", extent=Dim(128)),
        body=Body.coerce((Load(name="in1", input="x", index=idx), Accum(name="acc1", value="in1", op=ElementwiseImpl("add")))),
    )
    _fused, changed = _fuse(Body.coerce((rowmax, plainsum)))
    assert not changed


@requires_cuda
@pytest.mark.parametrize("shape", [(4, 128), (8, 256), (2, 64), (2, 4, 128)])
def test_online_softmax_matches_torch(shape) -> None:
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    torch.manual_seed(0)
    x = torch.randn(*shape)
    graph = trace_module(_Softmax().cpu(), (x,))
    backend = CudaBackend()
    compiled = backend.compile(graph)

    # The recognizer must have fired: exactly one Monoid kernel carrying the online-softmax fold.
    ks = [n for n in compiled.nodes if getattr(compiled.nodes[n].op, "kernel_source", None)]
    assert "combine(" in compiled.nodes[ks[0]].op.kernel_source or "acc0__mx" in compiled.nodes[ks[0]].op.kernel_source

    run_result, eager = backend.run(compiled, input_data={"x": x.numpy()}, pre_run=lambda: _Softmax()(x).numpy())
    got = list(run_result.outputs.values())[0]
    assert got.shape == eager.shape
    assert np.max(np.abs(got.flatten() - eager.flatten())) < 1e-4
