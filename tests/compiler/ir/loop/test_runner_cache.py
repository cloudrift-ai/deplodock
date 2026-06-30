"""Regression: the loop-runner JIT cache must key on kernel content, not
object identity.

``execute_loop_op_cpp`` memoizes JIT-compiled kernels so repeated calls with
the same kernel don't re-invoke Cling. The key used to include ``id(loop)``,
which is a use-after-free hazard: CPython recycles the address of a GC'd
``LoopOp``, so a later same-shape LoopOp (e.g. ``tanh`` after a freed
``negative``) could alias the old id and be handed the stale cached kernel —
silently returning ``-x`` for ``tanh``. This surfaced only under randomized /
parallel test ordering (the GC timing that triggers id reuse), never in
isolation.

We force the collision deterministically by pinning every ``id`` the runner
sees to a constant, then check two structurally-different same-shape kernels
each compute their own result. Under the old id-keyed cache this fails (the
second kernel is served the first's compiled function); under content-keying
it passes. cppyy-only — no CUDA required.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp

cppyy = pytest.importorskip("cppyy")


def _build(fn: str) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8)), node_id="x")
    g.add_node(op=ElementwiseOp(fn), inputs=["x"], output=Tensor("y", (4, 8)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    return g


def _run_loop(graph: Graph, x: np.ndarray) -> np.ndarray:
    from emmy.compiler.backend.loop import LoopBackend

    be = LoopBackend()
    return be.run(be.compile(graph), input_data={"x": x})[0].outputs["y"]


def test_cache_not_keyed_on_object_identity(monkeypatch):
    from emmy.compiler.ir.loop import runner

    # Pin every id() the runner module evaluates to one constant, forcing the
    # exact id-collision the old cache key was vulnerable to.
    monkeypatch.setattr(runner, "id", lambda _obj: 0xC0FFEE, raising=False)
    monkeypatch.setattr(runner, "_FN_CACHE", {})

    x = np.random.default_rng(0).uniform(0.1, 5.0, size=(4, 8)).astype(np.float32)

    neg = _run_loop(_build("negative"), x)
    np.testing.assert_allclose(neg, -x, rtol=2e-5, atol=1e-5)

    # Same shape, different body — must NOT be served the cached negate kernel.
    tanh = _run_loop(_build("tanh"), x)
    np.testing.assert_allclose(tanh, np.tanh(x), rtol=2e-5, atol=1e-5)
