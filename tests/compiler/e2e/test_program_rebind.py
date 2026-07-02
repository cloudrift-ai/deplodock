"""``CompiledProgram.rebind`` / ``run_once`` — the serving execution path.

One compiled dynamic-seq_len program serves request after request: ``rebind``
re-binds fresh inputs (re-sizing symbolic-shaped buffers, leaving weights
untouched), ``run_once`` launches without bench instrumentation. Built on a
real ``torch.nn.RMSNorm`` traced with a symbolic seq_len, mirroring
``tests/compiler/ir/test_dynamic_shapes.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..conftest import requires_cuda

pytestmark = [requires_cuda]


@pytest.fixture(scope="module")
def rmsnorm_setup():
    """Compiled dynamic-seq_len RMSNorm graph + its torch module."""
    pytest.importorskip("cupy")
    import torch

    from emmy.compiler.backend.cuda.backend import CudaBackend

    m = torch.nn.RMSNorm(256)
    with torch.no_grad():
        m.weight.copy_(torch.rand_like(m.weight) + 0.5)
    from emmy.compiler.trace.torch import trace_module

    seq = torch.export.Dim("seq_len", min=5, max=4096)
    graph = trace_module(m, (torch.randn(1, 32, 256),), dynamic_shapes={"x": {1: seq}})
    compiled = CudaBackend().compile(graph)
    return compiled, m


def _inputs(m, s: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    import torch

    x = np.random.RandomState(s).standard_normal((1, s, 256)).astype(np.float32)
    weight = m.weight.detach().numpy().astype(np.float32)
    with torch.no_grad():
        ref = torch.nn.functional.rms_norm(torch.from_numpy(x), (256,), m.weight, eps=m.eps).numpy()
    return {"x": x, "p_weight": weight}, ref


def test_rebind_runs_at_new_seq_lens(rmsnorm_setup):
    """Build at S=32, rebind through S=64 and S=8 — outputs track the new
    runtime shape and match torch eager at every size."""
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock

    compiled, m = rmsnorm_setup
    with gpu_lock():
        feed, ref = _inputs(m, 32)
        prog = CompiledProgram.build(compiled, feed)
        prog.run_once()
        out = next(iter(prog.outputs().values()))
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

        for s in (64, 8, 64):  # grow, shrink, grow again
            feed, ref = _inputs(m, s)
            prog.rebind(feed)
            prog.run_once()
            out = next(iter(prog.outputs().values()))
            assert out.shape == (1, s, 256)
            np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_rebind_same_shape_reuploads_in_place(rmsnorm_setup):
    """Same seq_len, new contents: device arrays are reused (no realloc) and
    the fresh values flow through."""
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock

    compiled, m = rmsnorm_setup
    with gpu_lock():
        import torch

        feed, _ = _inputs(m, 16)
        prog = CompiledProgram.build(compiled, feed)
        before = {name: id(arr) for name, arr in prog.arrays.items()}

        x2 = np.random.RandomState(99).standard_normal((1, 16, 256)).astype(np.float32)
        prog.rebind({"x": x2, "p_weight": feed["p_weight"]})
        prog.run_once()
        out = next(iter(prog.outputs().values()))
        assert {name: id(arr) for name, arr in prog.arrays.items()} == before
        with torch.no_grad():
            ref = torch.nn.functional.rms_norm(torch.from_numpy(x2), (256,), m.weight, eps=m.eps).numpy()
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_rebind_keeps_weight_array_and_drops_graphs(rmsnorm_setup):
    """Weights (static-shaped, un-supplied) keep their device array across a
    seq_len change; captured CUDA graphs are invalidated."""
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock

    compiled, m = rmsnorm_setup
    with gpu_lock():
        feed, _ = _inputs(m, 32)
        prog = CompiledProgram.build(compiled, feed)
        # Every static-shaped buffer that isn't the activation input — the
        # weight however the tracer named/classified it, plus static scratch.
        weight_names = [b.name for b in prog.compiled.bufs if not b.is_symbolic and b.name != "x"]
        assert weight_names, "expected at least one static-shaped buffer (the RMSNorm weight)"
        weight_ids = {name: id(prog.arrays[name]) for name in weight_names}

        prog.capture_launch_graphs([1] * len(prog.compiled.launches))
        assert prog._graphs is not None

        feed64, ref64 = _inputs(m, 64)
        prog.rebind({"x": feed64["x"]})  # weights not re-supplied — the serving pattern
        assert prog._graphs is None, "rebind must drop captured graphs (they bake old pointers)"
        assert {name: id(prog.arrays[name]) for name in weight_names} == weight_ids, "weights must not re-upload on rebind"
        prog.run_once()
        out = next(iter(prog.outputs().values()))
        np.testing.assert_allclose(out, ref64, rtol=1e-4, atol=1e-4)
