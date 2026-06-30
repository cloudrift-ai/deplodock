"""Masked cooperative reduce over a SYMBOLIC reduce axis.

A reduction whose reduce axis is symbolic (dynamic ``seq_len``) used to be
forced onto the degenerate scalar path (``010_partition_loops``: symbolic K →
``E_K=1``, ``BR=1``) — one thread serially reduces the whole runtime extent,
~6% occupancy. The masked cooperative reduce tiles the symbolic reduce at the
``Dim`` hint (ceil-div ``K_o``) and splits it across ``BR`` cooperative lanes
with a segmented-shuffle combine, exactly like a static reduce — the partial
last tile is handled by clamping the K gmem index for a safe read and masking
each ``Accum``'s input value to the op identity past ``seq_len`` (so the
overhang folds a no-op). The ``Accum`` stays a direct child of the reduce loop
(``is_reduce`` + the combine intact); only the post-reduce Write loop gets a
plain boundary ``Cond``.

This is the softmax-producer perf path (the dominant remaining dynamic-attention
kernel): the deployed softmax over a symbolic key axis can now reach the same
cooperative-reduce occupancy as the static twin.
"""

from __future__ import annotations

import numpy as np
import pytest


def _has_cuda() -> bool:
    try:
        import cupy as cp

        return cp.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _supports_shuffle() -> bool:
    """The cooperative combine uses ``__shfl_xor_sync`` (sm_30+, every CUDA GPU)."""
    return _has_cuda()


def _dynamic_softmax_graph():
    """Trace ``softmax(x, dim=1)`` with axis 1 (the reduce axis) symbolic."""
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs

    ds = build_torch_dynamic_shapes(parse_position_specs(["seq_len@x:1"]))
    graph, _, _ = graph_from_code("torch.softmax(torch.randn(8, 512), dim=1)", dynamic_shapes=ds)
    return graph


@pytest.mark.skipif(not _supports_shuffle(), reason="masked cooperative reduce runs/renders under CUDA")
def test_masked_cooperative_softmax_structure(monkeypatch):
    """With a cooperative ``BR`` pinned, the symbolic-reduce softmax deploys the
    cooperative combine (``__shfl_xor_sync``) over a hint-tiled, boundary-masked
    K — the runtime ``seq_len`` arg, the per-tile mask, and the clamped read are
    all present (vs the old degenerate per-thread serial reduce)."""
    monkeypatch.setenv("EMMY_REDUCE", "t64")
    from emmy.compiler.backend.cuda.backend import CudaBackend

    compiled = CudaBackend().compile(_dynamic_softmax_graph())
    src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None))
    assert "int seq_len" in src, "symbolic reduce must carry the runtime extent arg"
    assert "__shfl_xor_sync" in src, "cooperative reduce must emit the segmented-shuffle combine"
    assert "seq_len +" in src, "K_o must ceil-div the symbolic extent (numerator seq_len + tile-1) at the hint"
    assert "< seq_len" in src, "the partial last tile must be boundary-masked"
    assert "seq_len - 1" in src, "the K gmem read must clamp to seq_len-1 for a safe in-bounds load"


@pytest.mark.skipif(not _supports_shuffle(), reason="needs CUDA")
@pytest.mark.parametrize("seq", [1, 31, 64, 512, 513, 700])
def test_masked_cooperative_softmax_accuracy(monkeypatch, seq):
    """One compiled symbolic-reduce softmax kernel is accurate at runtime sizes
    below / at / above the 512 hint — the off-hint sizes (1, 31, 513, 700)
    straddle the BR·BK = 512-element tile, exercising the boundary mask (the
    overhang must fold the reduce identity, not garbage past ``seq_len``)."""
    monkeypatch.setenv("EMMY_REDUCE", "t64")
    from emmy.compiler.backend.cuda.backend import CudaBackend

    graph = _dynamic_softmax_graph()
    be = CudaBackend()
    compiled = be.compile(graph)
    inp, out = graph.inputs[0], graph.outputs[0]
    import torch

    x = np.random.default_rng(0).standard_normal((8, seq)).astype(np.float32)
    got = be.run(compiled, input_data={inp: x})[0].outputs[out]
    want = torch.softmax(torch.from_numpy(x), dim=1).numpy()
    assert got.shape == (8, seq)
    diff = float(np.abs(got - want).max())
    assert diff < 1e-4, f"seq={seq}: masked cooperative softmax mismatch (max abs err {diff})"
