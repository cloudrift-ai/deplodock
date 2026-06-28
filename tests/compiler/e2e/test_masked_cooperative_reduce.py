"""Cooperative reduce over a SYMBOLIC reduce axis (dynamic ``seq_len``).

A reduction whose reduce axis is symbolic used to be forced onto the degenerate scalar path
— one thread serially reducing the whole runtime extent. The cooperative reduce now splits
the symbolic axis across ``coop`` lanes exactly like a static reduce: each lane strides the
axis from its lane index to the **runtime** extent (``for k = lane; k < seq_len; k += coop``)
and the partials fold through the carrier-generic combine (``__shfl_xor_sync``). The strided
``< seq_len`` bound IS the masked tail — a lane whose start is past ``seq_len`` does zero
iterations and folds the carrier identity into the combine, so no ceil-div tiling, no gmem
clamp, and no explicit per-element mask are needed (unlike the old masked-tile lowering). The
``seq_len`` ``Dim`` becomes a runtime ``int`` kernel arg, resolved at launch from the input
shape; one compiled kernel runs at any size.

This is the softmax-producer perf path (the dominant remaining dynamic-attention kernel): the
deployed softmax over a symbolic key axis reaches the same cooperative-reduce occupancy as the
static twin.
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


def _dynamic_softmax_graph():
    """Trace ``softmax(x, dim=1)`` with axis 1 (the reduce axis) symbolic."""
    from deplodock.commands.trace import graph_from_code
    from deplodock.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs

    ds = build_torch_dynamic_shapes(parse_position_specs(["seq_len@x:1"]))
    graph, _, _ = graph_from_code("torch.softmax(torch.randn(8, 512), dim=1)", dynamic_shapes=ds)
    return graph


@pytest.mark.skipif(not _has_cuda(), reason="cooperative reduce runs/renders under CUDA")
def test_masked_cooperative_softmax_structure(monkeypatch):
    """With a cooperative ``REDUCE=b64`` pinned, the symbolic-reduce softmax deploys the
    cooperative combine (``__shfl_xor_sync``) over a runtime-bounded strided reduce — the
    ``seq_len`` runtime arg and the strided ``< seq_len`` bound (the masked tail) are present
    (vs the old degenerate per-thread serial reduce). No ceil-div / gmem clamp: the strided
    bound alone masks the overhang."""
    monkeypatch.setenv("DEPLODOCK_REDUCE", "b64")
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    compiled = CudaBackend().compile(_dynamic_softmax_graph())
    src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None))
    assert "int seq_len" in src, "symbolic reduce must carry the runtime extent arg"
    assert "__shfl_xor_sync" in src, "cooperative reduce must emit the segmented-shuffle combine"
    assert "< seq_len" in src, "each lane must stride to the runtime extent (the strided bound is the masked tail)"
    assert "__launch_bounds__(64)" in src, "the pinned coop=64 sets the per-CTA thread count"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("seq", [1, 31, 64, 512, 513, 700])
def test_masked_cooperative_softmax_accuracy(monkeypatch, seq):
    """One compiled symbolic-reduce softmax kernel is accurate at runtime sizes below / at /
    above the 512 hint — the off-hint sizes (1, 31, 513, 700) straddle the coop tile, so idle
    lanes (start past ``seq_len``) must fold the reduce identity, not garbage."""
    monkeypatch.setenv("DEPLODOCK_REDUCE", "b64")
    from deplodock.compiler.backend.cuda.backend import CudaBackend

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
    assert diff < 1e-4, f"seq={seq}: cooperative symbolic softmax mismatch (max abs err {diff})"
