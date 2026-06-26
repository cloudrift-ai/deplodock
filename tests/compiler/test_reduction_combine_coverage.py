"""Reduction-combine coverage — one accuracy matrix over (op type × reduction variant).

The cross-execution-unit reduction the monoid-combine work generalizes (lanes → warps →
CTAs) is **carrier-generic**: a plain reduction (``Accum``), online softmax / its stats
(``Accum`` max+sum), and flash attention (the ``(m, d, o)`` twisted ``Monoid``) all fold
through the SAME combine, differing only in carrier state. This test pins each reduction
*variant* and checks every op type stays bit-accurate vs torch, so a change to one combine
stage (warp-shuffle / hierarchical smem) or the cross-CTA finalize (atomic ``c<cta>a`` vs
deferred ``c<cta>k``) can't silently break a carrier it wasn't tuned on.

- **op types** — reduction (mean / max), softmax, attention (SDPA), matmul.
- **reduction variants** — the cooperative combine stage (native ``REDUCE`` coop field ``t<n>``
  = serial / warp-shuffle / hierarchical smem) for the reduce-carrier ops; the cross-CTA finalize
  fold (the ``REDUCE`` ``c``-letter ``c2a`` / ``c2k``) for the split-K matmul. All pins are the
  native ``DEPLODOCK_REDUCE`` codec — no legacy ``BR`` / ``SPLITK`` / ``NOATOMIC``.

Pure GPU accuracy (no ``-O1`` numerics change), so it runs in the correctness lane.
"""

from __future__ import annotations

import numpy as np
import pytest


def _has_cuda() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _has_cuda(), reason="reduction-combine accuracy runs on CUDA")


# --- op fixtures: code → graph, plus a torch reference over the ordered inputs ---


def _ref_mean(xs):
    return xs[0].mean(axis=1, keepdims=True)


def _ref_amax(xs):
    return xs[0].max(axis=1, keepdims=True)


def _ref_softmax(xs):
    import torch

    return torch.softmax(torch.from_numpy(xs[0]), dim=1).numpy()


def _ref_attention(xs):
    import torch

    q, k, v = (torch.from_numpy(a) for a in xs)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v).numpy()


def _ref_matmul(xs):
    return xs[0] @ xs[1]


# (label, code, ref_fn). The reduce axis is sized 1024 (reduction/softmax) so coop=128 reaches
# the 4-warp hierarchical smem combine; attention's KV is 64 (coop ≤ 64).
_OPS = {
    "mean": ("torch.randn(4, 1024).mean(dim=1, keepdim=True)", _ref_mean),
    "amax": ("torch.randn(4, 1024).amax(dim=1, keepdim=True)", _ref_amax),
    "softmax": ("torch.softmax(torch.randn(4, 1024), dim=1)", _ref_softmax),
    "attention": (
        "torch.nn.functional.scaled_dot_product_attention(torch.randn(1, 4, 64, 32), torch.randn(1, 4, 64, 32), torch.randn(1, 4, 64, 32))",
        _ref_attention,
    ),
    "matmul": ("torch.matmul(torch.randn(8, 1024), torch.randn(1024, 16))", _ref_matmul),
}


def _compile_run(code: str, env: dict[str, str], monkeypatch) -> tuple[np.ndarray, list[np.ndarray], str]:
    """Trace + compile ``code`` under the pinned ``env``, run on seeded inputs, and return
    ``(output, ordered_inputs, kernel_source)``."""
    from deplodock.commands.trace import graph_from_code
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.ir.base import ConstantOp

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    graph = graph_from_code(code)[0]
    be = CudaBackend()
    compiled = be.compile(graph)
    rng = np.random.default_rng(0)
    feed: dict[str, np.ndarray] = {}
    ordered: list[np.ndarray] = []
    for name in graph.inputs:
        shape = tuple(d.as_static() for d in graph.nodes[name].output.shape)
        arr = (rng.standard_normal(shape) * 0.5).astype(np.float32)
        feed[name] = arr
        ordered.append(arr)
    # Supply any baked constants the backend kept as ConstantOp nodes (e.g. RMSNorm weight).
    for nid, node in compiled.nodes.items():
        if isinstance(node.op, ConstantOp) and nid not in feed and node.op.value is not None:
            feed[nid] = np.array([node.op.value], dtype=np.float32)
    out_name = graph.outputs[0]
    got = np.asarray(be.run(compiled, input_data=feed)[0].outputs[out_name])
    src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None))
    return got, ordered, src


# Cooperative-combine variants for the reduce-carrier ops, pinned via the native REDUCE codec's
# coop field ``t<coop>`` (a partial pin — only ``coop`` is fixed, the offer still picks ``serial``
# so the reduce extent is covered): serial / single-warp shuffle / 4-warp hierarchical smem.
_COOP_VARIANTS = {"serial": "t1", "coop_warp": "t32", "coop_hier": "t128"}
_REDUCE_OPS = ("mean", "amax", "softmax")


@pytest.mark.parametrize("op", _REDUCE_OPS)
@pytest.mark.parametrize("variant", list(_COOP_VARIANTS))
def test_cooperative_combine_accuracy(op, variant, monkeypatch):
    """Every reduce-carrier op stays accurate across the three intra-CTA combine stages
    (serial → warp-shuffle → hierarchical smem), pinned via the native ``REDUCE`` coop field."""
    code, ref_fn = _OPS[op]
    got, xs, _src = _compile_run(code, {"DEPLODOCK_REDUCE": _COOP_VARIANTS[variant]}, monkeypatch)
    want = ref_fn(xs).reshape(got.shape)
    diff = float(np.abs(got - want).max())
    assert diff < 1e-3, f"{op}/{variant}: combine mismatch (max abs err {diff})"


@pytest.mark.parametrize("variant", ["serial", "coop_kv"])
def test_attention_combine_accuracy(variant, monkeypatch):
    """The flash ``(m, d, o)`` twisted-monoid carrier is accurate serially and with a
    cooperative-KV combine (the native ``REDUCE`` coop field over the static KV axis)."""
    env = {"DEPLODOCK_REDUCE": "t1"} if variant == "serial" else {"DEPLODOCK_REDUCE": "t32"}
    code, ref_fn = _OPS["attention"]
    got, xs, _src = _compile_run(code, env, monkeypatch)
    want = ref_fn(xs).reshape(got.shape)
    diff = float(np.abs(got - want).max())
    assert diff < 2e-3, f"attention/{variant}: flash mismatch (max abs err {diff})"


@pytest.mark.parametrize("finalize", ["atomic", "kernel"])
def test_cross_cta_finalize_accuracy_and_structure(finalize, monkeypatch):
    """The split-K matmul is accurate under BOTH cross-CTA finalize folds, and each emits the
    expected kernel set: ATOMIC (``c<cta>a``) = one kernel with ``atomicAdd``; deferred KERNEL
    (``c<cta>k``) = a second ``__global__`` combine kernel and no ``atomicAdd``. The finalize is
    the native ``REDUCE`` codec's ``c`` letter (``c2a`` / ``c2k``) — one knob owns split-K +
    finalize."""
    code, ref_fn = _OPS["matmul"]
    reduce_pin = "c2a" if finalize == "atomic" else "c2k"
    got, xs, src = _compile_run(code, {"DEPLODOCK_REDUCE": reduce_pin}, monkeypatch)
    want = ref_fn(xs).reshape(got.shape)
    diff = float(np.abs(got - want).max())
    assert diff < 1e-2, f"matmul/{finalize}: split-K mismatch (max abs err {diff})"
    n_global = src.count("__global__")
    if finalize == "atomic":
        assert "atomicAdd" in src, "the atomic finalize must emit atomicAdd"
        assert n_global == 1, f"atomic finalize is one kernel, got {n_global}"
    else:
        assert "atomicAdd" not in src, "the deferred kernel finalize must not emit atomicAdd"
        assert n_global == 2, f"deferred finalize splices a second combine kernel, got {n_global}"
