"""Reduction-combine coverage — one accuracy matrix over (op type × reduction variant).

The cross-execution-unit reduction the monoid-combine work generalizes (lanes → warps →
CTAs) is **carrier-generic**: a plain reduction (``Accum``), online softmax / its stats
(``Accum`` max+sum), and flash attention (the ``(m, d, o)`` twisted ``Monoid``) all fold
through the SAME combine, differing only in carrier state. This test pins each reduction
*variant* and checks every op type stays bit-accurate vs torch, so a change to one combine
stage (warp-shuffle / hierarchical smem) or the cross-CTA finalize (atomic ``c<cta>a`` vs
deferred ``c<cta>k``) can't silently break a carrier it wasn't tuned on.

- **op types** — reduction (mean / max), softmax, attention (SDPA), matmul.
- **reduction variants** — the cooperative combine stage (``REDUCE`` coop field ``b<n>``
  = serial / warp-shuffle / hierarchical smem) for the reduce-carrier ops; the cross-CTA finalize
  fold (the ``REDUCE`` GRID field ``g<n>a`` / ``g<n>k``) for the split-K matmul. All pins are the
  ``DEPLODOCK_REDUCE`` codec — no legacy ``BR`` / ``SPLITK`` / ``NOATOMIC``.

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


def _ref_sum(xs):
    return xs[0].sum(axis=1, keepdims=True)


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
    "sum": ("torch.randn(4, 1024).sum(dim=1, keepdim=True)", _ref_sum),
}


def _compile_run(
    code: str, env: dict[str, str], monkeypatch, *, dynamic: str | None = None, seq: int = 512
) -> tuple[np.ndarray, list[np.ndarray], str]:
    """Trace + compile ``code`` under the pinned ``env``, run on seeded inputs, and return
    ``(output, ordered_inputs, kernel_source)``. With ``dynamic`` (a ``parse_position_specs``
    spec like ``"seq_len@x:1"``) the named axis is traced symbolic and run at runtime size
    ``seq`` — one compiled kernel exercised at an off-hint length (the cooperative reduce
    strides to ``seq``, idle lanes folding the identity)."""
    from deplodock.commands.trace import graph_from_code
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.ir.base import ConstantOp

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    if dynamic is not None:
        from deplodock.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs

        ds = build_torch_dynamic_shapes(parse_position_specs([dynamic]))
        graph = graph_from_code(code, dynamic_shapes=ds)[0]
    else:
        graph = graph_from_code(code)[0]
    be = CudaBackend()
    compiled = be.compile(graph)
    rng = np.random.default_rng(0)
    feed: dict[str, np.ndarray] = {}
    ordered: list[np.ndarray] = []
    for name in graph.inputs:
        # A symbolic axis resolves to the runtime size ``seq``; static dims keep their extent.
        shape = tuple(d.as_static() if d.is_static else seq for d in graph.nodes[name].output.shape)
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


# Reduce-partition variants, pinned via the ``REDUCE`` codec (the schedule's ``020_schedule``
# decision, authoritative when pinned). Each has a distinct lowering STRUCTURE, asserted below
# (not just output accuracy):
#   serial (``""``)        — one thread per cell, no cross-thread / register fold
#   coop_warp (``b32``)    — a single-warp ``__shfl_xor_sync`` butterfly
#   coop_hier (``b128``)   — the 4-warp hierarchical shuffle→smem-tree
#   ilp (``r4``)           — standalone ILP: 4 register accumulators + a register tree (no coop)
#   ilp_coop (``r2/b32``)  — ILP composed with coop: 2 register accs, register tree, then shuffle
_COOP_VARIANTS = {"serial": "", "coop_warp": "b32", "coop_hier": "b128", "ilp": "r4", "ilp_coop": "r2/b32"}
_REDUCE_OPS = ("mean", "amax", "softmax")

# Shape modes — the reduce-carrier ops all reduce dim=1 of input ``x``, so the SAME matrix
# runs over a static reduce axis and a SYMBOLIC one (``seq_len@x:1``, traced dynamic, run at
# an off-hint runtime size so the cooperative reduce's strided ``< seq_len`` bound masks the
# tail). One compiled kernel per (op, variant) covers any length.
_SHAPES = {"static": None, "dynamic": "seq_len@x:1"}
_DYNAMIC_SEQ = 700  # off the 512 Dim hint → exercises the masked tail / partial final warp


def _assert_combine_structure(src: str, variant: str, label: str) -> None:
    """Assert the kernel source carries the intra-CTA combine each ``REDUCE`` variant pins —
    the structural counterpart to the accuracy check (a wrong schedule that still happens to
    compute the right answer serially must not pass as ``coop_warp`` / ``coop_hier``)."""
    has_shfl = "__shfl_xor_sync" in src
    has_smem_tree = "_smem[" in src and "for (int s =" in src  # the cross-warp TreeHalve slab
    has_reg = "__r1" in src  # a replicated register-accumulator copy (the ILP fold)
    if variant == "serial":
        assert not has_shfl, f"{label}/serial: unexpected warp-shuffle combine in a serial reduce"
        assert not has_reg, f"{label}/serial: unexpected register-fold replication"
        assert "__launch_bounds__(256)" in src, f"{label}/serial: expected the scalar-tier block (256)"
    elif variant == "coop_warp":
        assert has_shfl, f"{label}/coop_warp: expected a __shfl_xor_sync warp butterfly"
        assert not has_smem_tree, f"{label}/coop_warp: single-warp coop must NOT emit a cross-warp smem tree"
        assert "__launch_bounds__(32)" in src, f"{label}/coop_warp: expected a one-warp block (32)"
    elif variant == "coop_hier":
        assert has_shfl and has_smem_tree, f"{label}/coop_hier: expected the hierarchical shuffle→smem-tree combine"
        assert "__launch_bounds__(128)" in src, f"{label}/coop_hier: expected the 4-warp block (128)"
    elif variant == "ilp":
        assert has_reg, f"{label}/ilp: expected replicated register accumulators (the ILP fold)"
        assert not has_shfl, f"{label}/ilp: standalone ILP must NOT emit a cross-thread combine"
        assert "__launch_bounds__(256)" in src, f"{label}/ilp: standalone ILP runs the scalar-tier block (256)"
    else:  # ilp_coop
        assert has_reg, f"{label}/ilp_coop: expected replicated register accumulators (the ILP fold)"
        assert has_shfl, f"{label}/ilp_coop: ILP composed with coop must still emit the warp butterfly"
        assert "__launch_bounds__(32)" in src, f"{label}/ilp_coop: coop=32 sets a one-warp block"


@pytest.mark.parametrize("op", _REDUCE_OPS)
@pytest.mark.parametrize("variant", list(_COOP_VARIANTS))
@pytest.mark.parametrize("shape", list(_SHAPES))
def test_cooperative_combine_accuracy(op, variant, shape, monkeypatch):
    """Every reduce-carrier op — degenerate (``mean`` / ``amax``) AND twisted full-row
    (``softmax``, the online-softmax ``(m, d)`` carrier) — over BOTH a static and a SYMBOLIC
    reduce axis, stays accurate AND emits the pinned intra-CTA combine structure across the
    three stages (serial → warp-shuffle → hierarchical smem), pinned via the ``REDUCE`` coop
    field. The dynamic column proves the same combine deploys over a runtime ``seq_len`` (the
    strided ``< seq_len`` bound masking the tail), one kernel for any length. ``mean``'s
    divisor is the runtime extent too (a ``context_value`` constant resolved at launch), so
    dynamic ``mean`` divides by the right count."""
    code, ref_fn = _OPS[op]
    got, xs, src = _compile_run(code, {"DEPLODOCK_REDUCE": _COOP_VARIANTS[variant]}, monkeypatch, dynamic=_SHAPES[shape], seq=_DYNAMIC_SEQ)
    want = ref_fn(xs).reshape(got.shape)
    diff = float(np.abs(got - want).max())
    assert diff < 1e-3, f"{op}/{variant}/{shape}: combine mismatch (max abs err {diff})"
    _assert_combine_structure(src, variant, f"{op}/{shape}")
    if shape == "dynamic":
        assert "int seq_len" in src, f"{op}/dynamic: symbolic reduce must carry the runtime extent arg"
        assert "< seq_len" in src, f"{op}/dynamic: each lane must stride to the runtime extent (the masked tail)"


@pytest.mark.parametrize("variant", ["serial", "coop_warp"])
def test_attention_combine_accuracy(variant, monkeypatch):
    """The flash ``(m, l, O)`` twisted-monoid carrier is accurate AND emits the pinned combine
    serially and with a cooperative-KV combine — a 3-component warp butterfly over the static
    KV axis (the same carrier-generic combine, ``coop_warp`` = ``b32``)."""
    env = {"DEPLODOCK_REDUCE": _COOP_VARIANTS[variant]}
    code, ref_fn = _OPS["attention"]
    got, xs, src = _compile_run(code, env, monkeypatch)
    want = ref_fn(xs).reshape(got.shape)
    diff = float(np.abs(got - want).max())
    assert diff < 2e-3, f"attention/{variant}: flash mismatch (max abs err {diff})"
    if variant == "serial":
        assert "__shfl_xor_sync" not in src, "attention/serial: unexpected cooperative combine"
    else:
        # The flash carrier folds 3 state components (m, l, O) through the SAME butterfly.
        assert "__shfl_xor_sync" in src, "attention/coop_warp: expected the cooperative-KV warp butterfly"
        assert all(f"__shfl_xor_sync(__activemask(), {c}" in src for c in ("m_i", "l_i", "O_i")), (
            "attention/coop_warp: the flash combine must shuffle all 3 carrier components"
        )


# The carrier-generic cross-CTA producer + finalize, one case per (carrier × finalize). The
# additive carriers (matmul split-K, ``sum`` split-reduce) take BOTH finalize folds; the twisted
# flash ``(m, l, O)`` carrier is **kernel-only** (the ``e^{Δm}`` rescale can't be an ``atomicAdd``).
# ``flash`` flips on the fused streaming flash (``DEPLODOCK_FLASH``); all split the reduce/KV axis
# across 2 CTAs via the native ``REDUCE`` ``c2`` codec, the same knob for matmul / reduce / flash.
_CROSS_CTA = {
    "matmul": {"op": "matmul", "flash": False, "tol": 1e-2, "finalizes": ("atomic", "kernel")},
    "sum": {"op": "sum", "flash": False, "tol": 1e-2, "finalizes": ("atomic", "kernel")},
    "flash": {"op": "attention", "flash": True, "tol": 2e-3, "finalizes": ("kernel",)},
}
_CROSS_CTA_CASES = [(carrier, fin) for carrier, spec in _CROSS_CTA.items() for fin in spec["finalizes"]]


@pytest.mark.parametrize("carrier,finalize", _CROSS_CTA_CASES)
def test_cross_cta_finalize_accuracy_and_structure(carrier, finalize, monkeypatch):
    """The **carrier-generic cross-CTA producer + finalize**, one matrix over (carrier × finalize):
    a SEMIRING matmul split-K, an additive ``Accum`` ``sum`` split-reduce, and the twisted flash
    ``(m, l, O)`` split-KV (Flash-Decoding) all split their contraction axis across CTAs through
    the SAME fork — the matmul is the 1-component instantiation, flash the N-component twisted one.
    Each is accurate vs torch and emits the matching kernel set: ATOMIC (``c2a``) = one kernel with
    ``atomicAdd`` (additive only — illegal for the twisted carrier); deferred KERNEL (``c2k``) = a
    second ``__global__`` combine kernel writing/reading a ``__partial`` workspace, no ``atomicAdd``.
    The finalize is the native ``REDUCE`` codec's ``c`` letter — one knob owns split + finalize."""
    spec = _CROSS_CTA[carrier]
    code, ref_fn = _OPS[spec["op"]]
    env = {"DEPLODOCK_REDUCE": "g2a" if finalize == "atomic" else "g2k"}
    if spec["flash"]:
        env["DEPLODOCK_FLASH"] = "1"
    got, xs, src = _compile_run(code, env, monkeypatch)
    want = ref_fn(xs).reshape(got.shape)
    diff = float(np.abs(got - want).max())
    assert diff < spec["tol"], f"{carrier}/{finalize}: cross-CTA mismatch (max abs err {diff})"
    n_global = src.count("__global__")
    if finalize == "atomic":
        assert "atomicAdd" in src, "the atomic finalize must emit atomicAdd"
        assert n_global == 1, f"atomic finalize is one kernel, got {n_global}"
    else:
        assert "atomicAdd" not in src, "the deferred kernel finalize must not emit atomicAdd"
        assert n_global == 2, f"deferred finalize splices a second combine kernel, got {n_global}"
        assert "__partial" in src, "the producer writes its partial state to a workspace"
