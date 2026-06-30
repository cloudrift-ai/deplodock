"""Cooperative-reduce coverage — the carrier-generic monoid combine, one file.

The cross-execution-unit reduction the monoid-combine work generalizes (lanes → warps → CTAs)
is **carrier-generic**: a plain reduction (``Accum``), online softmax / its stats (``Accum``
max+sum), and flash attention (the ``(m, d, o)`` twisted ``Monoid``) all fold through the SAME
combine, differing only in carrier state. This file pins each reduction *variant* and checks
every op type stays accurate vs torch AND emits the matching lowering structure, so a change to
one combine stage (warp-shuffle / hierarchical smem) or the cross-CTA finalize (atomic ``c<cta>a``
vs deferred ``c<cta>k``) can't silently break a carrier it wasn't tuned on.

Sections:
- **cooperative combine matrix** — op type × reduction variant × (static / symbolic) reduce axis.
- **symbolic straddling sweep** — one symbolic-reduce softmax kernel run at off-hint runtime
  sizes (the strided ``< seq_len`` bound IS the masked tail).
- **flash carrier + cross-CTA finalize** — the twisted ``(m, l, O)`` combine, split-KV / split-K,
  atomic vs deferred-kernel finalize, and the projection-epilogue distributivity guard.
- **online-softmax fusion** — the two-pass → one-pass streaming recognizer (IR-unit + GPU).
- **2D segmented coop** — a pinned ``BN>1`` × ``BR>1`` reduce, segmented shuffle per row.

Pure GPU accuracy (no ``-O1`` numerics change), so it runs in the correctness lane. The
fusion-recognizer IR-unit tests need no GPU and stay ungated.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.axis import Axis, AxisRole
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop.ir import Accum, Assign, Body, Load, Loop
from deplodock.compiler.pipeline.passes.lowering.tile._softmax import _fuse, online_softmax_combine
from deplodock.compiler.trace.torch import trace_module

from ..conftest import requires_cuda

# --------------------------------------------------------------------------- #
# Shared harness: code → graph → compiled kernel, plus a torch reference.
# --------------------------------------------------------------------------- #


def _ref_mean(xs):
    return xs[0].mean(axis=1, keepdims=True)


def _ref_amax(xs):
    return xs[0].max(axis=1, keepdims=True)


def _ref_softmax(xs):
    return torch.softmax(torch.from_numpy(xs[0]), dim=1).numpy()


def _ref_attention(xs):
    q, k, v = (torch.from_numpy(a) for a in xs)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v).numpy()


def _ref_matmul(xs):
    return xs[0] @ xs[1]


def _ref_sum(xs):
    return xs[0].sum(axis=1, keepdims=True)


def _ref_sumsq(xs):
    # Prologue: the ⊗ pre-map (square) feeds the ⊕ fold — a Map partial under the Monoid.
    return (xs[0] ** 2).sum(axis=1, keepdims=True)


def _ref_l2(xs):
    # Prologue (square) + epilogue (sqrt projection) around the sum carrier.
    return np.sqrt((xs[0] ** 2).sum(axis=1, keepdims=True))


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
    # Prologue (a ⊗ pre-map under the fold) and prologue+epilogue (the φ projection around it):
    # ``sum(x·x)`` squares each element before the additive fold; ``l2`` then ``sqrt``s the result.
    "sumsq": ("(lambda t: (t * t).sum(dim=1, keepdim=True))(torch.randn(4, 1024))", _ref_sumsq),
    "l2": ("torch.sqrt((lambda t: (t * t).sum(dim=1, keepdim=True))(torch.randn(4, 1024)))", _ref_l2),
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


# --------------------------------------------------------------------------- #
# Cooperative combine matrix — op type × reduction variant × shape mode.
# --------------------------------------------------------------------------- #

# Reduce-partition variants, pinned via the ``REDUCE`` codec (the schedule's ``020_schedule``
# decision, authoritative when pinned). Each has a distinct lowering STRUCTURE, asserted below
# (not just output accuracy):
#   serial (``""``)        — one thread per cell, no cross-thread / register fold
#   coop_warp (``b32``)    — a single-warp ``__shfl_xor_sync`` butterfly
#   coop_hier (``b128``)   — the 4-warp hierarchical shuffle→smem-tree
#   ilp (``r4``)           — standalone ILP: 4 register accumulators + a register tree (no coop)
#   ilp_coop (``r2/b32``)  — ILP composed with coop: 2 register accs, register tree, then shuffle
_COOP_VARIANTS = {"serial": "", "coop_warp": "b32", "coop_hier": "b128", "ilp": "r4", "ilp_coop": "r2/b32"}
# Degenerate (mean / amax), twisted full-row (softmax), and the prologue / prologue+epilogue
# carriers (sumsq / l2) — the pre-map and projection ride the same cooperative combine.
_REDUCE_OPS = ("mean", "amax", "softmax", "sumsq", "l2")

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


@requires_cuda
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


# --------------------------------------------------------------------------- #
# Symbolic straddling sweep — one symbolic-reduce softmax kernel, off-hint sizes.
# --------------------------------------------------------------------------- #
# The cooperative reduce splits a symbolic axis across ``coop`` lanes exactly like a static
# reduce: each lane strides ``for k = lane; k < seq_len; k += coop`` and the partials fold
# through the carrier-generic combine. The strided ``< seq_len`` bound IS the masked tail — a
# lane whose start is past ``seq_len`` does zero iterations and folds the carrier identity, so
# no ceil-div tiling / gmem clamp / explicit per-element mask is needed. This is the deployed
# softmax-producer perf path over a symbolic key axis. ``coop=b64`` sets a 2-warp block.
_STRADDLE_SOFTMAX = "torch.softmax(torch.randn(8, 512), dim=1)"


@requires_cuda
@pytest.mark.parametrize("seq", [1, 31, 64, 512, 513, 700])
def test_symbolic_cooperative_softmax_sweep(monkeypatch, seq):
    """One compiled symbolic-reduce softmax kernel is accurate at runtime sizes below / at /
    above the 512 hint — the off-hint sizes (1, 31, 513, 700) straddle the coop tile, so idle
    lanes (start past ``seq_len``) must fold the reduce identity, not garbage. The same kernel
    carries the runtime ``seq_len`` arg + the cooperative ``__shfl_xor_sync`` combine over the
    strided ``< seq_len`` bound (vs the old degenerate per-thread serial reduce), in a 2-warp
    block."""
    got, xs, src = _compile_run(_STRADDLE_SOFTMAX, {"DEPLODOCK_REDUCE": "b64"}, monkeypatch, dynamic="seq_len@x:1", seq=seq)
    want = _ref_softmax(xs).reshape(got.shape)
    assert got.shape == (8, seq)
    diff = float(np.abs(got - want).max())
    assert diff < 1e-4, f"seq={seq}: cooperative symbolic softmax mismatch (max abs err {diff})"
    assert "int seq_len" in src, "symbolic reduce must carry the runtime extent arg"
    assert "__shfl_xor_sync" in src, "cooperative reduce must emit the segmented-shuffle combine"
    assert "< seq_len" in src, "each lane must stride to the runtime extent (the strided bound is the masked tail)"
    assert "__launch_bounds__(64)" in src, "the pinned coop=64 sets the per-CTA thread count"


# --------------------------------------------------------------------------- #
# Flash carrier + cross-CTA finalize.
# --------------------------------------------------------------------------- #


@requires_cuda
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


@requires_cuda
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


# A reduce carrier with a PROJECTION epilogue under cross-CTA split-reduce. The ATOMIC finalize
# applies the projection to each CTA's partition before the ``atomicAdd``, so it is correct only
# when the projection DISTRIBUTES over the add (``Σ φ(xₛ) = φ(Σ xₛ)``): ``mean``'s ``×1/N`` (a
# constant scale) distributes; ``l2``'s ``sqrt`` does not. The deferred-KERNEL finalize projects
# once after the cross-CTA combine, so it is correct for either. Regression guard for the bug
# where the atomic finalize wrote raw partial sums and silently dropped ``mean``'s ``×1/N``.
_PROJECTION_DISTRIBUTES = {"mean": True, "l2": False}


@requires_cuda
@pytest.mark.parametrize("op", list(_PROJECTION_DISTRIBUTES))
@pytest.mark.parametrize("finalize", ["atomic", "kernel"])
def test_split_reduce_projection_epilogue(op, finalize, monkeypatch):
    """A split-reduce carrier carrying a projection epilogue. ``mean``'s ``×1/N`` distributes over
    the atomic add, so the atomic finalize rides it per-partition and stays accurate; ``l2``'s
    ``sqrt`` does not, so the atomic finalize REFUSES (``NotImplementedError`` → pin ``g<n>k``).
    The deferred-kernel finalize projects once after the combine and is accurate for both."""
    code, ref_fn = _OPS[op]
    env = {"DEPLODOCK_REDUCE": "g2a" if finalize == "atomic" else "g2k"}
    if finalize == "atomic" and not _PROJECTION_DISTRIBUTES[op]:
        with pytest.raises(NotImplementedError, match="non-distributive projection"):
            _compile_run(code, env, monkeypatch)
        return
    got, xs, src = _compile_run(code, env, monkeypatch)
    want = ref_fn(xs).reshape(got.shape)
    diff = float(np.abs(got - want).max())
    assert diff < 1e-2, f"{op}/{finalize}: split-reduce projection mismatch (max abs err {diff})"
    if finalize == "atomic":
        assert "atomicAdd" in src, f"{op}/atomic: the per-partition projection still finalizes via atomicAdd"
        assert src.count("__global__") == 1, f"{op}/atomic: the atomic finalize is one kernel"
    else:
        assert "atomicAdd" not in src, f"{op}/kernel: the deferred finalize must not emit atomicAdd"
        assert src.count("__global__") == 2, f"{op}/kernel: the deferred finalize splices a combine kernel"


# --------------------------------------------------------------------------- #
# Online-softmax fusion — the two-pass → one-pass streaming recognizer.
# --------------------------------------------------------------------------- #
# The standalone two-pass softmax (row-max reduce + ``Σ exp(x − max)`` reduce + normalize) fuses
# into a single streaming online-softmax ``(m, d)`` ``Monoid`` pass (3 reads of ``x`` → 2). The
# IR-unit tests pin the recognition (3 loops → 2 + the monoid); the GPU test pins numerics vs
# torch and that the recognizer fired.


class _Softmax(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


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


def _unrelated_reduce_pair() -> Body:
    # A row-max followed by a plain sum (no exp(x − max)) — must NOT fuse.
    idx = (Var("a0"), Var("a1"))
    rowmax = Loop(
        axis=Axis(name="a1", extent=Dim(128)),
        body=Body.coerce((Load(name="in0", input="x", index=idx), Accum(name="acc0", value="in0", op=ElementwiseImpl("maximum")))),
    )
    plainsum = Loop(
        axis=Axis(name="a1", extent=Dim(128)),
        body=Body.coerce((Load(name="in1", input="x", index=idx), Accum(name="acc1", value="in1", op=ElementwiseImpl("add")))),
    )
    return Body.coerce((rowmax, plainsum))


def test_online_softmax_combine_builds_asymmetric_monoid() -> None:
    # state (m, d), partial (s); the asymmetric LSE monoid derives combine_states (the
    # cross-partition state⊕state combine) from its exp-family spec.
    mono = online_softmax_combine("m", "d", "s")
    assert mono.state.names == ("m", "d") and mono.partial_names() == ("s",)
    assert mono.combine_states, "combine_states must be derived for the asymmetric LSE monoid"


@pytest.mark.parametrize("kind,should_fuse", [("softmax_pair", True), ("unrelated_pair", False)])
def test_fuse_collapses_only_the_online_softmax_pair(kind, should_fuse) -> None:
    """``_fuse`` collapses the decomposed two-pass softmax (row-max + ``Σ exp(x − max)``) into one
    online-softmax loop + monoid (carrier keeps the original ``acc`` names), and is a no-op on an
    unrelated row-max + plain-sum pair."""
    body = _softmax_body() if should_fuse else _unrelated_reduce_pair()
    fused, changed = _fuse(body)
    assert changed == should_fuse
    if should_fuse:
        loops = [s for s in fused if isinstance(s, Loop)]
        assert len(loops) == 1, "the two reduce loops fuse into one online-softmax loop"
        fused_loop = loops[0]
        assert fused_loop.role is AxisRole.TWISTED and fused_loop.carrier is not None, "the fused loop is a TWISTED carrier"
        assert fused_loop.carrier.state.names == ("acc0", "acc1"), "carrier keeps the original acc names"


@requires_cuda
@pytest.mark.parametrize("shape", [(4, 128), (8, 256), (2, 64), (2, 4, 128)])
def test_online_softmax_matches_torch(shape) -> None:
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    torch.manual_seed(0)
    x = torch.randn(*shape)
    graph = trace_module(_Softmax().cpu(), (x,))
    backend = CudaBackend()
    compiled = backend.compile(graph)

    # The recognizer must have fired: a single fused kernel streaming one online-softmax loop
    # (the ``<rowmax>__osin`` fused-score input is the recognizer's signature, stable across the
    # carrier's internal temp naming).
    srcs = [getattr(compiled.nodes[n].op, "kernel_source", "") for n in compiled.nodes]
    assert any("__osin" in src for src in srcs), "online-softmax fusion did not fire"

    run_result, eager = backend.run(compiled, input_data={"x": x.numpy()}, pre_run=lambda: _Softmax()(x).numpy())
    got = list(run_result.outputs.values())[0]
    assert got.shape == eager.shape
    assert np.max(np.abs(got.flatten() - eager.flatten())) < 1e-4


# --------------------------------------------------------------------------- #
# 2D segmented coop — a pinned BN>1 × BR>1 reduce, segmented shuffle per row.
# --------------------------------------------------------------------------- #
# A pinned 2D cooperative reduce (``BN > 1`` free-axis threads alongside ``BR > 1`` cooperative-K
# lanes) must compute the same per-row sums as numpy: the cross-thread combine is a SEGMENTED
# warp shuffle over each row's ``BR`` lanes, combining each row independently. (Legacy ``BN`` /
# ``BR`` / ``FN`` / ``FK`` / ``BK`` knobs — the surviving backend-accuracy assertion from the
# deleted ``passes/test_strided_coop_rows.py``.)


def _reduce_graph(shape: tuple):
    from deplodock.compiler.graph import Graph, Tensor  # noqa: PLC0415
    from deplodock.compiler.ir.base import InputOp  # noqa: PLC0415
    from deplodock.compiler.ir.tensor.ir import ReduceOp  # noqa: PLC0415

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", shape), node_id="x")
    out_shape = (*shape[:-1], 1)
    g.add_node(ReduceOp(op="sum", axis=-1), ["x"], Tensor("o", out_shape), node_id="o")
    g.inputs, g.outputs = ["x"], ["o"]
    return g


@requires_cuda
def test_2d_segmented_coop_reduce_accuracy(monkeypatch):
    """A pinned 2D row (BN=8, BR=16) computes the same per-row sums as numpy —
    the segmented shuffle combines each row independently."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    for key, val in dict(BN=8, BR=16, FN=1, FK=1, BK=2).items():
        monkeypatch.setenv(f"DEPLODOCK_{key}", str(val))
    g = _reduce_graph((64, 128))
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 128)).astype(np.float32)
    be = CudaBackend()
    out = be.run(be.compile(g), input_data={"x": x})[0].outputs["o"]
    np.testing.assert_allclose(out, x.sum(-1, keepdims=True), rtol=1e-4, atol=1e-4)
