"""Curated (op, shape) cases for the perf suite.

Each ``Case`` describes one fused launch that Deplodock emits when
compiling a single Qwen3-Embedding-0.6B decoder layer at seq_len 32 /
128 / 512. The case list mirrors the 12 kernels produced by
``deplodock compile Qwen/Qwen3-Embedding-0.6B --layer 0`` after the
fusion pass, so the perf suite is an apples-to-apples view of what the
compiler actually runs at the model level.

For each Case both:

- the PyTorch reference closure (``build_torch_ref``) — eager call into
  ``torch.matmul`` / ``F.rms_norm`` / ``F.scaled_dot_product_attention``
  etc.;
- the Deplodock graph (``build_deplodock_graph``) — a small ``Graph``
  built from frontend ops that lowers to the same single fused
  kernel that comes out of the Qwen3-Embedding layer.

Shapes are pinned to Qwen3-Embedding-0.6B: hidden=1024, intermediate=
3072, num_heads=16, num_kv_heads=8, head_dim=128. Deplodock currently
emits FP32 only, so all cases run FP32 on both sides; the ``dtype``
field is kept on ``Case`` so an FP16 column can be added later
without touching callers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Case dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    name: str  # e.g. "matmul.qwen3emb.q_proj.s128"
    op: str  # one of: matmul, rmsnorm, sdpa, matmul_add, gated_mlp
    shapes: tuple[tuple[int, ...], ...]  # input shapes, in op order
    dtype: str = "fp32"
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def shape_str(self) -> str:
        return " x ".join("(" + ",".join(str(d) for d in s) + ")" for s in self.shapes)

    @property
    def code(self) -> str:
        """Inline ``deplodock run --code`` expression equivalent to this
        case. Used by the perf bencher to drive ``deplodock run --bench
        --profile`` per case so the benchmarking infra is shared with
        the CLI. Each input tensor is materialized via ``torch.randn``
        on the same shape; the final expression is what gets traced.

        No ``import`` preamble — the ``--code`` evaluator already binds
        ``torch`` / ``nn`` / ``F`` in scope (``commands/trace.trace_inline_code``),
        so the string drops straight into ``deplodock run -c "<this>"``."""
        s = self.shapes
        if self.op == "matmul":
            return f"a=torch.randn({s[0]}); b=torch.randn({s[1]}); torch.matmul(a,b)"
        if self.op == "rmsnorm":
            return f"x=torch.randn({s[0]}); w=torch.randn({s[1]}); F.rms_norm(x, {tuple(s[1])}, w, eps=1e-6)"
        if self.op == "sdpa":
            gqa = "True" if s[0][-3] != s[1][-3] else "False"
            return (
                f"q=torch.randn({s[0]}); k=torch.randn({s[1]}); v=torch.randn({s[2]}); "
                f"F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa={gqa})"
            )
        if self.op == "matmul_add":
            return f"x=torch.randn({s[0]}); w=torch.randn({s[1]}); r=torch.randn({s[2]}); torch.matmul(x,w)+r"
        if self.op == "gated_mlp":
            return f"x=torch.randn({s[0]}); wg=torch.randn({s[1]}); wu=torch.randn({s[2]}); F.silu(torch.matmul(x,wg))*torch.matmul(x,wu)"
        raise ValueError(f"unknown op: {self.op}")

    @property
    def iters(self) -> int:
        """Number of measured iterations the bencher should run for this
        case. Heavy ops (~ms-scale per iter) drop to 20 to keep the suite
        fast; everything else stays at 100. ``heavy`` is detected by
        total input element count — the dominant factor in matmul cost.
        Threshold (50 M elements) covers the fp32 cases that empirically
        run > 1 ms each (e.g. gated_mlp at s=512)."""
        total_elements = sum(_prod(s) for s in self.shapes)
        return 20 if total_elements > 50_000_000 else 100


def _prod(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


# ---------------------------------------------------------------------------
# Model dimensions — Qwen3-Embedding-0.6B
# ---------------------------------------------------------------------------

# Source: config.json from Qwen/Qwen3-Embedding-0.6B
_H = 1024  # hidden_size
_I = 3072  # intermediate_size
_HEADS = 16  # num_attention_heads
_KV_HEADS = 8  # num_key_value_heads
_DH = 128  # head_dim
_Q_DIM = _HEADS * _DH  # 2048 — fused Q-projection output width
_KV_DIM = _KV_HEADS * _DH  # 1024 — fused K/V-projection output width

_SEQ_LENS = (32, 128, 512)
_TAG = "qwen3emb"


# ---------------------------------------------------------------------------
# Per-kernel case builders
#
# The kernel ids below match the ``=== N: kernel_name ===`` blocks in
# ``deplodock compile Qwen/Qwen3-Embedding-0.6B --layer 0 --dump-dir DIR``,
# stage ``04_loop_fusion.kernels.txt``.
# ---------------------------------------------------------------------------


def _rmsnorm_layer_cases() -> list[Case]:
    """Kernel 0 (input_layernorm) and 9 (post_attention_layernorm) —
    identical shape, so just one case per seq_len."""
    cases: list[Case] = []
    for s in _SEQ_LENS:
        cases.append(
            Case(
                name=f"rmsnorm.{_TAG}.layer.s{s}",
                op="rmsnorm",
                shapes=((1, s, _H), (_H,)),
                tags=(_TAG, "rmsnorm", "layer_norm"),
            )
        )
    return cases


def _rmsnorm_qknorm_cases() -> list[Case]:
    """Kernels 4 / 5 — per-head RMSNorm on Q/K with the reduce over
    head_dim. The kernel produced by Deplodock fuses the reshape +
    transpose with the norm, but the perf-relevant arithmetic is the
    rmsnorm with a head_dim-wide reduce; we test that shape directly."""
    cases: list[Case] = []
    for s in _SEQ_LENS:
        for tag, heads in [("q_norm", _HEADS), ("k_norm", _KV_HEADS)]:
            cases.append(
                Case(
                    name=f"rmsnorm.{_TAG}.{tag}.s{s}",
                    op="rmsnorm",
                    shapes=((1, heads, s, _DH), (_DH,)),
                    tags=(_TAG, "rmsnorm", tag, "attn"),
                )
            )
    return cases


def _matmul_proj_cases() -> list[Case]:
    """Kernels 1 / 2 / 3 — Q/K/V projections out of the input rmsnorm.
    K and V share the same shape (KV_DIM), so we keep one ``kv_proj``
    case rather than testing both."""
    cases: list[Case] = []
    for s in _SEQ_LENS:
        for proj, n_out in [("q_proj", _Q_DIM), ("kv_proj", _KV_DIM)]:
            cases.append(
                Case(
                    name=f"matmul.{_TAG}.{proj}.s{s}",
                    op="matmul",
                    shapes=((1, s, _H), (_H, n_out)),
                    tags=(_TAG, "matmul", proj),
                )
            )
    return cases


def _sdpa_cases() -> list[Case]:
    """Kernels 6 + 7 — masked QK (with RoPE folded in) and softmax/AV.
    ``F.scaled_dot_product_attention`` is the natural eager reference;
    Deplodock's frontend ``SdpaOp`` decomposes and fuses into two
    launches matching the Qwen3 layer."""
    cases: list[Case] = []
    for s in _SEQ_LENS:
        cases.append(
            Case(
                name=f"sdpa.{_TAG}.s{s}",
                op="sdpa",
                shapes=((1, _HEADS, s, _DH), (1, _KV_HEADS, s, _DH), (1, _KV_HEADS, s, _DH)),
                tags=(_TAG, "sdpa", "attn"),
            )
        )
    return cases


def _matmul_add_cases() -> list[Case]:
    """Kernels 8 / 11 — o_proj+residual and down_proj+residual. Both
    fuse the residual add into the matmul epilogue."""
    cases: list[Case] = []
    for s in _SEQ_LENS:
        for proj, k_in in [("o_proj", _Q_DIM), ("down_proj", _I)]:
            cases.append(
                Case(
                    name=f"matmul_add.{_TAG}.{proj}.s{s}",
                    op="matmul_add",
                    shapes=((1, s, k_in), (k_in, _H), (1, s, _H)),
                    tags=(_TAG, "matmul_add", proj),
                )
            )
    return cases


def _gated_mlp_cases() -> list[Case]:
    """Kernel 10 — ``silu(x @ Wgate) * (x @ Wup)`` collapses into a
    single launch with two reductions over hidden_size sharing the same
    free axes. This is the dominant MLP-side kernel; do not confuse it
    with the older ``silu_mul_matmul`` variant (`(silu(g)*u) @ Wdown`)
    that does not appear in the Qwen3 layer."""
    cases: list[Case] = []
    for s in _SEQ_LENS:
        cases.append(
            Case(
                name=f"gated_mlp.{_TAG}.s{s}",
                op="gated_mlp",
                shapes=((1, s, _H), (_H, _I), (_H, _I)),
                tags=(_TAG, "gated_mlp", "mlp"),
            )
        )
    return cases


# Primitive vs fused split: ``PRIMITIVE_CASES`` covers single-frontend-op
# kernels (matmul, rmsnorm); ``FUSED_CASES`` covers kernels that fuse
# multiple frontend ops into one launch (sdpa, matmul_add, gated_mlp).
PRIMITIVE_CASES: list[Case] = _rmsnorm_layer_cases() + _rmsnorm_qknorm_cases() + _matmul_proj_cases()
FUSED_CASES: list[Case] = _sdpa_cases() + _matmul_add_cases() + _gated_mlp_cases()
CASES: list[Case] = PRIMITIVE_CASES + FUSED_CASES


# ---------------------------------------------------------------------------
# PyTorch reference builders
# ---------------------------------------------------------------------------


def build_torch_ref(case: Case) -> Callable[[], None]:
    """Pre-allocate inputs on CUDA and return a zero-arg eager closure."""
    import torch
    import torch.nn.functional as F

    device = "cuda"
    dtype = torch.float32  # deplodock is fp32-only today
    rng = torch.Generator(device=device).manual_seed(0)

    def _r(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=rng, device=device, dtype=dtype)

    if case.op == "matmul":
        a = _r(case.shapes[0])
        b = _r(case.shapes[1])
        return lambda: torch.matmul(a, b)

    if case.op == "rmsnorm":
        x = _r(case.shapes[0])
        w = _r(case.shapes[1])
        normalized_shape = case.shapes[1]
        return lambda: F.rms_norm(x, normalized_shape, w, eps=1e-6)

    if case.op == "sdpa":
        q = _r(case.shapes[0])
        k = _r(case.shapes[1])
        v = _r(case.shapes[2])
        # GQA expansion handled by SDPA when enable_gqa=True (PyTorch ≥ 2.5).
        return lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=q.shape[-3] != k.shape[-3])

    if case.op == "matmul_add":
        x = _r(case.shapes[0])
        w = _r(case.shapes[1])
        r = _r(case.shapes[2])
        return lambda: torch.matmul(x, w) + r

    if case.op == "gated_mlp":
        x = _r(case.shapes[0])
        wg = _r(case.shapes[1])
        wu = _r(case.shapes[2])
        return lambda: F.silu(torch.matmul(x, wg)) * torch.matmul(x, wu)

    raise ValueError(f"unknown op: {case.op}")


# ---------------------------------------------------------------------------
# Deplodock graph builders
# ---------------------------------------------------------------------------


def build_deplodock_graph(case: Case):
    """Build a small ``Graph`` for the case using frontend ops; the
    compiler decomposes/fuses on its own."""
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp, RmsNormOp, SdpaOp
    from deplodock.compiler.ir.tensor.ir import ElementwiseOp

    g = Graph()

    if case.op == "matmul":
        a_shape, b_shape = case.shapes
        out_shape = tuple(a_shape[:-1]) + (b_shape[-1],)
        g.add_node(InputOp(), [], Tensor("a", a_shape), node_id="a")
        g.add_node(InputOp(), [], Tensor("b", b_shape), node_id="b")
        g.add_node(MatmulOp(), ["a", "b"], Tensor("c", out_shape), node_id="c")
        g.inputs = ["a", "b"]
        g.outputs = ["c"]
        return g

    if case.op == "rmsnorm":
        x_shape, w_shape = case.shapes
        g.add_node(InputOp(), [], Tensor("x", x_shape), node_id="x")
        g.add_node(InputOp(), [], Tensor("w", w_shape), node_id="w")
        g.add_node(RmsNormOp(eps=1e-6), ["x", "w"], Tensor("y", x_shape), node_id="y")
        g.inputs = ["x", "w"]
        g.outputs = ["y"]
        return g

    if case.op == "sdpa":
        q_shape, k_shape, v_shape = case.shapes
        out_shape = tuple(q_shape[:-1]) + (v_shape[-1],)
        g.add_node(InputOp(), [], Tensor("q", q_shape), node_id="q")
        g.add_node(InputOp(), [], Tensor("k", k_shape), node_id="k")
        g.add_node(InputOp(), [], Tensor("v", v_shape), node_id="v")
        g.add_node(SdpaOp(is_causal=True), ["q", "k", "v"], Tensor("y", out_shape), node_id="y")
        g.inputs = ["q", "k", "v"]
        g.outputs = ["y"]
        return g

    if case.op == "matmul_add":
        x_shape, w_shape, r_shape = case.shapes
        mm_shape = tuple(x_shape[:-1]) + (w_shape[-1],)
        g.add_node(InputOp(), [], Tensor("x", x_shape), node_id="x")
        g.add_node(InputOp(), [], Tensor("w", w_shape), node_id="w")
        g.add_node(InputOp(), [], Tensor("r", r_shape), node_id="r")
        g.add_node(MatmulOp(), ["x", "w"], Tensor("m", mm_shape), node_id="m")
        g.add_node(ElementwiseOp("add"), ["m", "r"], Tensor("y", mm_shape), node_id="y")
        g.inputs = ["x", "w", "r"]
        g.outputs = ["y"]
        return g

    if case.op == "gated_mlp":
        x_shape, wg_shape, wu_shape = case.shapes
        mm_shape = tuple(x_shape[:-1]) + (wg_shape[-1],)
        g.add_node(InputOp(), [], Tensor("x", x_shape), node_id="x")
        g.add_node(InputOp(), [], Tensor("wg", wg_shape), node_id="wg")
        g.add_node(InputOp(), [], Tensor("wu", wu_shape), node_id="wu")
        g.add_node(MatmulOp(), ["x", "wg"], Tensor("mg", mm_shape), node_id="mg")
        g.add_node(MatmulOp(), ["x", "wu"], Tensor("mu", mm_shape), node_id="mu")
        g.add_node(ElementwiseOp("silu"), ["mg"], Tensor("sg", mm_shape), node_id="sg")
        g.add_node(ElementwiseOp("multiply"), ["sg", "mu"], Tensor("y", mm_shape), node_id="y")
        g.inputs = ["x", "wg", "wu"]
        g.outputs = ["y"]
        return g

    raise ValueError(f"unknown op: {case.op}")
