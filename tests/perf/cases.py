"""Curated (op, shape) cases for the perf suite.

Each ``Case`` describes a single sublayer that occurs inside a real
transformer block at a real shape. The same Case drives both:

- the PyTorch reference closure (``build_torch_ref``) — eager call into
  ``torch.matmul`` / ``F.rms_norm`` / ``F.softmax`` / ``F.silu`` /
  ``F.scaled_dot_product_attention`` etc.;
- the Deplodock graph (``build_deplodock_graph``) — a tiny ``Graph``
  built from the corresponding frontend op, then compiled by
  ``CudaBackend``.

Shapes are pinned to what TinyLlama-1.1B and Qwen2.5-7B blocks actually
see; we deliberately keep the list small (~30 cases) so the suite runs
in seconds and the summary table fits on one screen. Add cases by
appending to ``CASES`` — no other surgery needed.

Deplodock currently emits FP32 only, so all cases run FP32 on both
sides for an apples-to-apples comparison. The ``dtype`` field is kept
on ``Case`` so an FP16 column can be added later without touching
callers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Case dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    name: str  # e.g. "matmul.tinyllama.q_proj.s512"
    op: str  # one of: matmul, rmsnorm, softmax, silu_mul, rope, sdpa
    shapes: tuple[tuple[int, ...], ...]  # input shapes, in op order
    dtype: str = "fp32"
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def shape_str(self) -> str:
        return " x ".join("(" + ",".join(str(d) for d in s) + ")" for s in self.shapes)


# ---------------------------------------------------------------------------
# Model dimensions
# ---------------------------------------------------------------------------

# TinyLlama-1.1B-Chat-v1.0
_TL_H = 2048  # hidden_size
_TL_I = 5632  # intermediate_size
_TL_HEADS = 32
_TL_KV_HEADS = 4
_TL_DH = _TL_H // _TL_HEADS  # 64

# Qwen2.5-7B
_Q_H = 3584
_Q_I = 18944
_Q_HEADS = 28
_Q_KV_HEADS = 4
_Q_DH = _Q_H // _Q_HEADS  # 128

_SEQ_LENS = (32, 128, 512)


def _matmul_cases() -> list[Case]:
    cases: list[Case] = []
    for tag, hidden, inter, _heads, kv_heads, dh in [
        ("tinyllama", _TL_H, _TL_I, _TL_HEADS, _TL_KV_HEADS, _TL_DH),
        ("qwen", _Q_H, _Q_I, _Q_HEADS, _Q_KV_HEADS, _Q_DH),
    ]:
        kv_out = kv_heads * dh
        for s in _SEQ_LENS:
            for proj, n_out in [
                ("q_proj", hidden),
                ("kv_proj", kv_out),
                ("o_proj", hidden),
                ("gate_proj", inter),
                ("up_proj", inter),
                ("down_proj", hidden),
            ]:
                k_in = inter if proj == "down_proj" else hidden
                cases.append(
                    Case(
                        name=f"matmul.{tag}.{proj}.s{s}",
                        op="matmul",
                        shapes=((1, s, k_in), (k_in, n_out)),
                        tags=(tag, "matmul", proj),
                    )
                )
    return cases


def _rmsnorm_cases() -> list[Case]:
    cases: list[Case] = []
    for tag, hidden in [("tinyllama", _TL_H), ("qwen", _Q_H)]:
        for s in _SEQ_LENS:
            cases.append(
                Case(
                    name=f"rmsnorm.{tag}.s{s}",
                    op="rmsnorm",
                    shapes=((1, s, hidden), (hidden,)),
                    tags=(tag, "rmsnorm"),
                )
            )
    return cases


def _softmax_cases() -> list[Case]:
    cases: list[Case] = []
    for tag, heads in [("tinyllama", _TL_HEADS), ("qwen", _Q_HEADS)]:
        for s in _SEQ_LENS:
            cases.append(
                Case(
                    name=f"softmax.{tag}.s{s}",
                    op="softmax",
                    shapes=((1, heads, s, s),),
                    tags=(tag, "softmax"),
                )
            )
    return cases


def _silu_mul_cases() -> list[Case]:
    cases: list[Case] = []
    for tag, inter in [("tinyllama", _TL_I), ("qwen", _Q_I)]:
        for s in _SEQ_LENS:
            cases.append(
                Case(
                    name=f"silu_mul.{tag}.s{s}",
                    op="silu_mul",
                    shapes=((1, s, inter), (1, s, inter)),
                    tags=(tag, "silu_mul", "mlp"),
                )
            )
    return cases


def _sdpa_cases() -> list[Case]:
    cases: list[Case] = []
    for tag, heads, kv_heads, dh in [
        ("tinyllama", _TL_HEADS, _TL_KV_HEADS, _TL_DH),
        ("qwen", _Q_HEADS, _Q_KV_HEADS, _Q_DH),
    ]:
        for s in _SEQ_LENS:
            cases.append(
                Case(
                    name=f"sdpa.{tag}.s{s}",
                    op="sdpa",
                    shapes=((1, heads, s, dh), (1, kv_heads, s, dh), (1, kv_heads, s, dh)),
                    tags=(tag, "sdpa", "attn"),
                )
            )
    return cases


PRIMITIVE_CASES: list[Case] = _matmul_cases() + _rmsnorm_cases() + _softmax_cases() + _silu_mul_cases()
FUSED_CASES: list[Case] = _sdpa_cases()
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

    if case.op == "softmax":
        x = _r(case.shapes[0])
        return lambda: F.softmax(x, dim=-1)

    if case.op == "silu_mul":
        gate = _r(case.shapes[0])
        up = _r(case.shapes[1])
        return lambda: F.silu(gate) * up

    if case.op == "sdpa":
        q = _r(case.shapes[0])
        k = _r(case.shapes[1])
        v = _r(case.shapes[2])
        # GQA expansion handled by SDPA when enable_gqa=True (PyTorch ≥ 2.5).
        return lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=q.shape[-3] != k.shape[-3])

    raise ValueError(f"unknown op: {case.op}")


# ---------------------------------------------------------------------------
# Deplodock graph builders
# ---------------------------------------------------------------------------


def build_deplodock_graph(case: Case):
    """Build a small ``Graph`` for the case using frontend ops; the
    compiler decomposes/fuses on its own."""
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp, RmsNormOp, SdpaOp, SoftmaxOp
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

    if case.op == "softmax":
        (x_shape,) = case.shapes
        g.add_node(InputOp(), [], Tensor("x", x_shape), node_id="x")
        g.add_node(SoftmaxOp(axis=-1), ["x"], Tensor("y", x_shape), node_id="y")
        g.inputs = ["x"]
        g.outputs = ["y"]
        return g

    if case.op == "silu_mul":
        gate_shape, up_shape = case.shapes
        g.add_node(InputOp(), [], Tensor("gate", gate_shape), node_id="gate")
        g.add_node(InputOp(), [], Tensor("up", up_shape), node_id="up")
        g.add_node(ElementwiseOp("silu"), ["gate"], Tensor("s", gate_shape), node_id="s")
        g.add_node(ElementwiseOp("multiply"), ["s", "up"], Tensor("y", gate_shape), node_id="y")
        g.inputs = ["gate", "up"]
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

    raise ValueError(f"unknown op: {case.op}")
