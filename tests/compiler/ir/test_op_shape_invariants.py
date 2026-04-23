"""Tensor IR rank-preservation invariants.

Three invariants, one test each:

1. ElementwiseOp rejects mismatched input shapes — broadcasts must be
   expressed as explicit IndexMapOp wrappers upstream.
2. ReduceOp output is keepdim — the reduced dim stays at size 1, preserving
   rank.
3. After decomposition, every ElementwiseOp in the graph has all its inputs
   at the op's output shape. Protects against new decomp rules skipping the
   broadcast_to helper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp


def test_elementwise_rejects_mismatched_shapes():
    op = ElementwiseOp(fn="add")
    with pytest.raises(ValueError, match="must all match"):
        op.infer_output_shape([(4, 8), (8,)])


def test_elementwise_accepts_matching_shapes():
    op = ElementwiseOp(fn="add")
    assert op.infer_output_shape([(4, 8), (4, 8)]) == (4, 8)


@pytest.mark.parametrize("fn", ["sum", "max", "prod"])
@pytest.mark.parametrize(
    ("input_shape", "axis", "expected"),
    [
        ((4, 8), -1, (4, 1)),
        ((4, 8), 0, (1, 8)),
        ((1, 32, 2048), -1, (1, 32, 1)),
        ((2, 3, 4, 5), 2, (2, 3, 1, 5)),
    ],
)
def test_reduce_output_is_keepdim(fn, input_shape, axis, expected):
    assert ReduceOp(fn=fn, axis=axis).infer_output_shape([input_shape]) == expected


def test_decomposition_emits_broadcast_explicit_elementwise():
    """After ``compile_graph`` runs decomposition, every ElementwiseOp in the
    fused graph must have all inputs at the op's declared output shape.
    Catches new decomposition rules that forget to call broadcast_to.
    """
    import torch

    from deplodock.compiler.rewriter import run_pass
    from deplodock.compiler.trace.torch import trace_module

    rules_dir = Path(__file__).parent.parent.parent.parent / "deplodock" / "compiler" / "passes"

    # Exercise each decomp rule: RMSNorm hits decompose_rms_norm + decompose_mean.
    # Softmax hits decompose_softmax. Linear(x) hits decompose_linear +
    # decompose_matmul. SiLU via nn.SiLU hits decompose_silu.
    modules = [
        ("rms_norm", torch.nn.RMSNorm(32), (torch.randn(1, 8, 32),)),
        ("softmax", torch.nn.Softmax(dim=-1), (torch.randn(1, 8, 32),)),
        ("linear", torch.nn.Linear(32, 16, bias=True), (torch.randn(1, 8, 32),)),
        ("silu", torch.nn.SiLU(), (torch.randn(1, 8, 32),)),
    ]

    for name, module, inputs in modules:
        graph = trace_module(module, inputs)
        decomposed = run_pass(graph, rules_dir / "decomposition")
        for n in decomposed.nodes.values():
            if not isinstance(n.op, ElementwiseOp):
                continue
            out_shape = tuple(n.output.shape)
            for inp_id in n.inputs:
                inp_shape = tuple(decomposed.nodes[inp_id].output.shape)
                assert inp_shape == out_shape, (
                    f"{name}: ElementwiseOp {n.id} ({n.op.fn}) input {inp_id} shape {inp_shape} != output {out_shape}"
                )
