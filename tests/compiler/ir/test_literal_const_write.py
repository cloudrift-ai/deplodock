"""Tests for literal-scalar ``ConstantOp`` substitution inside ``Write.render``.

A ``Load`` reading from a scalar ``ConstantOp`` is skipped at render time
(its value gets inlined via ``ctx.literal_ssa``). Stmts that carry raw
SSA-name strings — ``Write.values`` in particular — must substitute the
literal at use sites instead of emitting the undefined SSA name.

Regression target: standalone broadcast-only kernels (e.g. ``unsqueeze``
on ``torch.zeros(1)``) used to render ``make_float2(in0, in0)`` with
``in0`` never declared, breaking compilation of every such kernel in
the Qwen3 attention-mask precompute path.
"""

from __future__ import annotations

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.kernel.ir import KernelOp
from deplodock.compiler.ir.kernel.render import render_kernelop
from deplodock.compiler.ir.loop import Axis, Load, Write
from deplodock.compiler.ir.tile.ir import ThreadTile
from deplodock.compiler.tensor import Tensor


def _broadcast_zero_kernel(width: int) -> tuple[KernelOp, dict]:
    """Kernel that loads a scalar literal constant ``zero`` once and writes
    it to every position of an output buffer. ``width`` is the Write width
    (1 = scalar, 2 = float2)."""
    a = Axis("a", 8)
    if width == 1:
        write = Write(output="out", index=(Var("a"),), value="z")
    else:
        write = Write(output="out", index=(Var("a"),), values=("z",) * width)
    body = (
        Load(name="z", input="zero", index=(Literal(0, "int"),), dtype=F32),
        ThreadTile(axes=(a,), body=(write,)),
    )
    kernel = KernelOp(body=body, name="k_broadcast_zero")
    tensors = {
        "zero": Tensor("zero", (1,), F32, constant=True, value=0.0),
        "out": Tensor("out", (8,), F32),
    }
    return kernel, tensors


def test_literal_scalar_inlines_in_scalar_write():
    kernel, tensors = _broadcast_zero_kernel(width=1)
    src = render_kernelop(kernel, tensors=tensors, literal_constants={"zero": 0.0})
    assert "out[" in src
    # The Write must reference the literal directly, not the SSA name.
    assert "= 0.0f;" in src
    # No undefined `float z = ...` decl AND no bare `= z;`.
    assert "float z" not in src
    assert "= z;" not in src


def test_literal_scalar_inlines_in_vector_write():
    kernel, tensors = _broadcast_zero_kernel(width=2)
    src = render_kernelop(kernel, tensors=tensors, literal_constants={"zero": 0.0})
    # The vector Write must pack literals, not the undefined SSA name.
    assert "make_float2(0.0f, 0.0f)" in src
    assert "make_float2(z, z)" not in src
    assert "float z" not in src
