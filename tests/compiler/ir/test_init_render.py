"""Tests for the ``Init`` Stmt — explicit accumulator declaration that
suppresses the renderer's default per-Loop Accum init.

Used by matmul-style materialization to handle nested-reduce shapes
(``Loop(k_o) > Loop(k_i) > Accum``) without resetting ``acc`` per
outer-loop iteration.
"""

from __future__ import annotations

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.axis import BIND_THREAD, BoundAxis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.kernel.ir import KernelOp, Tile
from deplodock.compiler.ir.kernel.render import render_kernelop
from deplodock.compiler.ir.loop import Accum, Axis, Load, Loop, Write
from deplodock.compiler.ir.stmt import Init


def _kernel_with_init_then_nested_reduce() -> KernelOp:
    """Models the shape produced by matmul materialization: Init at the
    Tile scope, then a Loop(k_o) > Loop(k_i) > Accum chain."""
    k_o = Axis("k_o", 4)
    k_i = Axis("k_i", 16)
    inner = (
        Load(name="a", input="A", index=(Var("k_o"), Var("k_i"))),
        Accum(name="acc", value="a", op=ElementwiseImpl("add")),
    )
    body = (
        Init(name="acc", op=ElementwiseImpl("add"), dtype=F32),
        Loop(axis=k_o, body=(Loop(axis=k_i, body=inner),)),
        Write(output="C", index=(), value="acc"),
    )
    encl = Tile(axes=(BoundAxis(axis=Axis("t0", 1), bind=BIND_THREAD),), body=body)
    return KernelOp(body=(encl,), name="k_init_test")


def test_init_emits_declaration_at_its_scope():
    src = render_kernelop(_kernel_with_init_then_nested_reduce(), shapes={"A": (4, 16), "C": ()})
    # The Init appears at Tile scope (above the for loops).
    lines = src.splitlines()
    init_line = next(i for i, line in enumerate(lines) if "float acc = 0.0f;" in line)
    k_o_line = next(i for i, line in enumerate(lines) if "for (int k_o" in line)
    k_i_line = next(i for i, line in enumerate(lines) if "for (int k_i" in line)
    assert init_line < k_o_line < k_i_line


def test_init_suppresses_default_accum_init():
    """The k_i Loop's Accum should NOT trigger a second ``float acc = ...``."""
    src = render_kernelop(_kernel_with_init_then_nested_reduce(), shapes={"A": (4, 16), "C": ()})
    # Only ONE init line for acc, and acc is declared exactly once.
    init_count = src.count("float acc = 0.0f;")
    assert init_count == 1


def _kernel_with_fp32_accum_over_fp16_load() -> tuple[KernelOp, dict]:
    """fp16 input + f32 accumulator. Reads ``A`` as ``__half``, accumulates
    in ``float`` with a ``__half2float`` insertion at the combine. Returns
    the kernel plus the ``tensors=`` map render needs to stamp A as F16."""
    from deplodock.compiler.tensor import Tensor  # noqa: PLC0415

    k = Axis("k", 8)
    inner = (
        Load(name="a", input="A", index=(Var("k"),)),
        Accum(name="acc", value="a", op=ElementwiseImpl("add"), dtype=F32),
    )
    body = (
        Init(name="acc", op=ElementwiseImpl("add"), dtype=F32),
        Loop(axis=k, body=inner),
        Write(output="C", index=(), value="acc"),
    )
    encl = Tile(axes=(BoundAxis(axis=Axis("t0", 1), bind=BIND_THREAD),), body=body)
    from deplodock.compiler import dtype as _dt  # noqa: PLC0415

    kop = KernelOp(body=(encl,), name="k_fp32_acc_fp16")
    tensors = {"A": Tensor("A", (8,), _dt.F16), "C": Tensor("C", (), _dt.F32)}
    return kop, tensors


def test_fp32_accumulator_over_fp16_loads_renders_conversion_at_combine():
    kop, tensors = _kernel_with_fp32_accum_over_fp16_load()
    src = render_kernelop(kop, tensors=tensors)
    # Accumulator declared as ``float`` (via Init), input loaded as
    # ``__half``, combine wraps the value with ``__half2float``.
    assert "float acc = 0.0f;" in src, src
    assert "__half a =" in src, src
    assert "acc += __half2float(a);" in src, src


def test_no_init_falls_back_to_default_per_loop_init():
    """Without Init, the inner Loop emits its own init — preserving prior behavior."""
    k_o = Axis("k_o", 4)
    k_i = Axis("k_i", 16)
    inner = (
        Load(name="a", input="A", index=(Var("k_o"), Var("k_i"))),
        Accum(name="acc", value="a", op=ElementwiseImpl("add")),
    )
    body = (
        Loop(axis=k_o, body=(Loop(axis=k_i, body=inner),)),
        Write(output="C", index=(), value="acc"),
    )
    encl = Tile(axes=(BoundAxis(axis=Axis("t0", 1), bind=BIND_THREAD),), body=body)
    src = render_kernelop(KernelOp(body=(encl,), name="k_no_init"), shapes={"A": (4, 16), "C": ()})
    assert src.count("float acc = 0.0f;") == 1  # emitted by the k_i Loop (the immediate parent)
