"""``FlashCombine`` streaming-reduce lowering — CPU forward parity (Step 2).

The carrier's per-iteration LSE rescale (``m_new = max(m, s); alpha = exp(m −
m_new); l = l·alpha + p; O = O·alpha + p·v; m = m_new``) renders to plain
assignments against ``Init``-declared carried scalars. These tests build a
hand-written ``LoopOp`` that streams the recurrence and check ``LoopOp.forward``
(the cppyy-JIT'd C++ reference) against a numpy online-softmax / weighted-average
reference — no GPU, no matmul nesting, the scalar core of flash attention.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.loop.ir import LoopOp
from deplodock.compiler.ir.stmt import Assign, FlashCombine, Init, Load, Loop, Write


def _softmax_loopop(n: int) -> LoopOp:
    """Online softmax of a vector ``x`` → ``out`` (length n).

    Streaming sweep carries ``(m, l)`` via ``FlashCombine``; a second free sweep
    divides ``exp(x − m) / l``. The two sibling sweeps over the same axis are the
    classic safe-softmax-in-one-pass shape — the reduce sweep must run first
    (a swap would read the identity m=−inf, l=0).
    """
    return LoopOp(
        body=(
            Init(name="m", op=ElementwiseImpl("maximum"), dtype="f32"),
            Init(name="l", op=ElementwiseImpl("add"), dtype="f32"),
            Loop(
                axis=Axis(name="j", extent=Dim(n)),
                body=(
                    Load(name="s", input="x", index=(Var("j"),)),
                    FlashCombine(state=("m", "l"), partial=("s",), axes=("j",)),
                ),
            ),
            Loop(
                axis=Axis(name="j", extent=Dim(n)),
                body=(
                    Load(name="s2", input="x", index=(Var("j"),)),
                    Assign(name="d", op="subtract", args=("s2", "m")),
                    Assign(name="e", op="exp", args=("d",)),
                    Assign(name="o", op="divide", args=("e", "l")),
                    Write(output="out", index=(Var("j"),), value="o"),
                ),
            ),
        )
    )


def _weighted_avg_loopop(n: int) -> LoopOp:
    """1-D attention: ``out = softmax(x) @ v`` (scalar), the full ``(m, l, O)``
    triple with a scalar value — ``O += p·v`` rides the same ``p`` the denom
    uses. Exercises the complete carried-tuple + rescale machinery with no MMA.
    """
    return LoopOp(
        body=(
            Init(name="m", op=ElementwiseImpl("maximum"), dtype="f32"),
            Init(name="l", op=ElementwiseImpl("add"), dtype="f32"),
            Init(name="acc", op=ElementwiseImpl("add"), dtype="f32"),
            Loop(
                axis=Axis(name="j", extent=Dim(n)),
                body=(
                    Load(name="s", input="x", index=(Var("j"),)),
                    Load(name="vj", input="v", index=(Var("j"),)),
                    FlashCombine(state=("m", "l", "acc"), partial=("s", "vj"), axes=("j",)),
                ),
            ),
            Assign(name="res", op="divide", args=("acc", "l")),
            Write(output="out", index=(Literal(0, "int"),), value="res"),
        )
    )


@pytest.mark.parametrize("n", [1, 4, 8, 33])
def test_flash_combine_online_softmax_matches_numpy(n: int) -> None:
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(n) * 4.0).astype(np.float32)  # wide range → exercises rescale
    out = _softmax_loopop(n).forward(x).flatten()
    ref = np.exp(x - x.max())
    ref = ref / ref.sum()
    np.testing.assert_allclose(out, ref, atol=1e-5)


@pytest.mark.parametrize("n", [1, 4, 16])
def test_flash_combine_weighted_average_matches_numpy(n: int) -> None:
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(n) * 3.0).astype(np.float32)
    v = rng.standard_normal(n).astype(np.float32)
    out = _weighted_avg_loopop(n).forward(x, v).flatten()
    p = np.exp(x - x.max())
    p = p / p.sum()
    ref = float((p * v).sum())
    np.testing.assert_allclose(out[0], ref, atol=1e-5)


def test_flash_combine_reduce_loop_recognized() -> None:
    """The streaming sweep is a reduce loop via the ReduceCarrier protocol — the
    pipeline's reduce-axis machinery keys off this with no FlashCombine-specific
    isinstance ladder."""
    op = _softmax_loopop(8)
    reduce_loops = [s for s in op.body if isinstance(s, Loop) and s.is_reduce]
    assert len(reduce_loops) == 1  # the (m, l) streaming sweep; the divide sweep is free
