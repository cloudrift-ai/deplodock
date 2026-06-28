"""``Monoid`` carrier — CPU forward parity (the streaming-reduce render).

A :class:`Monoid` carries internal state across a reduce loop and folds each partial via
its ``merge`` program (the associative operation as data). At lowering a carrier
:meth:`~deplodock.compiler.ir.stmt.algebra.Monoid.dissolve`\\ s into its fold ``Accum``\\ s
(``Loop.render`` seeds those from ``op.identity``); these tests stream that dissolved form —
the ``(m, l)`` online-softmax normalization and the full ``(m, l, O)`` weighted average —
plus the cross-partition ``combine_states`` state-merge (the one form still rendered via
``Monoid.render``), and check ``LoopOp.forward`` (the cppyy-JIT'd C++ reference) against
numpy, with no GPU and no matmul nesting.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.loop.ir import LoopOp
from deplodock.compiler.ir.stmt import Assign, Load, Loop, Monoid, State, Twist, Write
from deplodock.compiler.pipeline.passes.lowering.tile._flash import flash_combine
from deplodock.compiler.pipeline.passes.lowering.tile._softmax import online_softmax_combine


def _softmax_loopop(n: int) -> LoopOp:
    """Online softmax of a vector ``x`` → ``out``: the ``(m, l)`` streaming carrier
    (``online_softmax_combine``) dissolved into its fold ``Accum``\\ s, then a second free
    sweep dividing ``exp(x − m) / l``. The reduce sweep must run first (a swap would read
    the identity m=−inf, l=0). ``Loop.render`` seeds the fold ``Accum``\\ s — no explicit
    Init, no Monoid stmt."""
    combine = online_softmax_combine("m", "l", "s")
    return LoopOp(
        body=(
            Loop(
                axis=Axis(name="j", extent=Dim(n)),
                body=(
                    Load(name="s", input="x", index=(Var("j"),)),
                    *combine.dissolve(),  # the (m, l) fold Accums + ψ-rescale temps
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
    """1-D attention ``out = softmax(x) @ v`` (scalar): the full ``(m, l, O)`` flash carrier
    dissolved into its fold ``Accum``\\ s — ``O += p·v`` rides the same ``p`` the denom uses."""
    combine = flash_combine("m", "l", "acc", "s", "vj")
    return LoopOp(
        body=(
            Loop(
                axis=Axis(name="j", extent=Dim(n)),
                body=(
                    Load(name="s", input="x", index=(Var("j"),)),
                    Load(name="vj", input="v", index=(Var("j"),)),
                    *combine.dissolve(),  # the (m, l, acc) fold Accums + ψ-rescale temps
                ),
            ),
            Assign(name="res", op="divide", args=("acc", "l")),
            Write(output="out", index=(Literal(0, "int"),), value="res"),
        )
    )


def _streaming_half(m: str, ll: str, acc: str, x_buf: str, v_buf: str, lo: int, n: int, jname: str) -> tuple:
    """Streaming-reduce loop folding ``x[lo:n]`` / ``v[lo:n]`` into the ``(m, l, acc)`` state
    via the flash carrier dissolved into its fold ``Accum``\\ s — one partition of a
    two-partition split reduce. ``Loop.render`` seeds the fold ``Accum``\\ s (no explicit
    Init)."""
    idx = Var(jname) if lo == 0 else BinaryExpr("+", Var(jname), Literal(lo, "int"))
    combine = flash_combine(m, ll, acc, f"s_{jname}", f"v_{jname}")
    return (
        Loop(
            axis=Axis(name=jname, extent=Dim(n - lo)),
            body=(
                Load(name=f"s_{jname}", input=x_buf, index=(idx,)),
                Load(name=f"v_{jname}", input=v_buf, index=(idx,)),
                *combine.dissolve(),
            ),
        ),
    )


def _two_partition_loopop(n: int) -> LoopOp:
    """1-D attention ``out = softmax(x) @ v`` computed as TWO independent
    partition reduces merged by the flash ``Monoid``'s ``combine_states`` — the
    cross-partition (split-KV / split-K) form, with no atomics. Partition A folds
    the first half, partition B the second; one ``combine_states`` merges the two
    fully-reduced ``(m, l, O)`` states, then the weighted average finalizes."""
    h = n // 2
    state_merge = flash_combine("m", "l", "acc", "s", "v").as_state_merge(("m_o", "l_o", "acc_o"))
    return LoopOp(
        body=(
            *_streaming_half("m", "l", "acc", "x", "v", 0, h, "ja"),
            *_streaming_half("m_o", "l_o", "acc_o", "x", "v", h, n, "jb"),
            state_merge,  # merge the two partition states: (m, l, acc) <- combine((m,l,acc), (m_o,l_o,acc_o))
            Assign(name="res", op="divide", args=("acc", "l")),
            Write(output="out", index=(Literal(0, "int"),), value="res"),
        )
    )


@pytest.mark.parametrize("n", [2, 5, 16, 33])
def test_combine_states_two_partition_matches_numpy(n: int) -> None:
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(n) * 3.0).astype(np.float32)  # wide range → cross-partition rescale
    v = rng.standard_normal(n).astype(np.float32)
    out = _two_partition_loopop(n).forward(x, v).flatten()
    p = np.exp(x - x.max())
    p = p / p.sum()
    ref = float((p * v).sum())
    np.testing.assert_allclose(out[0], ref, atol=1e-5)


def test_combine_states_default_derived_for_additive() -> None:
    """An additive carrier (partial lifts to a state) auto-derives
    ``combine_states`` from ``merge`` — partial reads swapped for ``state_b``."""
    c = Monoid(
        state=State(names=("acc",)),
        partial=(),  # the partial ``p`` is read off the merge program
        twist=Twist(merge=(Assign("acc", "add", ("acc", "p")),)),
    )
    assert c.partial_names() == ("p",)  # derived from the merge's external read
    assert c.twist.state_b == ("acc__o",)
    assert len(c.twist.combine_states) == 1
    assert c.twist.combine_states[0].name == "acc"
    assert c.twist.combine_states[0].args == ("acc", "acc__o")  # partial p → state_b acc__o


@pytest.mark.parametrize("n", [1, 4, 8, 33])
def test_combine_online_softmax_matches_numpy(n: int) -> None:
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(n) * 4.0).astype(np.float32)  # wide range → exercises the rescale
    out = _softmax_loopop(n).forward(x).flatten()
    ref = np.exp(x - x.max())
    ref = ref / ref.sum()
    np.testing.assert_allclose(out, ref, atol=1e-5)


@pytest.mark.parametrize("n", [1, 4, 16])
def test_combine_weighted_average_matches_numpy(n: int) -> None:
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(n) * 3.0).astype(np.float32)
    v = rng.standard_normal(n).astype(np.float32)
    out = _weighted_avg_loopop(n).forward(x, v).flatten()
    p = np.exp(x - x.max())
    p = p / p.sum()
    ref = float((p * v).sum())
    np.testing.assert_allclose(out[0], ref, atol=1e-5)


def test_combine_reduce_loop_recognized() -> None:
    """The streaming sweep is a reduce loop because its body holds a carrier — the
    pipeline's reduce-axis machinery keys off this with no Monoid-specific
    isinstance ladder."""
    op = _softmax_loopop(8)
    reduce_loops = [s for s in op.body if isinstance(s, Loop) and s.is_reduce]
    assert len(reduce_loops) == 1  # the (m, l) streaming sweep; the divide sweep is free
