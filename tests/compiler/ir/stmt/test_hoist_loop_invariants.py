"""Tests for ``hoist_loop_invariants`` in ``stmt/normalize.py``.

Builds bodies by hand and asserts on the post-hoist structure so failures
point at the pass itself rather than upstream lowering.
"""

from __future__ import annotations

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt.blocks import Loop, StridedLoop
from deplodock.compiler.ir.stmt.leaves import Accum, Assign, Load, Write
from deplodock.compiler.ir.stmt.normalize import hoist_loop_invariants


def test_hoists_invariant_load_above_loop() -> None:
    a = Axis("a", 4)
    body = (Loop(axis=a, body=(Load(name="x", input="X", index=()), Write(output="o", index=(Var("a"),), value="x"))),)

    out = hoist_loop_invariants(body)

    assert len(out) == 2
    assert isinstance(out[0], Load) and out[0].name == "x"
    assert isinstance(out[1], Loop) and out[1].axis.name == "a"
    inner = tuple(out[1].body)
    assert len(inner) == 1 and isinstance(inner[0], Write)


def test_hoists_inner_loop_and_consumer_together() -> None:
    """The motivating case: an inner reduce-Loop is invariant in the outer
    axis, and a downstream Assign consumes its Accum result. Both must move
    above the outer Loop together — hoisting only the Assign would leave it
    referencing an Accum still defined inside the outer Loop body."""
    a = Axis("a", 4)
    b = Axis("b", 8)
    body = (
        Loop(
            axis=a,
            body=(
                Loop(
                    axis=b,
                    body=(
                        Load(name="xb", input="X", index=(Var("b"),)),
                        Accum(name="acc", value="xb"),
                    ),
                ),
                Assign(name="v", op="exp", args=("acc",)),
                Write(output="out", index=(Var("a"),), value="v"),
            ),
        ),
    )

    out = hoist_loop_invariants(body)

    assert len(out) == 3, f"expected [Loop(b), Assign v, Loop(a)], got {[type(s).__name__ for s in out]}"
    assert isinstance(out[0], Loop) and out[0].axis.name == "b"
    assert isinstance(out[1], Assign) and out[1].name == "v" and out[1].args == ("acc",)
    assert isinstance(out[2], Loop) and out[2].axis.name == "a"
    inner_a = tuple(out[2].body)
    assert len(inner_a) == 1 and isinstance(inner_a[0], Write)
    inner_b = tuple(out[0].body)
    assert any(isinstance(s, Accum) and s.name == "acc" for s in inner_b)


def test_keeps_inner_loop_when_it_references_outer_axis() -> None:
    a = Axis("a", 4)
    b = Axis("b", 8)
    body = (
        Loop(
            axis=a,
            body=(
                Loop(
                    axis=b,
                    body=(
                        Load(name="xab", input="X", index=(Var("a"), Var("b"))),
                        Accum(name="acc", value="xab"),
                    ),
                ),
                Write(output="out", index=(Var("a"),), value="acc"),
            ),
        ),
    )

    out = hoist_loop_invariants(body)

    assert len(out) == 1
    assert isinstance(out[0], Loop) and out[0].axis.name == "a"
    inner = tuple(out[0].body)
    assert any(isinstance(s, Loop) and s.axis.name == "b" for s in inner)


def test_hoists_invariant_load_above_strided_loop() -> None:
    """A ``StridedLoop`` is the same kind of sequential axis-binder as ``Loop``;
    hoisting partitions its body the same way."""
    a = Axis("a", 4)
    body = (
        StridedLoop(
            axis=a,
            start=Literal(0, "int"),
            step=Literal(1, "int"),
            body=(Load(name="x", input="X", index=()), Write(output="o", index=(Var("a"),), value="x")),
        ),
    )

    out = hoist_loop_invariants(body)

    assert len(out) == 2
    assert isinstance(out[0], Load) and out[0].name == "x"
    assert isinstance(out[1], StridedLoop) and out[1].axis.name == "a"
    inner = tuple(out[1].body)
    assert len(inner) == 1 and isinstance(inner[0], Write)


def test_hoists_inner_strided_loop_and_consumer_together() -> None:
    """Same shape as the Loop-block test, but the inner reduce is a ``StridedLoop``."""
    a = Axis("a", 4)
    b = Axis("b", 8)
    body = (
        Loop(
            axis=a,
            body=(
                StridedLoop(
                    axis=b,
                    start=Literal(0, "int"),
                    step=Literal(1, "int"),
                    body=(
                        Load(name="xb", input="X", index=(Var("b"),)),
                        Accum(name="acc", value="xb"),
                    ),
                ),
                Assign(name="v", op="exp", args=("acc",)),
                Write(output="out", index=(Var("a"),), value="v"),
            ),
        ),
    )

    out = hoist_loop_invariants(body)

    assert len(out) == 3, f"expected [StridedLoop(b), Assign v, Loop(a)], got {[type(s).__name__ for s in out]}"
    assert isinstance(out[0], StridedLoop) and out[0].axis.name == "b"
    assert isinstance(out[1], Assign) and out[1].name == "v" and out[1].args == ("acc",)
    assert isinstance(out[2], Loop) and out[2].axis.name == "a"
    inner_a = tuple(out[2].body)
    assert len(inner_a) == 1 and isinstance(inner_a[0], Write)


def test_does_not_hoist_block_containing_write() -> None:
    """Inner Loop has no axis dependency on the outer axis but contains a
    Write — hoisting would change observable side-effect count, so it stays."""
    a = Axis("a", 4)
    b = Axis("b", 8)
    body = (
        Loop(
            axis=a,
            body=(
                Loop(
                    axis=b,
                    body=(
                        Load(name="xb", input="X", index=(Var("b"),)),
                        Write(output="scratch", index=(Var("b"),), value="xb"),
                    ),
                ),
                Write(output="out", index=(Var("a"),), value="xb"),
            ),
        ),
    )

    out = hoist_loop_invariants(body)

    assert len(out) == 1 and isinstance(out[0], Loop) and out[0].axis.name == "a"
    inner = tuple(out[0].body)
    assert any(isinstance(s, Loop) and s.axis.name == "b" for s in inner)
