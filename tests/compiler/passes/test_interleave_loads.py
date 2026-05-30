"""Pass-level unit tests for ``095_interleave_loads``.

The sink-Loads peephole is small enough to exercise on a synthetic
``Body`` of ``Load`` + ``Assign`` + ``Accum`` stmts — checks the
load-position re-emission directly without needing the full
compile pipeline or CUDA. The end-to-end accuracy regression for
this pass lives in ``test_knob_pinning::test_article_reproduction_configs``
(``interleave_loads_disabled`` row exercises the opt-out path).
"""

from __future__ import annotations

import importlib

from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Literal
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.stmt.leaves import Accum, Assign, Load

_pass = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.kernel.095_interleave_loads")


def _ld(name: str, buf: str) -> Load:
    return Load(name=name, input=buf, index=(Literal(0, "int"),))


def test_sink_loads_places_each_load_before_first_consumer() -> None:
    """Flat ``Load × N + Assign × M`` block — each Load gets sunk to
    just before its first consumer. Loads with no consumer in the
    block stay at the top in original order."""
    body = Body(
        (
            _ld("in0", "B"),  # consumed by v0..v3 (positions 6..9)
            _ld("in1", "A"),  # consumed by v0  (position 6)
            _ld("in2", "A"),  # consumed by v1  (position 7)
            _ld("in3", "A"),  # consumed by v2  (position 8)
            _ld("in4", "A"),  # consumed by v3  (position 9)
            _ld("in_unused", "X"),  # no consumer — stays at top
            Assign(name="v0", op=ElementwiseImpl("multiply"), args=("in0", "in1")),
            Assign(name="v1", op=ElementwiseImpl("multiply"), args=("in0", "in2")),
            Assign(name="v2", op=ElementwiseImpl("multiply"), args=("in0", "in3")),
            Assign(name="v3", op=ElementwiseImpl("multiply"), args=("in0", "in4")),
        )
    )

    out = _pass._sink_loads(body)
    seq = tuple(out)
    names = [s.name if isinstance(s, (Load, Assign)) else type(s).__name__ for s in seq]

    # Same length, no stmts dropped.
    assert len(seq) == 10
    # ``in_unused`` (no consumer) stays at the top.
    assert names[0] == "in_unused"
    # ``in0`` (B-broadcast, first consumer at v0) sinks before v0.
    # ``in1`` (a0, first consumer v0) also sinks before v0 — both share the
    # same first-consumer position; their relative order is the source order.
    # Expected layout:
    #   in_unused, in0, in1, v0, in2, v1, in3, v2, in4, v3
    assert names == [
        "in_unused",
        "in0",
        "in1",
        "v0",
        "in2",
        "v1",
        "in3",
        "v2",
        "in4",
        "v3",
    ]


def test_sink_loads_is_idempotent_on_already_interleaved_body() -> None:
    """A body already in interleaved layout (Load + immediate consumer)
    is a fixpoint — the second pass produces the same Body."""
    body = Body(
        (
            _ld("in0", "B"),
            _ld("in1", "A"),
            Assign(name="v0", op=ElementwiseImpl("multiply"), args=("in0", "in1")),
            _ld("in2", "A"),
            Assign(name="v1", op=ElementwiseImpl("multiply"), args=("in0", "in2")),
        )
    )
    once = _pass._sink_loads(body)
    twice = _pass._sink_loads(once)
    assert tuple(once) == tuple(twice)


def test_sink_loads_preserves_assign_and_accum_relative_order() -> None:
    """Sinking is load-only — non-Load stmts (Assigns, Accums) keep
    their relative order. Reorder of an Assign would change which
    accumulator gets which addend on subsequent K-loop iterations."""
    body = Body(
        (
            _ld("in0", "A"),  # consumer: v0
            _ld("in1", "A"),  # consumer: v1
            Assign(name="v0", op=ElementwiseImpl("multiply"), args=("in0", "in0")),
            Assign(name="v1", op=ElementwiseImpl("multiply"), args=("in1", "in1")),
            Accum(name="acc0", value="v0"),
            Accum(name="acc1", value="v1"),
        )
    )
    out = _pass._sink_loads(body)
    names = [s.name for s in out]
    # Non-Load order (v0, v1, acc0, acc1) preserved.
    assert names.index("v0") < names.index("v1")
    assert names.index("acc0") < names.index("acc1")
    # Each Load is sunk to before its consumer.
    assert names.index("in0") < names.index("v0")
    assert names.index("in1") < names.index("v1")
