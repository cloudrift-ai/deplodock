"""Tests for ``Body._structural_key`` and the two new normalization
passes (``sort_commutative_args``, ``canonicalize_buffer_names``).

The structural key is the canonical text rendering used for dedup
queries — two bodies that differ only by SSA / axis names, commutative
arg order, or external-buffer names must produce the same key.
"""

from __future__ import annotations

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.stmt.blocks import Loop
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.stmt.leaves import Assign, Load, Write
from deplodock.compiler.ir.stmt.normalize import (
    canonicalize_buffer_names,
    normalize_body,
    sort_commutative_args,
)

# ---------------------------------------------------------------------------
# sort_commutative_args
# ---------------------------------------------------------------------------


def test_sort_commutative_args_orders_add() -> None:
    body = Body((Assign(name="v0", op="add", args=("y", "x")),))
    out = sort_commutative_args(body)
    assert out[0].args == ("x", "y")


def test_sort_commutative_args_leaves_subtract_alone() -> None:
    body = Body((Assign(name="v0", op="subtract", args=("y", "x")),))
    out = sort_commutative_args(body)
    assert out[0].args == ("y", "x")


def test_sort_commutative_args_recurses_into_loop() -> None:
    a = Axis("a", 4)
    body = Body((Loop(axis=a, body=(Assign(name="v0", op="multiply", args=("y", "x")),)),))
    out = sort_commutative_args(body)
    assert out[0].body[0].args == ("x", "y")


def test_sort_commutative_args_idempotent() -> None:
    body = Body((Assign(name="v0", op="add", args=("y", "x")),))
    once = sort_commutative_args(body)
    twice = sort_commutative_args(once)
    assert tuple(once) == tuple(twice)


# ---------------------------------------------------------------------------
# canonicalize_buffer_names
# ---------------------------------------------------------------------------


def test_canonicalize_buffer_names_renames_loads_and_writes() -> None:
    a = Axis("a", 4)
    body = Body(
        (
            Loop(
                axis=a,
                body=(
                    Load(name="x", input="my_input", index=(Var("a"),)),
                    Write(output="my_output", index=(Var("a"),), value="x"),
                ),
            ),
        )
    )
    out = canonicalize_buffer_names(body)
    inner = tuple(out[0].body)
    assert isinstance(inner[0], Load) and inner[0].input == "b0"
    assert isinstance(inner[1], Write) and inner[1].output == "b1"


def test_canonicalize_buffer_names_first_seen_wins() -> None:
    """Encounter order via ``Body.iter`` — the first reference to each
    buffer name fixes its canonical id."""
    body = Body(
        (
            Load(name="x", input="Z", index=(Var("a"),)),
            Load(name="y", input="A", index=(Var("a"),)),
            Load(name="z", input="Z", index=(Var("a"),)),  # reuse Z
        )
    )
    out = canonicalize_buffer_names(body)
    assert out[0].input == "b0"  # Z first
    assert out[1].input == "b1"  # A second
    assert out[2].input == "b0"  # Z again


def test_canonicalize_buffer_names_already_canonical_returns_same() -> None:
    body = Body((Load(name="x", input="b0", index=()),))
    out = canonicalize_buffer_names(body)
    assert out is body


# ---------------------------------------------------------------------------
# Body._structural_key
# ---------------------------------------------------------------------------


def _matmul_body(input_x: str, input_y: str, output: str) -> Body:
    """Tiny multiply-and-write body parameterized by buffer names."""
    a = Axis("a", 4)
    return Body(
        (
            Loop(
                axis=a,
                body=(
                    Load(name="x", input=input_x, index=(Var("a"),)),
                    Load(name="y", input=input_y, index=(Var("a"),)),
                    Assign(name="z", op="multiply", args=("x", "y")),
                    Write(output=output, index=(Var("a"),), value="z"),
                ),
            ),
        )
    )


def test_structural_key_equal_for_renamed_buffers() -> None:
    a = _matmul_body("X", "Y", "O")
    b = _matmul_body("foo", "bar", "baz")
    assert a._structural_key == b._structural_key


def test_structural_key_equal_for_swapped_commutative_args() -> None:
    a = Axis("a", 4)
    body_xy = Body(
        (
            Loop(
                axis=a,
                body=(
                    Load(name="lx", input="X", index=(Var("a"),)),
                    Load(name="ly", input="Y", index=(Var("a"),)),
                    Assign(name="z", op="add", args=("lx", "ly")),
                    Write(output="O", index=(Var("a"),), value="z"),
                ),
            ),
        )
    )
    body_yx = Body(
        (
            Loop(
                axis=a,
                body=(
                    Load(name="ly", input="Y", index=(Var("a"),)),
                    Load(name="lx", input="X", index=(Var("a"),)),
                    Assign(name="z", op="add", args=("ly", "lx")),
                    Write(output="O", index=(Var("a"),), value="z"),
                ),
            ),
        )
    )
    assert body_xy._structural_key == body_yx._structural_key


def test_structural_key_distinguishes_subtract_from_add() -> None:
    a = Axis("a", 4)

    def make(op: str) -> Body:
        return Body(
            (
                Loop(
                    axis=a,
                    body=(
                        Load(name="x", input="X", index=(Var("a"),)),
                        Load(name="y", input="Y", index=(Var("a"),)),
                        Assign(name="z", op=op, args=("x", "y")),
                        Write(output="O", index=(Var("a"),), value="z"),
                    ),
                ),
            )
        )

    assert make("add")._structural_key != make("subtract")._structural_key


def test_structural_key_distinguishes_swapped_noncommutative_args() -> None:
    """``subtract(x, y)`` and ``subtract(y, x)`` are NOT structurally
    equal — the sort pass must skip non-commutative ops."""
    a = Axis("a", 4)

    def make(args: tuple[str, str]) -> Body:
        return Body(
            (
                Loop(
                    axis=a,
                    body=(
                        Load(name="x", input="X", index=(Var("a"),)),
                        Load(name="y", input="Y", index=(Var("a"),)),
                        Assign(name="z", op="subtract", args=args),
                        Write(output="O", index=(Var("a"),), value="z"),
                    ),
                ),
            )
        )

    assert make(("x", "y"))._structural_key != make(("y", "x"))._structural_key


def test_structural_key_idempotent() -> None:
    """Round-tripping a body through ``normalize_body(canonical_buffers=True)``
    fixes its structural key — re-querying yields the same string."""
    body = _matmul_body("X", "Y", "O")
    k1 = body._structural_key
    renormalized = Body(normalize_body(body, hoist=False, canonical_buffers=True))
    assert renormalized._structural_key == k1


def test_structural_key_is_string_and_hashable() -> None:
    body = _matmul_body("X", "Y", "O")
    key = body._structural_key
    assert isinstance(key, str)
    assert hash(key) == hash(body._structural_key)  # cached, deterministic


def test_structural_key_cached_property() -> None:
    """Accessing twice returns the *same* string object (cached)."""
    body = _matmul_body("X", "Y", "O")
    a = body._structural_key
    b = body._structural_key
    assert a is b
