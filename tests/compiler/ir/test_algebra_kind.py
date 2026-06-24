"""Bottom-up algebra classifier tests (Part B of
``plans/algebraic-carrier-analysis.md``).

`classify_algebra` / `Loop.algebra_kind` tags each reduce loop by reading its
carrier: a matmul cell → ``SEMIRING``, an associative `Accum` → ``MONOID``, a
recognized tuple `Monoid` → ``MONOID`` too (a twisted monoid is a monoid —
transport of structure; the streaming-flash schedule is selected structurally one
layer below, not by a distinct algebra kind), a non-reduce scope → ``MAP``. The
tag is a derived read — it never enters equality / `op_cache_key` and stays
consistent across a normalize round-trip.
"""

from __future__ import annotations

from deplodock.compiler.ir.algebra import AlgebraKind, classify_algebra, contains_matmul_reduce
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.loop import Accum, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Assign, Mma, Monoid
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY


def _matmul_loop(k_ext: int = 8) -> Loop:
    """``acc += a[m,k] * b[k,n]`` — a scalar matmul cell (Accum + multiply)."""
    return Loop(
        axis=Axis("k", k_ext),
        body=(
            Load(name="a", input="A", index=(Var("m"), Var("k"))),
            Load(name="b", input="B", index=(Var("k"), Var("n"))),
            Assign(name="p", op="multiply", args=("a", "b")),
            Accum(name="acc", value="p", op="add"),
        ),
    )


def _mma_loop() -> Loop:
    """The tensor-core fused form of the same cell (Mma carrier)."""
    return Loop(
        axis=Axis("k", 16),
        body=(
            Load(name="a0", input="A", index=(Var("m"), Var("k"))),
            Load(name="b0", input="B", index=(Var("k"), Var("n"))),
            Mma(a="a0", b="b0", c="acc", atom=ATOM_REGISTRY["mma_m16n8k16_f16"]),
        ),
    )


def _reduce_loop(op: str) -> Loop:
    return Loop(
        axis=Axis("k", 8),
        body=(Load(name="x", input="X", index=(Var("k"),)), Accum(name="s", value="x", op=op)),
    )


def _monoid_loop() -> Loop:
    combine = Monoid(
        state=("m_i", "l_i"),
        partial=("s",),
        merge=(
            Assign("mx", "maximum", ("m_i", "s")),
            Assign("l_i", "add", ("l_i", "mx")),
            Assign("m_i", "copy", ("mx",)),
        ),
        identity=(Literal(-1e30), Literal(0.0)),
        axes=("kv",),
    )
    return Loop(
        axis=Axis("kv", 8),
        body=(Load(name="s", input="S", index=(Var("kv"),)), combine),
    )


# --- the four kinds ----------------------------------------------------------


def test_matmul_cell_is_semiring():
    assert _matmul_loop().algebra_kind is AlgebraKind.SEMIRING


def test_mma_cell_is_semiring():
    assert _mma_loop().algebra_kind is AlgebraKind.SEMIRING


def test_sum_reduce_is_monoid():
    assert _reduce_loop("add").algebra_kind is AlgebraKind.MONOID


def test_max_reduce_is_monoid():
    assert _reduce_loop("maximum").algebra_kind is AlgebraKind.MONOID


def test_monoid_carrier_is_monoid():
    # A tuple `Monoid` carrier (online-softmax LSE) IS a monoid (transport of
    # structure), so it classifies as MONOID — same kind as a scalar `Accum`. The
    # streaming-flash schedule is structural (nested contraction), chosen below.
    assert _monoid_loop().algebra_kind is AlgebraKind.MONOID


def test_non_reduce_loop_is_map():
    free = Loop(
        axis=Axis("n", 8),
        body=(Load(name="x", input="X", index=(Var("n"),)), Write(output="O", index=(Var("n"),), value="x")),
    )
    assert free.algebra_kind is AlgebraKind.MAP


def test_two_load_max_reduce_is_not_semiring():
    # Two K-indexed loads but a max combine (no distributing product) — a monoid,
    # not a contraction. Guards against is_matmul_reduce over-firing.
    loop = Loop(
        axis=Axis("k", 8),
        body=(
            Load(name="a", input="A", index=(Var("m"), Var("k"))),
            Load(name="b", input="B", index=(Var("k"), Var("n"))),
            Assign(name="p", op="add", args=("a", "b")),  # add, not multiply → no semiring
            Accum(name="acc", value="p", op="maximum"),
        ),
    )
    assert loop.algebra_kind is AlgebraKind.MONOID


# --- derived-cache invariants ------------------------------------------------


def test_kind_survives_loopop_normalize_roundtrip():
    # Wrap the matmul cell in a LoopOp (runs normalize_body) and confirm the
    # classification of its reduce loop is unchanged — a derived read stays
    # consistent with the (re)normalized body.
    cell = _matmul_loop()
    op = LoopOp(
        body=(
            Loop(
                axis=Axis("m", 4),
                body=(Loop(axis=Axis("n", 4), body=(cell, Write(output="O", index=(Var("m"), Var("n")), value="acc"))),),
            ),
        )
    )
    reduce_loops = [s for s in op.body.iter() if isinstance(s, Loop) and s.is_reduce]
    assert len(reduce_loops) == 1
    assert reduce_loops[0].algebra_kind is AlgebraKind.SEMIRING
    # Reconstructing the same LoopOp re-normalizes; the classification holds.
    op2 = LoopOp(body=op.body)
    reduce_loops2 = [s for s in op2.body.iter() if isinstance(s, Loop) and s.is_reduce]
    assert reduce_loops2[0].algebra_kind is AlgebraKind.SEMIRING


def test_contains_matmul_reduce_recurses():
    # The shared helper finds a matmul reduce transitively nested under a
    # non-reduce loop (the fused-prologue / demoted-split probe shape).
    nest = Loop(
        axis=Axis("n", 4),
        body=(Loop(axis=Axis("m", 4), body=(_matmul_loop(),)), Write(output="O", index=(Var("n"),), value="x")),
    )
    assert contains_matmul_reduce(nest)
    # A pure pointwise / monoid nest has none.
    monoid_nest = Loop(axis=Axis("n", 4), body=(_reduce_loop("add"),))
    assert not contains_matmul_reduce(monoid_nest)


def test_kind_not_in_equality_or_op_cache_key():
    # The kind is computed, never stored — two structurally identical reduce
    # loops are equal and the tag adds no field to compare/hash.
    a, b = _matmul_loop(), _matmul_loop()
    assert a == b and hash(a) == hash(b)
    # classify is a pure function of the body, not a stored attribute.
    assert classify_algebra(a) is classify_algebra(b)
    assert "algebra" not in {f for f in a.__dataclass_fields__}
