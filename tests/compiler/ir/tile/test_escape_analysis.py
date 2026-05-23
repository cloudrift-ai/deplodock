"""Tests for ``deplodock.compiler.ir.tile.escape_analysis``.

Verifies the three queries (cooperative thread axes per Accum, atomic
axes per Write, broadcast guard axes per Write) match the answers
``001_coordination`` produces for canonical kernel shapes.
"""

from __future__ import annotations

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.loop import Accum, Assign, Load
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.stmt.leaves import Write
from deplodock.compiler.ir.tile.escape_analysis import analyze
from deplodock.compiler.ir.tile.ir import GridTile, SerialTile, ThreadTile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _index_expr(*names: str) -> tuple:
    return tuple(Var(n) for n in names)


def _flat_index(*parts: tuple[str, int]) -> tuple:
    """Build an Add chain: ``parts[0][0]*parts[0][1] + ...``. Used to build
    the canonical matmul output index ``M_b*BM + M_t``."""
    pieces = []
    for name, mul in parts:
        if mul == 1:
            pieces.append(Var(name))
        else:
            pieces.append(BinaryExpr("*", Var(name), Literal(mul, "int")))
    expr = pieces[0]
    for p in pieces[1:]:
        expr = BinaryExpr("+", expr, p)
    return (expr,)


# ---------------------------------------------------------------------------
# Pointwise — baseline (no Accum, no atomic, no cooperation)
# ---------------------------------------------------------------------------


def test_pointwise_no_coordination():
    """Plain pointwise: thread-partitioned Write, no reductions. Nothing
    needs coordination."""
    i = Axis("i", 4)
    j = Axis("j", 8)
    tile = ThreadTile(
        axes=(i, j),
        body=(
            Load(name="x_v", input="x", index=(Var("i"), Var("j"))),
            Write(output="o", index=(Var("i"), Var("j")), value="x_v"),
        ),
    )
    result = analyze(Body((tile,)))

    assert result.cooperative_thread_axes == frozenset()
    assert all(not result.atomic_axes(w) for w in result.writes)
    assert all(not result.broadcast_axes(w) for w in result.writes)


# ---------------------------------------------------------------------------
# Plain matmul — Accum escapes K_o into a Write that fully indexes the
# thread axes. No cooperation; no atomic; no broadcast.
# ---------------------------------------------------------------------------


def test_plain_matmul_no_cooperation():
    """Output-partition matmul: each thread covers a distinct (M_t, N_t)
    cell. ``acc`` escapes ``for K_o`` but the consumer Write references
    both M_t and N_t, so no thread axis is cooperative."""
    M_t = Axis("M_t", 16)
    N_t = Axis("N_t", 32)
    K_o = Axis("K_o", 64)
    tile = ThreadTile(
        axes=(M_t, N_t),
        body=(
            SerialTile(
                axis=K_o,
                body=(
                    Load(name="a_v", input="A", index=(Var("M_t"), Var("K_o"))),
                    Load(name="b_v", input="B", index=(Var("K_o"), Var("N_t"))),
                    Assign(name="ab", op="multiply", args=("a_v", "b_v")),
                    Accum(name="acc", value="ab", op=ElementwiseImpl("add")),
                ),
                kind="stage_inner",
            ),
            Write(output="C", index=(Var("M_t"), Var("N_t")), value="acc"),
        ),
    )
    result = analyze(Body((tile,)))

    assert result.accum_cooperative_axes["acc"] == frozenset()
    assert result.cooperative_thread_axes == frozenset()
    # Single Write, no GridTile wrapping ⇒ no atomic axes.
    (only_write,) = result.writes
    assert result.atomic_axes(only_write) == frozenset()
    assert result.broadcast_axes(only_write) == frozenset()


# ---------------------------------------------------------------------------
# Cooperative-K RMSNorm shape — single ThreadTile, Accum escapes the
# reduce loop, consumer Write references the thread axis only via
# striding (so by-name the axis is missing from the index).
# ---------------------------------------------------------------------------


def test_cooperative_rmsnorm_acc_escapes():
    """RMSNorm-style: 256 threads cooperatively reduce a 3584-wide row.

    Body shape:
        ThreadTile(t:256):
            for a1 in 0..3584:  # reduce — really a stride-by-256 over t
                in2 = load x[a1]
                v0 = multiply(in2, in2)
                acc <- add(acc, v0)
            v1 = divide(acc, n)            # post-reduce scalar — escape
            out_scalar[0] = v1             # broadcast Write (no t in index)

    Expected: ``acc`` is cooperative over ``t``; the post-reduce Write
    needs a ``Cond(t == 0)`` guard.
    """
    t = Axis("t", 256)
    a1 = Axis("a1", 3584)
    tile = ThreadTile(
        axes=(t,),
        body=(
            SerialTile(
                axis=a1,
                body=(
                    Load(name="in2", input="x", index=(Var("a1"),)),
                    Assign(name="v0", op="multiply", args=("in2", "in2")),
                    Accum(name="acc", value="v0", op=ElementwiseImpl("add"), axes=("t", "a1")),
                ),
                kind="stage_inner",
            ),
            Load(name="n_v", input="n", index=(Literal(0, "int"),)),
            Assign(name="v1", op="divide", args=("acc", "n_v")),
            Write(output="o", index=(Literal(0, "int"),), value="v1"),
        ),
    )
    result = analyze(Body((tile,)))

    assert result.accum_cooperative_axes["acc"] == frozenset({"t"})
    assert result.cooperative_thread_axes == frozenset({"t"})

    # The post-reduce Write needs a broadcast guard over t.
    (only_write,) = result.writes
    assert result.broadcast_axes(only_write) == frozenset({"t"})
    # No GridTile ⇒ no atomic axes.
    assert result.atomic_axes(only_write) == frozenset()


def test_cooperative_rmsnorm_write_inside_thread_loop_no_guard():
    """Variant: the post-reduce work writes a per-element output
    (StridedLoop over the row again). Those Writes reference the
    cooperative axis via the loop var, so no broadcast guard is needed.

    Body shape:
        ThreadTile(t:256):
            for a1 in 0..3584:  # reduce
                acc <- add(acc, ...)
            for a2 in 0..3584:  # free
                v = ... * acc
                out[a2] = v       # a2 doesn't reference t by name, but the
                                  # Write is per-row-element, not broadcast
    """
    t = Axis("t", 256)
    a1 = Axis("a1", 3584)
    a2 = Axis("a2", 3584)
    tile = ThreadTile(
        axes=(t,),
        body=(
            SerialTile(
                axis=a1,
                body=(
                    Load(name="in1", input="x", index=(Var("a1"),)),
                    Accum(name="acc", value="in1", op=ElementwiseImpl("add"), axes=("t", "a1")),
                ),
                kind="stage_inner",
            ),
            SerialTile(
                axis=a2,
                body=(
                    Load(name="in2", input="x", index=(Var("a2"),)),
                    Assign(name="v", op="multiply", args=("in2", "acc")),
                    Write(output="o", index=(Var("a2"),), value="v"),
                ),
                kind="plain",
            ),
        ),
    )
    result = analyze(Body((tile,)))

    # acc still escapes the reduce loop, and the Write doesn't reference
    # t — so by the structural rule, the Write IS broadcast-guarded over
    # t. (In the real RMSNorm dump this works because the Write index
    # references `a2` which is striped over `t` by the planner; that
    # stride relationship is lost at this IR level, so the helper
    # conservatively flags it. Coordination's `_write_indexed_by` does
    # the same conservative thing today, so this is the matching
    # behavior.)
    assert result.accum_cooperative_axes["acc"] == frozenset({"t"})
    (only_write,) = result.writes
    assert result.broadcast_axes(only_write) == frozenset({"t"})


# ---------------------------------------------------------------------------
# Split-K matmul — Write missing the SPLITK block axis ⇒ atomic.
# ---------------------------------------------------------------------------


def test_split_k_matmul_is_atomic():
    """Split-K matmul: K_s GridTile axis splits the reduce across 4 CTAs.
    The Write to C is indexed by M_b/N_b/M_t/N_t but NOT by K_s, so
    multiple CTAs race ⇒ atomic."""
    M_b = Axis("M_b", 8)
    N_b = Axis("N_b", 4)
    K_s = Axis("K_s", 4)
    M_t = Axis("M_t", 16)
    N_t = Axis("N_t", 32)
    K_o = Axis("K_o", 16)
    tile = GridTile(
        axes=(M_b, N_b, K_s),
        body=(
            ThreadTile(
                axes=(M_t, N_t),
                body=(
                    SerialTile(
                        axis=K_o,
                        body=(
                            Load(name="a_v", input="A", index=(Var("M_t"), Var("K_o"))),
                            Load(name="b_v", input="B", index=(Var("K_o"), Var("N_t"))),
                            Assign(name="ab", op="multiply", args=("a_v", "b_v")),
                            Accum(name="acc", value="ab", op=ElementwiseImpl("add")),
                        ),
                        kind="stage_inner",
                    ),
                    Write(
                        output="C",
                        index=_flat_index(("M_b", 16), ("M_t", 1)) + _flat_index(("N_b", 32), ("N_t", 1)),
                        value="acc",
                    ),
                ),
            ),
        ),
    )
    result = analyze(Body((tile,)))

    (only_write,) = result.writes
    assert result.atomic_axes(only_write) == frozenset({"K_s"})
    # No cooperative threads — Write index references both M_t and N_t.
    assert result.accum_cooperative_axes["acc"] == frozenset()
    assert result.broadcast_axes(only_write) == frozenset()


def test_plain_matmul_in_grid_no_atomic():
    """Same shape as split-K but without the K_s axis — Write fully
    covers M_b/N_b ⇒ no atomic needed."""
    M_b = Axis("M_b", 8)
    N_b = Axis("N_b", 4)
    M_t = Axis("M_t", 16)
    N_t = Axis("N_t", 32)
    K_o = Axis("K_o", 16)
    tile = GridTile(
        axes=(M_b, N_b),
        body=(
            ThreadTile(
                axes=(M_t, N_t),
                body=(
                    SerialTile(
                        axis=K_o,
                        body=(
                            Load(name="a_v", input="A", index=(Var("M_t"), Var("K_o"))),
                            Load(name="b_v", input="B", index=(Var("K_o"), Var("N_t"))),
                            Assign(name="ab", op="multiply", args=("a_v", "b_v")),
                            Accum(name="acc", value="ab", op=ElementwiseImpl("add")),
                        ),
                        kind="stage_inner",
                    ),
                    Write(
                        output="C",
                        index=_flat_index(("M_b", 16), ("M_t", 1)) + _flat_index(("N_b", 32), ("N_t", 1)),
                        value="acc",
                    ),
                ),
            ),
        ),
    )
    result = analyze(Body((tile,)))

    (only_write,) = result.writes
    assert result.atomic_axes(only_write) == frozenset()
    assert result.accum_cooperative_axes["acc"] == frozenset()
