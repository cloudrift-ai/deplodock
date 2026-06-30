"""Tests for ``unify_sibling_reduce_axes`` (relaxed overlap matching) and
``merge_sibling_reduce_loops`` in ``stmt/normalize.py``.

Builds bodies by hand and asserts on the post-pass structure so failures
point at the pass itself rather than upstream lowering. The motivating
case is the gated-MLP pattern ``silu(x @ Wg) * (x @ Wu)`` where the two
sibling reduce-loops index ``x`` at the same K position but bring in
distinct weight tensors — unify must group them on the shared x slot,
then merge must concatenate the bodies into one K-loop.
"""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Load, Write
from emmy.compiler.ir.stmt.normalize import (
    merge_sibling_reduce_loops,
    unify_sibling_reduce_axes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matmul_reduce(axis: Axis, w_buffer: str, out_acc: str, k_load: str = "x") -> Loop:
    """Build a single reduce-Loop body: ``acc += W[k, n] * x[k]``.

    Used to assemble the gated-MLP / matmul-sibling patterns the new
    passes target. Inner SSA names are namespaced by ``out_acc`` so two
    instances composed in one parent body have disjoint defs.
    """
    return Loop(
        axis=axis,
        body=(
            Load(name=f"w_{out_acc}", input=w_buffer, index=(Var(axis.name), Var("n"))),
            Load(name=f"x_{out_acc}", input=k_load, index=(Var(axis.name),)),
            Assign(name=f"m_{out_acc}", op="multiply", args=(f"w_{out_acc}", f"x_{out_acc}")),
            Accum(name=out_acc, value=f"m_{out_acc}"),
        ),
    )


# ---------------------------------------------------------------------------
# unify_sibling_reduce_axes — overlap-based grouping
# ---------------------------------------------------------------------------


def test_unify_groups_loops_with_shared_load_position() -> None:
    """Two reduce-loops over different axis names but both indexing ``x``
    at the same dim must end up sharing one canonical axis name."""
    a = Axis("a2", 16)
    b = Axis("a3", 16)
    body = Body(
        (
            _matmul_reduce(a, w_buffer="Wg", out_acc="acc0"),
            _matmul_reduce(b, w_buffer="Wu", out_acc="acc1"),
        )
    )

    out = unify_sibling_reduce_axes(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert len(loops) == 2
    assert loops[0].axis.name == loops[1].axis.name == "a2"


def test_unify_skips_when_extents_differ() -> None:
    """Overlap on (source, dim) is necessary but not sufficient: extents
    must also match — different extents mean different iteration spaces
    even if positions overlap."""
    a = Axis("a2", 16)
    b = Axis("a3", 32)
    body = Body(
        (
            _matmul_reduce(a, w_buffer="Wg", out_acc="acc0"),
            _matmul_reduce(b, w_buffer="Wu", out_acc="acc1"),
        )
    )

    out = unify_sibling_reduce_axes(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert loops[0].axis.name == "a2"
    assert loops[1].axis.name == "a3"


def test_unify_skips_disjoint_load_positions() -> None:
    """Two reduce-loops with no shared (source, dim) overlap stay in
    distinct groups — they index unrelated tensors at the reduce axis."""
    a = Axis("a2", 16)
    b = Axis("a3", 16)
    body = Body(
        (
            Loop(axis=a, body=(Load(name="p", input="A", index=(Var("a2"),)), Accum(name="acc0", value="p"))),
            Loop(axis=b, body=(Load(name="q", input="B", index=(Var("a3"),)), Accum(name="acc1", value="q"))),
        )
    )

    out = unify_sibling_reduce_axes(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert loops[0].axis.name == "a2"
    assert loops[1].axis.name == "a3"


def test_unify_transitively_groups_three_loops() -> None:
    """Union-find on the overlap relation: A overlaps B on (x,0), B
    overlaps C on (y,0), so all three unify to one canonical name."""
    a = Axis("a", 8)
    b = Axis("b", 8)
    c = Axis("c", 8)
    body = Body(
        (
            Loop(axis=a, body=(Load(name="p", input="x", index=(Var("a"),)), Accum(name="acc_a", value="p"))),
            Loop(
                axis=b,
                body=(
                    Load(name="q", input="x", index=(Var("b"),)),
                    Load(name="r", input="y", index=(Var("b"),)),
                    Assign(name="s", op="multiply", args=("q", "r")),
                    Accum(name="acc_b", value="s"),
                ),
            ),
            Loop(axis=c, body=(Load(name="t", input="y", index=(Var("c"),)), Accum(name="acc_c", value="t"))),
        )
    )

    out = unify_sibling_reduce_axes(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert {loop.axis.name for loop in loops} == {"a"}


# ---------------------------------------------------------------------------
# merge_sibling_reduce_loops — concatenation under safety gates
# ---------------------------------------------------------------------------


def test_merge_concatenates_matching_reduce_bodies() -> None:
    """The motivating case: after unify renames both axes to ``a2``,
    merge concatenates the two bodies into one Loop, eliminating the
    duplicate K traversal."""
    a = Axis("k", 16)
    body = Body(
        (
            _matmul_reduce(a, w_buffer="Wg", out_acc="acc0"),
            _matmul_reduce(a, w_buffer="Wu", out_acc="acc1"),
            Write(output="out", index=(Var("n"),), value="acc1"),
        )
    )

    out = merge_sibling_reduce_loops(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert len(loops) == 1, f"expected one merged Loop, got {len(loops)}"
    inner_accs = [s.name for s in loops[0].body if isinstance(s, Accum)]
    assert inner_accs == ["acc0", "acc1"]


def test_merge_skips_when_second_loop_reads_first_loops_accum() -> None:
    """Softmax pattern: first reduce produces ``acc_max``, second reduce
    body reads it inside the iteration. Merging would change the read
    from finalized to in-flight — must be skipped."""
    a = Axis("k", 16)
    body = Body(
        (
            Loop(axis=a, body=(Load(name="x1", input="X", index=(Var("k"),)), Accum(name="acc_max", value="x1", op="maximum"))),
            Loop(
                axis=a,
                body=(
                    Load(name="x2", input="X", index=(Var("k"),)),
                    Assign(name="d", op="subtract", args=("x2", "acc_max")),
                    Assign(name="e", op="exp", args=("d",)),
                    Accum(name="acc_sum", value="e"),
                ),
            ),
        )
    )

    out = merge_sibling_reduce_loops(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert len(loops) == 2, "softmax-style sequential reduces must stay distinct"


def test_merge_skips_on_ssa_name_collision() -> None:
    """If both bodies define the same SSA name, merging would create
    inner-scope shadowing — skip rather than guess a rename."""
    a = Axis("k", 16)
    body = Body(
        (
            Loop(axis=a, body=(Load(name="v", input="A", index=(Var("k"),)), Accum(name="acc0", value="v"))),
            Loop(axis=a, body=(Load(name="v", input="B", index=(Var("k"),)), Accum(name="acc1", value="v"))),
        )
    )

    out = merge_sibling_reduce_loops(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert len(loops) == 2, "name collision on 'v' must block the merge"


def test_merge_skips_when_between_stmt_def_used_by_second_loop() -> None:
    """A pure stmt between the two loops defines an SSA name the second
    loop's body reads — merging would move the read above the def."""
    a = Axis("k", 16)
    body = Body(
        (
            Loop(axis=a, body=(Load(name="p", input="A", index=(Var("k"),)), Accum(name="acc0", value="p"))),
            Assign(name="bias", op="exp", args=("acc0",)),
            Loop(
                axis=a,
                body=(
                    Load(name="q", input="B", index=(Var("k"),)),
                    Assign(name="m", op="multiply", args=("q", "bias")),
                    Accum(name="acc1", value="m"),
                ),
            ),
        )
    )

    out = merge_sibling_reduce_loops(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert len(loops) == 2, "between-stmt def consumed by second loop must block the merge"


def test_merge_keeps_intermediate_pure_stmts_in_place() -> None:
    """Pure stmts between the two reduce-loops that only depend on the
    first loop's Accum (not the second's body) remain valid post-merge
    — first Accum is still in scope above them, and they don't feed the
    second loop's body."""
    a = Axis("k", 16)
    body = Body(
        (
            _matmul_reduce(a, w_buffer="Wg", out_acc="acc0"),
            Assign(name="silu_val", op="silu", args=("acc0",)),  # depends only on acc0
            _matmul_reduce(a, w_buffer="Wu", out_acc="acc1"),
            Assign(name="result", op="multiply", args=("silu_val", "acc1")),
            Write(output="out", index=(Var("n"),), value="result"),
        )
    )

    out = merge_sibling_reduce_loops(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert len(loops) == 1, "intermediate stmt depending only on first Accum must not block the merge"
    inner_accs = [s.name for s in loops[0].body if isinstance(s, Accum)]
    assert inner_accs == ["acc0", "acc1"]


def test_merge_recurses_into_nested_loops() -> None:
    """Outer free-loops over (n, m) wrap sibling reduce-loops over k —
    the pass must descend through the outer Loops and merge the inner
    siblings."""
    k = Axis("k", 16)
    n = Axis("n", 4)
    body = Body(
        (
            Loop(
                axis=n,
                body=(
                    _matmul_reduce(k, w_buffer="Wg", out_acc="acc0"),
                    _matmul_reduce(k, w_buffer="Wu", out_acc="acc1"),
                    Write(output="o", index=(Var("n"),), value="acc1"),
                ),
            ),
        )
    )

    out = merge_sibling_reduce_loops(body)

    outer = out[0]
    assert isinstance(outer, Loop) and outer.axis.name == "n"
    inner_loops = [s for s in outer.body if isinstance(s, Loop)]
    assert len(inner_loops) == 1, "inner sibling reduces must merge inside the outer Loop body"


def test_unify_then_merge_collapses_gated_mlp_pattern() -> None:
    """End-to-end: the gated-MLP pattern ``silu(x@Wg) * (x@Wu)`` enters
    with distinct axis names ``a2`` / ``a3`` indexing the same x slot.
    After unify + merge, one K-loop carries both matmul reductions and
    the post-loop multiply still references both Accums correctly."""
    a = Axis("a2", 16)
    b = Axis("a3", 16)
    body = Body(
        (
            _matmul_reduce(a, w_buffer="Wg", out_acc="acc0"),
            Assign(name="silu_val", op="silu", args=("acc0",)),
            _matmul_reduce(b, w_buffer="Wu", out_acc="acc1"),
            Assign(name="result", op="multiply", args=("silu_val", "acc1")),
            Write(output="out", index=(Var("n"),), value="result"),
        )
    )

    unified = unify_sibling_reduce_axes(body)
    merged = merge_sibling_reduce_loops(unified)

    loops = [s for s in merged if isinstance(s, Loop)]
    assert len(loops) == 1
    accs_inside = [s.name for s in loops[0].body if isinstance(s, Accum)]
    assert accs_inside == ["acc0", "acc1"]
    x_loads = [s for s in loops[0].body if isinstance(s, Load) and s.input == "x"]
    assert len(x_loads) == 2, "dedup happens in a later pass — merge alone leaves both x loads"
