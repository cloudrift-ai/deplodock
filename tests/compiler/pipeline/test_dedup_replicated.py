"""Tests for ``011_dedup_replicated`` — the Tile-IR CSE pass that folds
duplicate ``Load`` / ``Assign`` stmts after the register-tile replicator
unwraps ``RegisterTile`` bodies F× per cell.

Each test feeds the rule a synthetic Tile-IR shape that mocks what the
replicator emits for a small unroll (FN=4 typical) and asserts the
deduped output: one Load / Assign instead of N, downstream uses
rewritten to the surviving SSA name.

``TileOp.__post_init__`` runs the full ``normalize_body`` pipeline
(including ``rename_ssa_sequential``) on construction, so assertions
key off counts and "all consumers share one value name" invariants,
never literal SSA names — those are canonicalized to ``in0`` / ``in1``
/ … on the way in and out.
"""

from __future__ import annotations

import importlib

import pytest

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Assign, Body, Cond, Load, Write
from deplodock.compiler.ir.tile.ir import RegisterTile, ThreadTile, TileOp
from deplodock.compiler.pipeline import RuleSkipped
from deplodock.compiler.tensor import Tensor

_dedup = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.kernel.011_dedup_replicated")


def _run_rule(op: TileOp) -> TileOp:
    g = Graph()
    g.add_node(op=op, inputs=[], output=Tensor(op.name, ()), node_id="op")
    return _dedup.rewrite(g.nodes["op"])


def _loads(body: Body) -> list[Load]:
    return [s for s in body.iter() if isinstance(s, Load)]


def _assigns(body: Body) -> list[Assign]:
    return [s for s in body.iter() if isinstance(s, Assign)]


def _writes(body: Body) -> list[Write]:
    return [s for s in body.iter() if isinstance(s, Write)]


# ---------------------------------------------------------------------------
# Sibling Load-CSE
# ---------------------------------------------------------------------------


def test_two_identical_loads_fold_to_one():
    """Two ``Load`` stmts at the same scope with identical
    ``(input, index)`` — the post-replicator residue for an N-invariant
    read on an FN=2 cell — collapse to one. Both consumer Writes
    reference the same surviving Load's name."""
    body = Body(
        (
            ThreadTile(
                axes=(Axis("m", 8),),
                body=Body(
                    (
                        Load(name="a_0", input="x", index=(Var("m"),)),
                        Load(name="a_1", input="x", index=(Var("m"),)),
                        Write(output="o", index=(Var("m"), Literal(0, "int")), value="a_0"),
                        Write(output="o", index=(Var("m"), Literal(1, "int")), value="a_1"),
                    )
                ),
            ),
        )
    )
    op = TileOp(body=body, name="k_test")
    result = _run_rule(op)
    loads = _loads(result.body)
    assert len(loads) == 1
    # Both Writes now name the surviving Load.
    writes = _writes(result.body)
    surviving = loads[0].name
    assert [w.values for w in writes] == [(surviving,), (surviving,)]


def test_loads_with_different_indices_dont_fold():
    """A pair of ``Load``s reading different elements of the same input
    have distinct keys (``index`` pretty-prints differ); both survive."""
    body = Body(
        (
            ThreadTile(
                axes=(Axis("m", 4),),
                body=Body(
                    (
                        Load(name="a_0", input="x", index=(Var("m"), Literal(0, "int"))),
                        Load(name="a_1", input="x", index=(Var("m"), Literal(1, "int"))),
                        Write(output="o", index=(Var("m"), Literal(0, "int")), value="a_0"),
                        Write(output="o", index=(Var("m"), Literal(1, "int")), value="a_1"),
                    )
                ),
            ),
        )
    )
    op = TileOp(body=body, name="k_test")
    with pytest.raises(RuleSkipped):
        _run_rule(op)


def test_loads_from_different_inputs_dont_fold():
    """``input`` is part of the key — same index, different buffer names
    are distinct values."""
    body = Body(
        (
            ThreadTile(
                axes=(Axis("m", 4),),
                body=Body(
                    (
                        Load(name="a", input="x", index=(Var("m"),)),
                        Load(name="b", input="y", index=(Var("m"),)),
                        Write(output="o", index=(Var("m"), Literal(0, "int")), value="a"),
                        Write(output="o", index=(Var("m"), Literal(1, "int")), value="b"),
                    )
                ),
            ),
        )
    )
    op = TileOp(body=body, name="k_test")
    with pytest.raises(RuleSkipped):
        _run_rule(op)


# ---------------------------------------------------------------------------
# Assign-CSE
# ---------------------------------------------------------------------------


def test_two_identical_assigns_fold_to_one():
    """Same ``(op, args)`` → kept once. The dropped Assign's name is
    rewired in the consumer Write."""
    body = Body(
        (
            ThreadTile(
                axes=(Axis("m", 4),),
                body=Body(
                    (
                        Load(name="a", input="x", index=(Var("m"),)),
                        Load(name="b", input="y", index=(Var("m"),)),
                        Assign(name="c_0", op="multiply", args=("a", "b")),
                        Assign(name="c_1", op="multiply", args=("a", "b")),
                        Write(output="o", index=(Var("m"), Literal(0, "int")), value="c_0"),
                        Write(output="o", index=(Var("m"), Literal(1, "int")), value="c_1"),
                    )
                ),
            ),
        )
    )
    op = TileOp(body=body, name="k_test")
    result = _run_rule(op)
    assigns = _assigns(result.body)
    assert len(assigns) == 1
    surviving = assigns[0].name
    writes = _writes(result.body)
    assert [w.values for w in writes] == [(surviving,), (surviving,)]


def test_assigns_with_different_ops_dont_fold():
    """Same args, different op — distinct values, both kept."""
    body = Body(
        (
            ThreadTile(
                axes=(Axis("m", 4),),
                body=Body(
                    (
                        Load(name="a", input="x", index=(Var("m"),)),
                        Load(name="b", input="y", index=(Var("m"),)),
                        Assign(name="c_0", op="multiply", args=("a", "b")),
                        Assign(name="c_1", op="add", args=("a", "b")),
                        Write(output="o", index=(Var("m"), Literal(0, "int")), value="c_0"),
                        Write(output="o", index=(Var("m"), Literal(1, "int")), value="c_1"),
                    )
                ),
            ),
        )
    )
    op = TileOp(body=body, name="k_test")
    with pytest.raises(RuleSkipped):
        _run_rule(op)


# ---------------------------------------------------------------------------
# Combined Load + Assign fold (the actual replicator residue)
# ---------------------------------------------------------------------------


def test_replicated_fn4_cell_collapses_through_load_then_assign():
    """Mock the per-cell residue an FN=4 register-tile replicator emits
    when the inner body reads one N-invariant value and squares it: four
    Loads of x[m] feeding four ``mul(a_i, a_i)`` Assigns feeding four
    Writes to o[m, i]. After CSE there is one Load, one Assign, and
    four Writes (the per-cell Write index *does* reference the cell
    constant, so the Writes are all distinct and must survive)."""
    body = Body(
        (
            ThreadTile(
                axes=(Axis("m", 8),),
                body=Body(
                    (
                        Load(name="a_0", input="x", index=(Var("m"),)),
                        Assign(name="s_0", op="multiply", args=("a_0", "a_0")),
                        Write(output="o", index=(Var("m"), Literal(0, "int")), value="s_0"),
                        Load(name="a_1", input="x", index=(Var("m"),)),
                        Assign(name="s_1", op="multiply", args=("a_1", "a_1")),
                        Write(output="o", index=(Var("m"), Literal(1, "int")), value="s_1"),
                        Load(name="a_2", input="x", index=(Var("m"),)),
                        Assign(name="s_2", op="multiply", args=("a_2", "a_2")),
                        Write(output="o", index=(Var("m"), Literal(2, "int")), value="s_2"),
                        Load(name="a_3", input="x", index=(Var("m"),)),
                        Assign(name="s_3", op="multiply", args=("a_3", "a_3")),
                        Write(output="o", index=(Var("m"), Literal(3, "int")), value="s_3"),
                    )
                ),
            ),
        )
    )
    op = TileOp(body=body, name="k_test")
    result = _run_rule(op)
    assert len(_loads(result.body)) == 1
    assert len(_assigns(result.body)) == 1
    writes = _writes(result.body)
    assert len(writes) == 4
    # All Writes name the same surviving Assign result.
    surviving = _assigns(result.body)[0].name
    assert {w.values for w in writes} == {(surviving,)}


# ---------------------------------------------------------------------------
# Scope walks
# ---------------------------------------------------------------------------


def test_dedup_descends_into_register_tile_body():
    """The replicator may leave a ``RegisterTile`` wrapping its
    already-replicated body (the wrapper is consumed by the time the
    pass runs in practice, but the dedup recursion must not rely on
    that). Two identical Loads inside the ``RegisterTile.body`` fold."""
    body = Body(
        (
            ThreadTile(
                axes=(Axis("m", 4),),
                body=Body(
                    (
                        RegisterTile(
                            axes=(Axis("n_r", 1),),
                            body=Body(
                                (
                                    Load(name="a_0", input="x", index=(Var("m"),)),
                                    Load(name="a_1", input="x", index=(Var("m"),)),
                                    Write(output="o", index=(Var("m"), Literal(0, "int")), value="a_0"),
                                    Write(output="o", index=(Var("m"), Literal(1, "int")), value="a_1"),
                                )
                            ),
                        ),
                    )
                ),
            ),
        )
    )
    op = TileOp(body=body, name="k_test")
    result = _run_rule(op)
    assert len(_loads(result.body)) == 1


def test_outer_load_reused_inside_nested_cond():
    """A Load at outer scope flows through into a nested ``Cond`` body —
    an inner Load with the same ``(input, index)`` key folds to the
    outer name. The inner scope's local table inherits the outer
    ``env``; matches the existing ``loop/fusion/020_dedup_loads``
    cross-scope semantics."""
    body = Body(
        (
            ThreadTile(
                axes=(Axis("m", 4),),
                body=Body(
                    (
                        Load(name="a_out", input="x", index=(Var("m"),)),
                        Cond(
                            cond=Var("m").lt(Literal(2, "int")),
                            body=Body(
                                (
                                    Load(name="a_in", input="x", index=(Var("m"),)),
                                    Write(output="o", index=(Var("m"),), value="a_in"),
                                )
                            ),
                        ),
                        Write(output="o2", index=(Var("m"),), value="a_out"),
                    )
                ),
            ),
        )
    )
    op = TileOp(body=body, name="k_test")
    result = _run_rule(op)
    # Outer Load kept; inner Load dropped (same key as outer).
    loads = _loads(result.body)
    assert len(loads) == 1
    surviving = loads[0].name
    # Both Writes now name the outer Load.
    writes = _writes(result.body)
    assert len(writes) == 2
    assert all(w.values == (surviving,) for w in writes)


def test_sibling_scopes_dont_share_assigns():
    """Two Cond branches are sibling scopes — an Assign produced in one
    branch's body is not visible to the other. The dedup table is
    per-scope for Assigns; both ``then`` and ``else`` keep their own."""
    body = Body(
        (
            ThreadTile(
                axes=(Axis("m", 4),),
                body=Body(
                    (
                        Load(name="a", input="x", index=(Var("m"),)),
                        Load(name="b", input="y", index=(Var("m"),)),
                        Cond(
                            cond=Var("m").lt(Literal(2, "int")),
                            body=Body(
                                (
                                    Assign(name="t_then", op="multiply", args=("a", "b")),
                                    Write(output="o", index=(Var("m"),), value="t_then"),
                                )
                            ),
                            else_body=Body(
                                (
                                    Assign(name="t_else", op="multiply", args=("a", "b")),
                                    Write(output="o2", index=(Var("m"),), value="t_else"),
                                )
                            ),
                        ),
                    )
                ),
            ),
        )
    )
    op = TileOp(body=body, name="k_test")
    with pytest.raises(RuleSkipped):
        _run_rule(op)


# ---------------------------------------------------------------------------
# Idempotence + RuleSkipped
# ---------------------------------------------------------------------------


def test_rule_skipped_when_already_deduped():
    """A body with no duplicate Loads / Assigns raises ``RuleSkipped`` —
    the engine relies on this to converge a pass scan."""
    body = Body(
        (
            ThreadTile(
                axes=(Axis("m", 4),),
                body=Body(
                    (
                        Load(name="a", input="x", index=(Var("m"),)),
                        Write(output="o", index=(Var("m"),), value="a"),
                    )
                ),
            ),
        )
    )
    op = TileOp(body=body, name="k_test")
    with pytest.raises(RuleSkipped, match="no duplicate"):
        _run_rule(op)
