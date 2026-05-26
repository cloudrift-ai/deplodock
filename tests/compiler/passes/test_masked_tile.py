"""Tests for masked tiles — output-axis extents with no power-of-2 divisor.

When the partition planner faces an output axis like ``vocab=151669`` whose
``_TUNE_AXIS_CHOICES`` divisor set is ``{1}``, it still picks a normal
``BN`` and emits ``Axis.real_extent`` on the block axis plus a ``Cond``
predicate wrapping the σ-rewritten body. The Cond predicate references the
register-tile axis, so ``010_split_register_axes`` must replicate the Cond
itself (not just descend into its body) to avoid leaving dangling Var refs.
"""

from __future__ import annotations

from deplodock.compiler.dim import DEFAULT_SEQ_HINT, Dim
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.stmt import Cond, Load, Write
from deplodock.compiler.ir.tile.ir import RegisterTile, ThreadTile, TileOp
from deplodock.compiler.pipeline import KERNEL_PASSES, TILE_PASSES, Pipeline


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


def test_planner_admits_non_divisor_n_with_real_extent(recording_dump):
    """N=47 has no divisor in ``_TUNE_AXIS_CHOICES`` other than 1. The planner
    should still emit a TileOp whose N block axis carries ``real_extent=47``
    and whose body contains a ``Cond`` masking OOB lanes."""
    g = Graph()
    _input(g, "a", (256, 64))
    _input(g, "b", (64, 47))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (256, 47)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]

    out = Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))

    # Find any axis with real_extent set — should be the N block axis at 47.
    real_extents = []
    for stmt in tile_op.body.iter():
        for ax in getattr(stmt, "axes", ()):
            if isinstance(ax, Axis) and ax.real_extent is not None:
                real_extents.append((ax.name, ax.real_extent))
    assert 47 in [e for _, e in real_extents], f"expected real_extent=47 on a block axis, got {real_extents}"

    # The body should contain at least one Cond (the mask) referencing 47.
    conds = list(tile_op.body.iter_of_type(Cond))
    assert conds, "expected at least one mask Cond wrapping the σ-rewritten body"
    pred_text = conds[0].cond.pretty()
    assert "47" in pred_text, f"mask predicate should reference real extent 47, got {pred_text!r}"


def test_split_register_axes_replicates_cond_with_axis_dep_predicate():
    """When a Cond's predicate references the register-tile axis being
    replicated, the pass must replicate the entire Cond (not just descend
    into its body). Each replica's predicate gets the σ-substituted literal
    so NVRTC sees fully-resolved conditions — and there is no dangling
    reference to a no-longer-defined register-axis Var."""
    a_thread = Axis("a_thread", 4)
    a_reg = Axis("a_reg", 3)
    inner_body = (
        Cond(
            cond=BinaryExpr("<", BinaryExpr("+", BinaryExpr("*", Var("a_thread"), Literal(3, "int")), Var("a_reg")), Literal(10, "int")),
            body=(
                Load(name="x_v", input="x", index=(Var("a_thread"), Var("a_reg"))),
                Write(output="o", index=(Var("a_thread"), Var("a_reg")), value="x_v"),
            ),
        ),
    )
    tile = ThreadTile(
        axes=(a_thread,),
        body=(RegisterTile(axes=(a_reg,), body=inner_body),),
    )
    tile_op = TileOp(body=(tile,), name="t")

    g = Graph()
    _input(g, "x", (4, 3))
    g.add_node(op=tile_op, inputs=["x"], output=Tensor("o", (4, 3)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    out = Pipeline.build(KERNEL_PASSES, select={"split_register_axes"}).run(g)
    new_tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    new_tile = next(s for s in new_tile_op.body if isinstance(s, ThreadTile))

    # RegisterTile is gone, body is replicated 3 times — so 3 Conds (not 1).
    assert not any(isinstance(s, RegisterTile) for s in new_tile.body)
    conds = [s for s in new_tile.body if isinstance(s, Cond)]
    assert len(conds) == 3, f"expected 3 replicated Conds (one per a_reg literal), got {len(conds)}"

    # Each replica's predicate should have a_reg substituted to its literal —
    # no surviving ``a_reg`` Var references in any predicate.
    for c in conds:
        free = set(c.cond.free_vars())
        assert "a_reg" not in free, f"a_reg should be σ-substituted, got predicate {c.cond.pretty()!r}"


# ---------------------------------------------------------------------------
# Symbolic (hint-driven) masked tiles — a dynamic axis is tiled for its Dim
# hint and emitted as a masked tile (ceil-div grid + boundary Cond against the
# runtime symbolic value), so one kernel runs correctly at any seq_len.
# ---------------------------------------------------------------------------


def test_dim_defaults_symbolic_hint():
    """Atomic symbolic Dims carry the default expected size; static and
    composite dims don't (the planner only reads the hint off the input axis)."""
    assert Dim("seq_len").hint == DEFAULT_SEQ_HINT
    assert Dim(32).hint is None
    assert (Dim("seq_len") * 2).hint is None  # composite
    assert Dim("seq_len", hint=128).hint == 128  # explicit wins
    # Hint is metadata: excluded from identity so cache keys stay hint-independent.
    assert Dim("seq_len", hint=128) == Dim("seq_len")
    assert hash(Dim("seq_len", hint=128)) == hash(Dim("seq_len"))


def test_planner_masks_symbolic_m_axis_at_hint(recording_dump):
    """A matmul with a symbolic M (seq_len) is tiled for the hint and masked:
    the M block axis becomes a composite ceil-div over ``seq_len`` and the body
    carries a boundary Cond referencing the symbolic ``seq_len`` (not a literal)."""
    g = Graph()
    _input(g, "a", (Dim("seq_len"), 64))
    _input(g, "b", (64, 512))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (Dim("seq_len"), 512)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]

    out = Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))

    # A block axis should carry a composite ceil-div extent over seq_len (not a
    # bare Var, not static) — i.e. a real tile, not the degenerate whole-axis bind.
    def _is_ceildiv_over_seq(ax: Axis) -> bool:
        e = ax.extent
        return not e.is_static and not isinstance(e.expr, Var) and "seq_len" in e.expr.free_vars()

    composite_block = [
        ax for stmt in tile_op.body.iter() for ax in getattr(stmt, "axes", ()) if isinstance(ax, Axis) and _is_ceildiv_over_seq(ax)
    ]
    assert composite_block, "expected a ceil-div block axis over seq_len (hint-driven masked tile)"

    # The boundary Cond should compare against the symbolic seq_len, not a literal.
    conds = list(tile_op.body.iter_of_type(Cond))
    assert conds, "expected a mask Cond"
    assert any("seq_len" in c.cond.free_vars() for c in conds), f"mask should gate against seq_len, got {[c.cond.pretty() for c in conds]}"


def test_resolve_dim_evaluates_ceildiv_grid_factor():
    """A grid spec carrying a composite ceil-div Expr resolves at launch from
    ``sym_values`` (one cached kernel, any runtime seq_len)."""
    from deplodock.compiler.ir.cuda.ir import resolve_dim

    ceil_div = ((Dim("seq_len") + 31) // 32).expr  # (seq_len + 31) // 32
    spec = (4, ceil_div, 2)  # heads * M_blocks * splitk
    assert resolve_dim(spec, {"seq_len": 100}) == 4 * ((100 + 31) // 32) * 2  # 4*4*2 = 32
    assert resolve_dim(spec, {"seq_len": 512}) == 4 * 16 * 2


def test_resolve_symbolic_falls_back_to_hint_without_inputs():
    """Benchmarking a symbolic graph with no input arrays (the autotuner case)
    resolves each symbolic dim to its ``Dim`` hint instead of raising."""
    from deplodock.compiler.backend.cuda.program import _Compiled, _resolve_symbolic, _symbolic_hints

    g = Graph()
    _input(g, "x", (1, Dim("seq_len"), 2048))
    g.inputs = ["x"]
    g.outputs = ["x"]
    hints = _symbolic_hints(g)
    assert hints == {"seq_len": DEFAULT_SEQ_HINT}

    compiled = _Compiled(
        bufs=[],
        buf_by_name={},
        constants={},
        kernels={},
        launches=[],
        symbolic_bindings={"seq_len": ("x", 1)},
        symbolic_hints=hints,
    )
    # No input_data → fall back to the hint.
    assert _resolve_symbolic(compiled, {}) == {"seq_len": DEFAULT_SEQ_HINT}
