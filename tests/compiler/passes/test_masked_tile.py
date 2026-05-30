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


def test_planner_admits_non_divisor_n_with_real_extent(recording_dump, monkeypatch):
    """N=47 has no divisor in ``_TUNE_AXIS_CHOICES`` other than 1. The planner
    should still emit a TileOp whose N block axis carries ``real_extent=47``
    and whose body contains a ``Cond`` masking OOB lanes.

    The greedy prior may pick a tile that masks on M instead (M=256 with a
    non-divisor FM also gets admitted under the wider F-choice set), but the
    test is checking the N-masking path. Pin BN to force it.
    """
    # Pin (BN, FN) so the planner masks only N — pick BN*FN that doesn't
    # divide 47 (always, since 47 is prime — but make the masked-cell count
    # small). Pin (BM, FM) so BM*FM divides 256 cleanly, avoiding an extra
    # M-mask Cond that would land first in body iteration order and trip the
    # 'first Cond' lookup below.
    for k, v in (("BN", "8"), ("FN", "4"), ("BM", "32"), ("FM", "4"), ("BK", "32"), ("SPLITK", "1")):
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
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


def _n_write_index(tile_op):
    """Return the N-column index Expr of the matmul output Write (the last
    dim of the 2D output ``o[m, n]``), or None."""
    for w in tile_op.body.iter_of_type(Write):
        if w.output == "o" and len(w.index) == 2:
            return w.index[-1]
    return None


def _tile_axis_names(tile_op, tile_cls):
    """Axis names carried by every ``tile_cls`` (ThreadTile / RegisterTile)
    in the body. ``normalize_body`` renames the planner's ``*_t`` / ``*_r``
    axes to ``aN``, so identify them structurally by tile flavor rather than
    by name suffix."""
    return {ax.name for stmt in tile_op.body.iter() if isinstance(stmt, tile_cls) for ax in stmt.axes if isinstance(ax, Axis)}


def _n_decode_coeffs(tile_op):
    """Return ``(thread_coeff, register_coeff)`` of the N-column Write index —
    the stride applied to the N ThreadTile axis and N RegisterTile axis. The
    M tile axes don't appear in the N column (coeff 0), so passing all tile
    axes is safe. ``None`` for an axis that's been inlined (extent 1)."""
    from deplodock.compiler.ir.expr import affine_form

    n_idx = _n_write_index(tile_op)
    assert n_idx is not None, "no 2D matmul output Write found"
    t_names = _tile_axis_names(tile_op, ThreadTile)
    r_names = _tile_axis_names(tile_op, RegisterTile)
    af = affine_form(n_idx, t_names | r_names)
    assert af is not None, f"N index not affine in tile axes: {n_idx.pretty()}"
    _, coeffs = af
    t_coeff = next((coeffs[n] for n in t_names if n in coeffs), None)
    r_coeff = next((coeffs[n] for n in r_names if n in coeffs), None)
    return t_coeff, r_coeff


def test_masked_n_uses_interleaved_thread_minor_decode(recording_dump, monkeypatch):
    """A masked (non-divisor) N tile must decode thread-minor: the N register
    cell strides by BN so consecutive threads map to consecutive columns
    (coalesced global weight loads). Regression guard for the lm_head
    1024→vocab projection (94 ms → 3.5 ms). Shape picked so BN≠FN — otherwise
    blocked and interleaved share the same coefficient and can't be told
    apart.

    Pinned via ``DEPLODOCK_KNOBS`` to BN=256, FN=8 so the test asserts the
    register-decode rewrite property independent of which masked variant the
    planner's prior happens to rank first — the smem-fit + TMA-eligibility
    signals can reasonably prefer narrower BN at this shape, but the
    register-decode logic must still emit a thread-minor stride at BN=256.
    """
    # ``DEPLODOCK_KNOBS`` would be too late (``apply_knobs_env`` runs at module
    # import); set the individual per-knob env vars that ``Knob.narrow`` reads.
    for k, v in (("BN", "256"), ("FN", "8"), ("BM", "1"), ("FM", "4"), ("BK", "16"), ("SPLITK", "1")):
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    g = Graph()
    _input(g, "a", (32, 256))
    _input(g, "b", (256, 8191))  # N=8191: no divisor → masked
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (32, 8191)), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]

    out = Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    t_coeff, r_coeff = _n_decode_coeffs(tile_op)
    # Interleaved: thread axis is the minor one (coeff 1), register strides by BN.
    assert t_coeff == 1, f"masked N should be thread-minor (thread coeff 1), got t={t_coeff} r={r_coeff}"
    assert r_coeff is not None and r_coeff > 1, f"register cell should stride by BN>1, got t={t_coeff} r={r_coeff}"


def test_clean_divisor_n_uses_blocked_thread_major_decode(recording_dump):
    """A clean-divisor N tile keeps the blocked (thread-major) decode so a
    thread's FN cells stay contiguous (vectorizable / smem-conflict-free):
    the N register cell is the minor axis (coeff 1)."""
    g = Graph()
    _input(g, "a", (32, 256))
    _input(g, "b", (256, 8192))  # N=8192: clean divisor → not masked
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (32, 8192)), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]

    out = Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    t_coeff, r_coeff = _n_decode_coeffs(tile_op)
    # Blocked: the register cell is the minor axis (coeff 1); the thread axis
    # carries the FN stride. (If FN==1 the register axis inlines away → None.)
    if r_coeff is not None:
        assert r_coeff == 1, f"clean N should be thread-major (register coeff 1), got t={t_coeff} r={r_coeff}"
        assert t_coeff is not None and t_coeff > 1, f"thread axis should stride by FN>1, got t={t_coeff} r={r_coeff}"
