"""Masked-tile boundary guards on the MMA store path (M9).

The boundary ``Cond`` a masked warp tile carries only gates the atom tile's
BASE coordinate — the fragment lane offsets (``_g`` / ``_t``) are render-local
— so a tile straddling the bound passes the Cond while its trailing rows /
cols are out of range. ``kernel/005_lower_atom_tile`` classifies the Cond's
predicate against the cell Write's M / N coordinates and stamps per-element
guards onto the ``RegStore``; the render predicates each element's store (and
its epilogue gmem reads) at its own coordinate. These tests pin the guard
classification, the guarded/unguarded renders, and the unstaged-gated-operand
decline — pure unit shapes, no planner involvement (the planner only emits
masked AtomTiles once the warp tier admits symbolic axes).
"""

from __future__ import annotations

import importlib

import pytest

from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel.ir import RegStore
from deplodock.compiler.ir.stmt import Body, Cond, Load, Write
from deplodock.compiler.ir.stmt.base import RenderCtx
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY
from deplodock.compiler.pipeline import RuleSkipped

_mod = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.kernel.005_lower_atom_tile")


def _store(**kw) -> RegStore:
    return RegStore(dst_buffer="o", dst_index=(Var("m0"), Var("n0")), frag="c_frag", shape=(16, 8, 16), ldm=64, **kw)


def test_unguarded_regstore_renders_vectorized_unconditionally():
    """No guards → the existing fast path: one vec2 store per row pair, no
    boundary ``if``s anywhere."""
    src = "\n".join(_store().render(RenderCtx(buffer_dtypes={"o": "f16"})))
    assert src.count("__floats2half2_rn") == 2
    assert "if (" not in src


def test_m_guard_keeps_vectorized_pairs_under_row_checks():
    """An ``m_guard`` alone predicates each row pair at its own row (``_g`` /
    ``_g + 8``) against the symbolic bound, keeping the vec2 stores."""
    src = "\n".join(_store(m_guard=(Var("m0"), Var("seq_len"))).render(RenderCtx(buffer_dtypes={"o": "f16"})))
    assert "if ((m0) + _g < (seq_len))" in src
    assert "if ((m0) + _g + 8 < (seq_len))" in src
    assert src.count("__floats2half2_rn") == 2, f"row-guarded store should keep the vectorized pairs:\n{src}"


def test_n_guard_falls_back_to_per_element_scalar_stores():
    """An ``n_guard`` splits the column pair (its two columns straddle the
    bound independently) into per-element scalar stores, each under the
    conjunction of the live guards."""
    src = "\n".join(
        _store(m_guard=(Var("m0"), Var("seq_len")), n_guard=(Var("n0"), Literal(47, "int"))).render(RenderCtx(buffer_dtypes={"o": "f16"}))
    )
    assert "__floats2half2_rn" not in src
    assert "(n0) + _t * 2 < (47)" in src
    assert "(n0) + _t * 2 + 1 < (47)" in src
    assert "(m0) + _g < (seq_len) && " in src
    assert src.count("if (") == 4


def test_boundary_guards_classify_m_and_n_predicates():
    """Predicate LHS struct-equal to the Write's second-to-last index dim →
    ``m_guard``; equal to the last → ``n_guard``. The Conds stay in the body
    (whole-tile skip); only the classification is extracted here."""
    m_expr = Var("mb") * Literal(32, "int") + Var("mt")
    n_expr = Var("nb") * Literal(8, "int")
    w = Write(output="o", index=(m_expr, n_expr), value="acc")
    body = Body(
        (
            Cond(
                cond=BinaryExpr("<", m_expr, Var("seq_len")),
                body=Body((Cond(cond=BinaryExpr("<", n_expr, Literal(47, "int")), body=Body((w,))),)),
            ),
        )
    )
    m_guard, n_guard = _mod._boundary_guards(body, w)
    assert m_guard == (m_expr, Var("seq_len"))
    assert n_guard == (n_expr, Literal(47, "int"))


def test_boundary_guards_none_without_conds():
    w = Write(output="o", index=(Var("m"), Var("n")), value="acc")
    assert _mod._boundary_guards(Body((w,)), w) == (None, None)


def test_boundary_guards_skip_literal_padding_dims():
    """Real-trace outputs pad with literal dims (``o[0, m, 0, n]`` — the
    q_proj reshape); classification must use the var-bearing dims, not raw
    positions."""
    zero = Literal(0, "int")
    m_expr = Var("mb") * Literal(16, "int")
    n_expr = Var("nb") * Literal(8, "int")
    w = Write(output="o", index=(zero, m_expr, zero, n_expr), value="acc")
    body = Body((Cond(cond=BinaryExpr("<", m_expr, Var("seq_len")), body=Body((w,))),))
    m_guard, n_guard = _mod._boundary_guards(body, w)
    assert m_guard == (m_expr, Var("seq_len"))
    assert n_guard is None


def test_boundary_guards_fail_loud_on_unmatched_predicate():
    """A boundary predicate that matches neither Write coordinate means the
    planner and this pass disagree about the masked axis — RuleSkipped (the
    variant pins a bench_fail row) rather than an unguarded straddling store."""
    w = Write(output="o", index=(Var("m"), Var("n")), value="acc")
    body = Body((Cond(cond=BinaryExpr("<", Var("something_else"), Var("seq_len")), body=Body((w,))),))
    with pytest.raises(RuleSkipped, match="matches no Write coordinate"):
        _mod._boundary_guards(body, w)


def test_emit_chain_clamps_unstaged_gated_operand():
    """A masked cell whose gated-axis operand was not staged takes the
    clamped gmem-direct fragment load (a plain gmem-direct read at
    straddling-tile coords would run past the runtime-sized buffer); the
    clean-axis operand keeps the unclamped helper. Every enumerated variant
    must lower — staging declines (smem budget) are legitimate and a greedy
    pick must not crash on the fallback."""
    from deplodock.compiler.ir.kernel.ir import LdmatrixLoad
    from deplodock.compiler.ir.stmt.base import RenderCtx

    spec = ATOM_REGISTRY["mma_m16n8k16_f16"]
    a_load = Load(name="a0", input="a", index=(Var("m"), Var("k")))
    b_load = Load(name="b0", input="b", index=(Var("k"), Var("n")))
    chain = _mod._emit_chain(
        spec,
        a_load=a_load,
        b_load=b_load,
        a_frag="af",
        b_frag="bf",
        c_frag="cf",
        smem_sources={},
        m_guard=(Var("m"), Var("seq_len")),
    )
    a_ld, b_ld = (s for s in chain if isinstance(s, LdmatrixLoad))
    assert a_ld.gmem_guard == (Var("m"), Var("seq_len"))
    assert b_ld.gmem_guard is None
    ctx = RenderCtx(shapes={"a": (512, 64), "b": (64, 512)})
    src = "\n".join(a_ld.render(ctx))
    assert "dpl_mma_load_a_gmem_mclamp(" in src
    assert "(seq_len) - (m)" in src, f"clamp arg should be the in-range rows left from the tile base:\n{src}"
    assert "dpl_mma_load_b_gmem_nclamp" not in "\n".join(b_ld.render(ctx))


def test_masked_shape_c_cell_lowers_through_the_cond():
    """A masked K-filtered (shape C) cell wraps ``[Load a, Load b, Mma,
    Write]`` in the boundary Cond; the role loads + Mma sit one level down.
    The cell must lower (chain + guarded store, Cond dropped — gmem-direct
    loads clamp, the store carries the per-element guards) instead of
    raising and leaving the AtomTile to crash render — which killed a whole
    tune at the first such variant."""
    from deplodock.compiler.ir.kernel.ir import LdmatrixLoad
    from deplodock.compiler.ir.stmt import Mma

    m_expr = Var("mb") * Literal(16, "int")
    n_expr = Var("nb") * Literal(8, "int")
    atom = ATOM_REGISTRY["mma_m16n8k16_f16"]
    cell = Body(
        (
            Cond(
                cond=BinaryExpr("<", m_expr, Var("seq_len")),
                body=Body(
                    (
                        Load(name="a0", input="a", index=(m_expr, Literal(0, "int"))),
                        Load(name="b0", input="b", index=(Literal(0, "int"), n_expr)),
                        Mma(a="a0", b="b0", c="acc", atom=atom),
                        Write(output="o", index=(m_expr, n_expr), value="acc"),
                    )
                ),
            ),
        )
    )
    out = _mod._lower_cell(cell, smem_sources={})
    kinds = [type(s).__name__ for s in out]
    assert "MmaSyncPtx" in kinds, f"shape-C masked cell must lower to the mma chain, got {kinds}"
    store = next(s for s in out if isinstance(s, RegStore))
    assert store.m_guard == (m_expr, Var("seq_len"))
    a_ld = next(s for s in out if isinstance(s, LdmatrixLoad) and s.role == "a")
    assert a_ld.gmem_guard == (m_expr, Var("seq_len")), "unstaged gated A operand must clamp"


def test_unstaged_masked_k_gate_detection():
    """Masked-K correctness gate (``_unstaged_masked_k``): an mma operand whose
    REDUCE (K) dim is symbolic must NOT lower gmem-direct — the gmem fragment
    load has no K zero-fill (only the staged ``_stage_expand`` path does), so it
    over-reads the mask-padded slab (the SDPA P@V hang / OOB bench_fails). A's K
    is the last index dim; B's K is a non-last dim (N is last). M-symbolic-only
    (A's non-last) and fully-static operands stay fine gmem-direct (M is clamped
    by ``gmem_guard``)."""
    from deplodock.compiler import dtype as _dt
    from deplodock.compiler.dim import Dim
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp

    f16 = _dt.get("f16")
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a_symK", (16, Dim("seq_len")), f16), node_id="a_symK")  # A: K (last) symbolic
    g.add_node(InputOp(), [], Tensor("a_symM", (Dim("seq_len"), 128), f16), node_id="a_symM")  # A: M (first) symbolic, K static
    g.add_node(InputOp(), [], Tensor("dense", (64, 128), f16), node_id="dense")  # fully static
    g.add_node(InputOp(), [], Tensor("b_symK", (Dim("seq_len"), 64), f16), node_id="b_symK")  # B: K (non-last) symbolic

    def ld(buf: str) -> Load:
        return Load(names=("x",), input=buf, index=(), dtype=f16)

    f = _mod._unstaged_masked_k
    assert f(ld("a_symK"), "a", g) is True  # A reduce (last) symbolic → gate
    assert f(ld("a_symM"), "a", g) is False  # A's K (last) static; symbolic M is gmem-clamped → fine
    assert f(ld("dense"), "a", g) is False  # fully static → fine
    assert f(ld("b_symK"), "b", g) is True  # B reduce (non-last) symbolic → gate
    assert f(ld("dense"), "b", g) is False
    assert f(ld("dense"), "a", None) is False  # no graph → no info, don't gate
    assert f(ld("missing"), "a", g) is False  # buffer absent from graph → don't gate
