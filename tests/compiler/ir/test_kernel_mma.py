"""Tests for the MMA fragment Kernel-IR Stmts (M4 of
``plans/mma-fragment-factorization.md``)."""

from __future__ import annotations

from deplodock.compiler.dtype import F16, F32
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.kernel import MmaFill, MmaFragment, MmaLoad, MmaStore, MmaSync
from deplodock.compiler.ir.stmt import RenderCtx


def _ctx() -> RenderCtx:
    """Bare RenderCtx with one smem buffer registered so render_index
    has a shape to flatten against."""
    ctx = RenderCtx()
    ctx.shapes["c"] = (16, 16)
    ctx.shapes["smem_a"] = (16, 16)
    ctx.shapes["smem_b"] = (16, 16)
    return ctx


def test_mma_fragment_pretty():
    f = MmaFragment(name="a_frag", role="a", shape=(16, 16, 16), dtype=F16)
    assert f.pretty() == ["MmaFragment a:f16 a_frag (16x16x16, row_major)"]
    assert f.defines() == ("a_frag",)
    assert f.local_decls() == ("a_frag",)


def test_mma_fragment_render_operand_a():
    """Operand fragments carry a matrix tag + layout."""
    f = MmaFragment(name="a_frag", role="a", shape=(16, 16, 16), dtype=F16, layout="row_major")
    out = f.render(_ctx())
    assert "wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;" in out[0]


def test_mma_fragment_render_operand_b_col_major():
    f = MmaFragment(name="b_frag", role="b", shape=(16, 16, 16), dtype=F16, layout="col_major")
    out = f.render(_ctx())
    assert "wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;" in out[0]


def test_mma_fragment_render_accumulator():
    """The C fragment has no layout in the type signature."""
    f = MmaFragment(name="c_frag", role="c", shape=(16, 16, 16), dtype=F32)
    out = f.render(_ctx())
    assert "wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;" in out[0]


def test_mma_fragment_rejects_unknown_role():
    import pytest

    with pytest.raises(ValueError, match="unsupported role"):
        MmaFragment(name="x", role="d", shape=(16, 16, 16), dtype=F16).render(_ctx())


def test_mma_fill_render():
    out = MmaFill(frag="c_frag", value=0.0).render(_ctx())
    assert "wmma::fill_fragment(c_frag, 0.0f);" in out[0]


def test_mma_load_render():
    """``wmma::load_matrix_sync(frag, &<buffer>[<offset>], ldm);``"""
    load = MmaLoad(frag="a_frag", src_buffer="smem_a", src_index=(Literal(0, "int"), Literal(0, "int")), ldm=16)
    out = load.render(_ctx())
    assert "wmma::load_matrix_sync(a_frag, &smem_a[" in out[0]
    assert "], 16);" in out[0]
    assert load.deps() == ("a_frag",)
    assert load.external_reads() == ("smem_a",)


def test_mma_sync_render():
    sync = MmaSync(c_frag="c", a_frag="a", b_frag="b")
    out = sync.render(_ctx())
    assert "wmma::mma_sync(c, a, b, c);" in out[0]
    assert sync.deps() == ("c", "a", "b")


def test_mma_store_render_row_major():
    store = MmaStore(
        dst_buffer="c",
        dst_index=(Literal(0, "int"), Literal(0, "int")),
        frag="c_frag",
        ldm=16,
        layout="row_major",
    )
    out = store.render(_ctx())
    assert "wmma::store_matrix_sync(&c[" in out[0]
    assert "], c_frag, 16, wmma::mem_row_major);" in out[0]
    assert store.deps() == ("c_frag",)


def test_mma_chain_round_trip():
    """A C-fragment fill + per-K-step load+load+sync + final store —
    structurally the chain the MMA materializer (M5) emits per
    (M_r, N_r) cell."""
    stmts = [
        MmaFragment(name="c", role="c", shape=(16, 16, 16), dtype=F32),
        MmaFragment(name="a", role="a", shape=(16, 16, 16), dtype=F16),
        MmaFragment(name="b", role="b", shape=(16, 16, 16), dtype=F16, layout="col_major"),
        MmaFill(frag="c"),
        MmaLoad(frag="a", src_buffer="smem_a", src_index=(Var("k_i"),), ldm=16),
        MmaLoad(frag="b", src_buffer="smem_b", src_index=(Var("k_i"),), ldm=16),
        MmaSync(c_frag="c", a_frag="a", b_frag="b"),
        MmaStore(dst_buffer="c", dst_index=(Literal(0, "int"),), frag="c", ldm=16),
    ]
    # Render in order — names should appear in declared order, fills /
    # loads / sync use them, store emits the final write.
    rendered = []
    for s in stmts:
        rendered.extend(s.render(_ctx()))
    assert any("wmma::fragment<wmma::accumulator" in line for line in rendered)
    assert any("wmma::fragment<wmma::matrix_a" in line for line in rendered)
    assert any("wmma::fragment<wmma::matrix_b" in line for line in rendered)
    assert any("wmma::fill_fragment(c" in line for line in rendered)
    assert any("wmma::load_matrix_sync(a, " in line for line in rendered)
    assert any("wmma::load_matrix_sync(b, " in line for line in rendered)
    assert any("wmma::mma_sync(c, a, b, c)" in line for line in rendered)
    assert any("wmma::store_matrix_sync(&c[" in line for line in rendered)
