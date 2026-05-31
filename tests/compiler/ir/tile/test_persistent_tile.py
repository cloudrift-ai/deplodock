"""Render + structural tests for the PersistentTile primitive (adaptive Stream-K).

Hand-builds a minimal matmul-shape ``PersistentTile`` and exercises its generic
contract (deps, frozen identity, repr/eval round-trip, with_bodies). The adaptive
MAC-segment render is pinned by the ``_adaptive`` tests further down; the
runtime-bounded K-loop (StridedTile / SerialTile) gets its own tests too.

``PersistentTile`` is adaptive-only — it always splits K via the MAC-segment walk
(``kernel/_streamk``). There is no tile-parallel variant.
"""

from __future__ import annotations

from deplodock.compiler.graph import _eval_stmt
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt.base import RenderCtx
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.tile.ir import Accum, PersistentTile, StridedTile, ThreadTile, Write


def _persistent(m_b: int = 8, n_b: int = 4, k_blocks: int = 4) -> PersistentTile:
    """``PersistentTile(M_b, N_b, k_blocks) > ThreadTile > Write`` — a minimal
    adaptive Stream-K tile for the generic structural tests."""
    inner = ThreadTile(axes=(Axis("m_t", 16), Axis("n_t", 16)), body=Body((Write(output="C", index=(Var("m_b"),), value="acc"),)))
    return PersistentTile(axes=(Axis("m_b", m_b), Axis("n_b", n_b)), body=Body((inner,)), k_blocks=k_blocks)


def _render(pt: PersistentTile) -> str:
    return "\n".join(pt.render(RenderCtx(indent=0)))


def test_inner_threadtile_renders_cooperative():
    """The inner ThreadTile renders under inside_grid_tile=True — threadIdx
    decode, no linear-tid bounds guard (the __global__ supplies threads)."""
    src = _render(_persistent())
    assert "int m_t = threadIdx.x / (16);" in src
    assert "int n_t = threadIdx.x % 16;" in src
    assert "blockIdx.x * blockDim.x" not in src  # standalone-tid path must NOT fire


def test_deps_are_the_work_range_arrays():
    pt = _persistent()
    assert pt.deps() == ("streamk_work_start", "streamk_work_end")
    assert pt.binds_axes() == frozenset({"m_b", "n_b"})


def test_with_bodies_preserves_fields():
    pt = PersistentTile(
        axes=(Axis("m_b", 8), Axis("n_b", 4)),
        body=Body((Write(output="C", index=(Var("m_b"),), value="acc"),)),
        k_blocks=7,
        work_start="ws",
        work_end="we",
    )
    pt2 = pt.with_bodies((Body((Write(output="C", index=(Var("m_b"),), value="acc2"),)),))
    assert isinstance(pt2, PersistentTile)
    assert (pt2.k_blocks, pt2.work_start, pt2.work_end) == (7, "ws", "we")


def test_pretty_label():
    pt = _persistent(k_blocks=4)
    assert any("persistent[num_sms] streamk K_blocks=4" in line for line in pt.pretty())


def test_frozen_and_hashable():
    pt = _persistent()
    assert hash(pt) is not None
    assert pt == _persistent()


def test_repr_eval_round_trip():
    """``repr(PersistentTile(...))`` must eval back in the Graph serializer's
    Stmt scope — the JSON dump/load path stringifies Stmts via repr."""
    pt = _persistent()
    rebuilt = _eval_stmt(repr(pt))
    assert isinstance(rebuilt, PersistentTile)
    assert rebuilt == pt
    assert (rebuilt.k_blocks, rebuilt.work_start, rebuilt.work_end) == (pt.k_blocks, pt.work_start, pt.work_end)


# ---------------------------------------------------------------------------
# Runtime-bounded StridedTile (Stream-K Phase B keystone): the adaptive K-loop
# runs a per-CTA sub-range [k_lo, k_hi) of the K chunks, both bounds runtime.
# ---------------------------------------------------------------------------


def _strided(stop=None):
    return StridedTile(
        axis=Axis("k_o", 16),
        body=Body((Accum(name="acc", value="v"),)),
        start=Var("k_lo"),
        step=Literal(1, "int"),
        stop=stop,
    )


def test_strided_runtime_stop_renders_dynamic_upper_bound():
    src = "\n".join(_strided(stop=Var("k_hi")).render(RenderCtx(indent=0)))
    assert "for (int k_o = k_lo; k_o < k_hi; k_o += 1) {" in src
    assert "float acc = 0.0f;" in src  # accumulator init prelude still fires


def test_strided_stop_none_falls_back_to_static_extent():
    src = "\n".join(_strided(stop=None).render(RenderCtx(indent=0)))
    assert "for (int k_o = k_lo; k_o < 16; k_o += 1) {" in src


def test_strided_exprs_include_stop_for_rewrites():
    st = _strided(stop=Var("k_hi"))
    names = {v for e in st.exprs() for v in e.free_vars()}
    assert {"k_lo", "k_hi"} <= names


def test_strided_with_bodies_preserves_stop():
    st = _strided(stop=Var("k_hi"))
    st2 = st.with_bodies((Body((Accum(name="acc", value="w"),)),))
    assert isinstance(st2, StridedTile)
    assert st2.stop == Var("k_hi")
    assert st2.start == Var("k_lo")


def test_strided_runtime_stop_round_trips():
    st = _strided(stop=Var("k_hi"))
    assert _eval_stmt(repr(st)) == st


# ---------------------------------------------------------------------------
# Adaptive PersistentTile render (Stream-K Phase B3a): the MAC-segment walk
# that implements kernel/_streamk.cta_segments in CUDA — a while-loop over the
# CTA's MAC slice, per-segment decode of (tile, k_lo, k_hi), a runtime-bounded
# partial K-loop, and a full-vs-partial write branch.
# ---------------------------------------------------------------------------


def _adaptive(k_blocks=4):
    from deplodock.compiler.ir.expr import BinaryExpr
    from deplodock.compiler.ir.tile.ir import STREAMK_K_HI, STREAMK_K_LO, Cond, StridedTile

    kloop = StridedTile(
        axis=Axis("k_o", k_blocks),
        body=Body((Accum(name="acc", value="v"),)),
        start=Var(STREAMK_K_LO),
        step=Literal(1, "int"),
        stop=Var(STREAMK_K_HI),
    )
    is_full = BinaryExpr(
        "&&",
        BinaryExpr("==", Var(STREAMK_K_LO), Literal(0, "int")),
        BinaryExpr("==", Var(STREAMK_K_HI), Literal(k_blocks, "int")),
    )
    branch = Cond(
        cond=is_full,
        body=Body((Write(output="C", index=(Var("m_b"), Var("n_b")), value="acc"),)),
        else_body=Body((Write(output="scratch", index=(Var("m_b"), Var("n_b")), value="acc"),)),
    )
    inner = ThreadTile(axes=(Axis("m_t", 16), Axis("n_t", 16)), body=Body((kloop, branch)))
    return PersistentTile(axes=(Axis("m_b", 8), Axis("n_b", 4)), body=Body((inner,)), k_blocks=k_blocks)


def test_adaptive_renders_mac_segment_walk():
    src = "\n".join(_adaptive(k_blocks=4).render(RenderCtx(indent=0)))
    # MAC-walk over the CTA's slice, advancing by segment length.
    assert "int __mac = streamk_work_start[blockIdx.x];" in src
    assert "while (__mac < __wend) {" in src
    assert "__mac = __seg_end;" in src
    # Per-segment decode (kernel/_streamk: tile_id = mac//K_blocks, k_lo = mac%K_blocks).
    assert "int __tile = __mac / 4;" in src
    assert "int streamk_k_lo = __mac % 4;" in src
    assert "int __tile_end = __mac - streamk_k_lo + 4;" in src
    assert "int __seg_end = (__wend < __tile_end) ? __wend : __tile_end;" in src
    assert "int streamk_k_hi = streamk_k_lo + (__seg_end - __mac);" in src
    # Block axes decode off __tile (not blockIdx, not a fixed tile_iter).
    assert "int m_b = __tile / (4);" in src
    assert "int n_b = __tile % 4;" in src


def test_adaptive_partial_k_loop_and_write_branch():
    src = "\n".join(_adaptive(k_blocks=4).render(RenderCtx(indent=0)))
    # Runtime-bounded partial K-loop over [k_lo, k_hi).
    assert "for (int k_o = streamk_k_lo; k_o < streamk_k_hi; k_o += 1) {" in src
    # Full tile → output; boundary partial → scratch.
    assert "if (streamk_k_lo == 0 && streamk_k_hi == 4) {" in src
    assert "C[m_b + n_b] = acc;" in src
    assert "scratch[m_b + n_b] = acc;" in src


def test_adaptive_with_bodies_and_round_trip_preserve_k_blocks():
    pt = _adaptive(k_blocks=8)
    pt2 = pt.with_bodies((pt.body,))
    assert pt2.k_blocks == 8
    assert _eval_stmt(repr(pt)).k_blocks == 8


def test_serialtile_runtime_bounds_render_and_preserve_kind():
    """The K-loop carries runtime [lo, hi) bounds but stays a SerialTile with its
    kind — so staging / K_o detection (which keys on the flavor + kind) still
    recognizes it. None bounds → the static 0..extent."""
    from deplodock.compiler.ir.tile.ir import STREAMK_K_HI, STREAMK_K_LO, Accum, SerialTile

    bounded = SerialTile(
        axis=Axis("k_o", 4),
        body=Body((Accum(name="acc", value="v"),)),
        kind="serial_outer",
        lo=Var(STREAMK_K_LO),
        hi=Var(STREAMK_K_HI),
    )
    src = "\n".join(bounded.render(RenderCtx(indent=0)))
    assert "for (int k_o = streamk_k_lo; k_o < streamk_k_hi; k_o++) {" in src
    assert bounded.kind == "serial_outer"
    assert {v for e in bounded.exprs() for v in e.free_vars()} == {"streamk_k_lo", "streamk_k_hi"}

    plain = SerialTile(axis=Axis("k_o", 4), body=Body((Accum(name="acc", value="v"),)), kind="serial_outer")
    assert "for (int k_o = 0; k_o < 4; k_o++) {" in "\n".join(plain.render(RenderCtx(indent=0)))
    assert _eval_stmt(repr(bounded)) == bounded
    # with_bodies preserves the bounds + kind.
    rebound = bounded.with_bodies((bounded.body,))
    assert (rebound.lo, rebound.hi, rebound.kind) == (bounded.lo, bounded.hi, "serial_outer")
