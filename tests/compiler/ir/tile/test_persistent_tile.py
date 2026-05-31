"""Render + structural tests for the PersistentTile primitive (Stream-K M1).

Hand-builds a minimal matmul-shape ``PersistentTile(M_b, N_b) > ThreadTile``
and exercises the primitive's contract in isolation: the per-CTA work-range
loop, the row-major block-axis decode against the loop variable (NOT
``blockIdx.x``), the cooperative threadIdx decode of the inner ThreadTile, the
work-range array deps, and the repr/eval round-trip the Graph serializer needs.

No upstream pass emits ``PersistentTile`` yet — ``tile/018_persistent_streamk``
lands in a later milestone; this pins the IR surface the rewrite will target.
"""

from __future__ import annotations

from deplodock.compiler.graph import _eval_stmt
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt.base import RenderCtx
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.tile.ir import Accum, PersistentTile, StridedTile, ThreadTile, Write


def _persistent(m_b: int = 8, n_b: int = 4) -> PersistentTile:
    """``PersistentTile(M_b, N_b) > ThreadTile(M_t, N_t) > Write``.

    The inner ThreadTile + Write is a placeholder standing in for the real
    matmul cell — the milestone's surface is the persistent work-loop and the
    block-axis decode, not the cooperative cell body.
    """
    inner = ThreadTile(axes=(Axis("m_t", 16), Axis("n_t", 16)), body=Body((Write(output="C", index=(), value="acc"),)))
    return PersistentTile(axes=(Axis("m_b", m_b), Axis("n_b", n_b)), body=Body((inner,)))


def _render(pt: PersistentTile) -> str:
    return "\n".join(pt.render(RenderCtx(indent=0)))


def test_render_emits_work_range_loop():
    """Each CTA reads its [start, end) range from the two int32 arrays and
    walks it with a serial loop — the body indents under the loop's brace."""
    src = _render(_persistent())
    assert "int __wbeg = streamk_work_start[blockIdx.x];" in src
    assert "int __wend = streamk_work_end[blockIdx.x];" in src
    assert "for (int tile_iter = __wbeg; tile_iter < __wend; tile_iter++) {" in src
    assert src.strip().endswith("}")


def test_block_axes_decode_from_work_var_not_blockidx():
    """The block axes decode row-major from the work-loop variable, NOT
    ``blockIdx.x`` — that's the whole point: one CTA computes many tiles."""
    src = _render(_persistent(m_b=8, n_b=4))
    assert "int m_b = tile_iter / (4);" in src
    assert "int n_b = tile_iter % 4;" in src
    assert "blockIdx.x" not in src.split("for (int tile_iter")[1]  # no blockIdx past the loop head


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
        body=Body((Write(output="C", index=(), value="acc"),)),
        work_start="ws",
        work_end="we",
        work_var="wi",
    )
    pt2 = pt.with_bodies((Body((Write(output="C", index=(), value="acc2"),)),))
    assert isinstance(pt2, PersistentTile)
    assert (pt2.work_start, pt2.work_end, pt2.work_var) == ("ws", "we", "wi")


def test_pretty_label():
    pt = _persistent()
    assert any("persistent[num_sms]" in line for line in pt.pretty())


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
    assert (rebuilt.work_start, rebuilt.work_end, rebuilt.work_var) == (pt.work_start, pt.work_end, pt.work_var)


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


def test_adaptive_property_and_default_is_tile_parallel():
    assert _adaptive().adaptive is True
    assert _persistent().adaptive is False  # k_blocks=None default
    # tile-parallel render keeps the plain for-loop, no MAC-walk.
    assert "while (__mac" not in _render(_persistent())


def test_adaptive_with_bodies_and_round_trip_preserve_k_blocks():
    pt = _adaptive(k_blocks=8)
    pt2 = pt.with_bodies((pt.body,))
    assert pt2.k_blocks == 8 and pt2.adaptive
    assert _eval_stmt(repr(pt)).k_blocks == 8
