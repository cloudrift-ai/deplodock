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
from deplodock.compiler.ir.stmt.base import RenderCtx
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.tile.ir import PersistentTile, ThreadTile, Write


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
