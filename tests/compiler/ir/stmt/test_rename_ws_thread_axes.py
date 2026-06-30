"""Regression test for ``rename_ssa_sequential`` canonicalizing a
``WarpSpecialize``'s ``consumer_thread_axes``.

After ``085_warp_specialize`` removes the consumer ``ThreadTile``, the consumer
thread coords survive only as free Vars in the consumer body plus the off-body
``WarpSpecialize.consumer_thread_axes`` field — no ``ParallelTile`` binds them.
The canonical renamer used to skip that field, so a thread coord that isn't
anchored to a recorded scope (e.g. the N-thread index of a matmul whose B
operand isn't TMA-staged, hence absent from every StageBundle cache axis) kept an
uncontrolled name. When that surviving name happened to equal the canonical slot
a *different* thread coord's cache axis was renamed to, the two distinct thread
axes collapsed to one name — the materializer then emitted two identically-named
thread loops (a duplicate ``int aN`` declaration that fails to compile).
"""

from __future__ import annotations

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt import Body, Load, Write
from emmy.compiler.ir.stmt.normalize import rename_ssa_sequential
from emmy.compiler.ir.tile.ir import (
    GridTile,
    Source,
    StageBundle,
    StagePolicy,
    WarpSpecialize,
    WarpTile,
)


def _find_ws(body: Body) -> WarpSpecialize:
    for s in Body.coerce(body).iter():
        if isinstance(s, WarpSpecialize):
            return s
    raise AssertionError("no WarpSpecialize in body")


def test_ws_consumer_thread_axes_stay_distinct_after_rename() -> None:
    # Shape mirrors a post-085 warp-specialized matmul:
    #   GridTile(g0, g1) > WarpTile(role) > WarpSpecialize
    # so g0/g1/role take canonical slots a0/a1/a2. The producer stages operand A
    # over cache axis "m_t" (the M-thread coord) — recorded next, so "m_t" → a3.
    # The N-thread coord "a3" is unstaged (B isn't staged here), so it appears
    # only as a free Var + in consumer_thread_axes. Its literal name "a3"
    # collides with the slot "m_t" lands on — the exact collision that produced
    # two ``int a3`` thread loops.
    thread_m = Axis("m_t", 8)
    thread_n = Axis("a3", 16)  # literal name collides with m_t's canonical slot

    producer = Body(
        (
            StageBundle(
                sources=(
                    Source(
                        name="a_smem",
                        buf="a",
                        cache_axes=(thread_m, Axis("a_k", 16)),
                        origin=(Var("m_t"), Literal(0, "int")),
                    ),
                ),
                body=Body(()),
                policy=StagePolicy.TMA,
                buffer_count=2,
                phase=Literal(0, "int"),
                pipeline_depth=2,
            ),
        )
    )
    # Consumer reads B directly (unstaged) at the N-thread coord "a3" and writes
    # the output at (m_t, a3) — both thread coords appear as free Vars.
    consumer = Body(
        (
            Load(name="b", input="b", index=(Var("a3"),)),
            Write(output="o", index=(Var("m_t"), Var("a3")), value="b"),
        )
    )

    ws = WarpSpecialize(
        producer_body=producer,
        consumer_body=consumer,
        ring_depth=2,
        n_producer_threads=32,
        consumer_thread_axes=(thread_m, thread_n),
    )
    body = Body(
        (
            GridTile(
                axes=(Axis("g0", 4), Axis("g1", 4)),
                body=Body((WarpTile(axes=(Axis("role", 5),), body=Body((ws,))),)),
            ),
        )
    )

    out = rename_ssa_sequential(body)
    ws_out = _find_ws(out)
    names = [a.name for a in ws_out.consumer_thread_axes]

    # The two distinct thread coords must keep distinct canonical names.
    assert len(names) == len(set(names)), f"thread axes collapsed: {names}"

    # And the consumer body's free Var references must match the (renamed) axes,
    # so the materializer's ThreadTile decode binds the names the body reads.
    write = next(s for s in Body.coerce(out).iter() if isinstance(s, Write))
    body_coords = [e.name for e in write.index]
    assert body_coords == names, (body_coords, names)
