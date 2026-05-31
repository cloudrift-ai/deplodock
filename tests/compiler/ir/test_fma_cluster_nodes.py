"""The ``FmaCluster`` kernel-IR node (plans/inline-fma-cluster.md, M1+M2).

``FmaCluster`` wraps a flat per-thread matmul cell (``Load`` A/B,
``Assign(multiply)``, ``Accum(add)``) assembled by
``kernel/120_assemble_fma_clusters``. It carries the matched cell in ``body``
(the round-trip / M2-placeholder payload — render re-emits it verbatim) plus
the A/B/acc SSA-name views the M3 inline-PTX emitter will drive from.

These tests pin the node contract: frozen-dataclass hashability/equality, the
``repr`` → eval round-trip the JSON serializer relies on, a full
``Graph.to_dict`` → ``from_dict`` body round-trip, and the transparent
structural surface (``nested`` / ``defines`` / ``deps`` / ``external_reads``
derive from the carried cell).
"""

from __future__ import annotations

from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Graph, Tensor, _stmt_eval_scope
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.kernel.ir import FmaCluster, KernelOp
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load

MUL = ElementwiseImpl("multiply")
ADD = ElementwiseImpl("add")


def _cell_body() -> tuple:
    """A canonical FM=2, FN=1 outer-product cell (one K-iteration)."""
    return (
        Load(names=("a0",), input="a_smem", index=(Var("k"), Var("m0"))),
        Load(names=("a1",), input="a_smem", index=(Var("k"), Var("m1"))),
        Load(names=("b0",), input="b_smem", index=(Var("k"), Var("n0"))),
        Assign(name="v0", op=MUL, args=("a0", "b0")),
        Assign(name="v1", op=MUL, args=("a1", "b0")),
        Accum(name="acc0", value="v0", op=ADD),
        Accum(name="acc1", value="v1", op=ADD),
    )


def _cluster() -> FmaCluster:
    return FmaCluster(
        body=_cell_body(),
        a_names=("a0", "a1"),
        b_names=("b0",),
        acc_names=("acc0", "acc1"),
        fm=2,
        fn=1,
    )


def test_frozen_hashable_and_equal():
    c = _cluster()
    assert hash(c) == hash(_cluster())
    assert c == _cluster()
    rebuilt = FmaCluster(**{f: getattr(c, f) for f in c.__dataclass_fields__})
    assert rebuilt == c and hash(rebuilt) == hash(c)


def test_defaults():
    c = _cluster()
    assert c.dtype is F32
    assert c.policy == "B_INNER"


def test_structural_surface_derives_from_body():
    """``nested`` exposes the cell so analysis sees it; ``defines`` / ``deps`` /
    ``external_reads`` aggregate the carried leaves. ``defines`` covers every
    SSA name the cell binds (operands, products, accumulators); ``deps`` is the
    bound index axis Vars the loads reference (``k`` / ``m*`` / ``n*``) — the
    accumulator loop-carry is implicit, so cell-internal SSA never leaks; the
    smem buffers are external reads."""
    c = _cluster()
    (nested,) = c.nested()
    assert tuple(nested) == _cell_body()
    assert set(c.defines()) == {"a0", "a1", "b0", "v0", "v1", "acc0", "acc1"}
    deps = set(c.deps())
    assert deps == {"k", "m0", "m1", "n0"}
    assert deps.isdisjoint(c.defines())  # nothing the cell binds leaks as a dep
    assert set(c.external_reads()) == {"a_smem", "b_smem"}


def test_with_bodies_round_trips_the_cell():
    c = _cluster()
    rebuilt = c.with_bodies((Body(c.body),))
    assert rebuilt == c


def test_repr_eval_round_trip_in_stmt_scope():
    """Op bodies serialize via ``repr`` and reload via ``eval`` in
    ``_stmt_eval_scope`` — ``FmaCluster`` was added to that scope."""
    scope = dict(_stmt_eval_scope())
    c = _cluster()
    back = eval(repr(c), scope)  # noqa: S307 — trusted IR repr, sandboxed scope
    assert back == c and hash(back) == hash(c)


def test_render_emits_b_inner_inline_ptx():
    """``render`` emits the operand Loads as C then one ``asm volatile`` of
    ``fm*fn`` ``fma.rn.f32`` in B_INNER order (n outer, m inner). Operand slots:
    accumulators (``+f``, slots ``0..n_acc-1``, row-major ``acc[m*fn+n]``), then
    A operands (``f``, ``n_acc+m``), then B operands (``f``, ``n_acc+fm+n``).
    Holding b[n] in one slot across the inner m-run is what pins it to port Rb."""
    from deplodock.compiler.ir.stmt import RenderCtx

    # 2×3 cell: a0,a1 (A) × b0,b1,b2 (B) → 6 accumulators.
    body = (
        Load(names=("a0",), input="a_smem", index=(Var("k"), Var("r0"))),
        Load(names=("a1",), input="a_smem", index=(Var("k"), Var("r1"))),
        Load(names=("b0",), input="b_smem", index=(Var("k"), Var("c0"))),
        Load(names=("b1",), input="b_smem", index=(Var("k"), Var("c1"))),
        Load(names=("b2",), input="b_smem", index=(Var("k"), Var("c2"))),
    )
    c = FmaCluster(
        body=body,
        a_names=("a0", "a1"),
        b_names=("b0", "b1", "b2"),
        acc_names=tuple(f"acc{m * 3 + n}" for m in range(2) for n in range(3)),
        fm=2,
        fn=3,
    )
    lines = c.render(RenderCtx(shapes={"a_smem": (64, 8), "b_smem": (64, 8)}))
    text = "\n".join(lines)
    fmas = [ln.strip() for ln in lines if "fma.rn.f32" in ln]
    assert len(fmas) == 6
    # B_INNER walk: n=0 → m=0,1 ; n=1 → m=0,1 ; n=2 → m=0,1. n_acc=6, fm=2.
    # acc_slot=m*3+n, a_slot=6+m, b_slot=6+2+n.
    assert fmas[0] == '"fma.rn.f32 %0, %6, %8, %0;\\n"'  # (m=0,n=0): acc0, a0, b0
    assert fmas[1] == '"fma.rn.f32 %3, %7, %8, %3;\\n"'  # (m=1,n=0): acc3, a1, b0 — b0 still %8 → Rb reuse
    assert fmas[2] == '"fma.rn.f32 %1, %6, %9, %1;\\n"'  # (m=0,n=1): acc1, a0, b1
    # constraint lists: 6 in/out accumulators, then 2 A + 3 B read-only inputs.
    assert '"+f"(acc0)' in text and '"+f"(acc5)' in text
    assert '"f"(a0)' in text and '"f"(b2)' in text
    assert "asm volatile(" in text and text.rstrip().endswith(");")


def test_graph_json_body_round_trip():
    """Full ``Graph.to_dict`` → ``from_dict`` round-trip with the node inside a
    ``KernelOp`` body — the path ``deplodock run --ir <dump>`` exercises."""
    g = Graph()
    g.add_node(KernelOp(body=Body((_cluster(),)), name="k_kernel"), [], Tensor("k", (2, 1)), node_id="k")
    (stmt,) = Graph.from_dict(g.to_dict()).nodes["k"].op.body
    assert stmt == _cluster()
