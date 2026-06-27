"""The scalar combiner (``ir/twist.ScalarCombiner``) — CPU-only, the structural oracle for
"realize the carrier merge at the scalar (undistributed) tier".

Asserts that ``ScalarCombiner.combine`` projects the ``flash_combine`` online-softmax ``Monoid``
onto plain scalar ``Assign``s — the identity projection — so the emitted phases reproduce the
carrier's ``merge`` algebra exactly (the same faithfulness ``render_merge_program`` has, but as IR
split into the carrier-generic ``CombinePhases`` it shares with the fragment ``MmaTwist``). No GPU,
no fragment ops.
"""

from __future__ import annotations

from collections import Counter

from deplodock.compiler.ir.expr import Literal
from deplodock.compiler.ir.stmt import Assign, Init, Monoid
from deplodock.compiler.ir.twist import ScalarCombiner
from deplodock.compiler.pipeline.passes.loop.recognize._flash import flash_combine


def _carrier():
    return flash_combine("m_i", "l_i", "O_i", "s", "v")


def _softmax_carrier() -> Monoid:
    # Pure online softmax: state (m, d), partial (s) — NO value, NO O accumulator (non-twisted).
    t = lambda s: f"m__{s}"  # noqa: E731
    return Monoid(
        state=("m", "d"),
        partial=("s",),
        merge=(
            Assign(t("mx"), "maximum", ("m", "s")),
            Assign(t("dm"), "subtract", ("m", t("mx"))),
            Assign(t("al"), "exp", (t("dm"),)),
            Assign(t("ds"), "subtract", ("s", t("mx"))),
            Assign(t("p"), "exp", (t("ds"),)),
            Assign(t("dl"), "multiply", ("d", t("al"))),
            Assign("d", "add", (t("dl"), t("p"))),
            Assign("m", "copy", (t("mx"),)),
        ),
        identity=(Literal(-1e30), Literal(0.0)),
        commutative=True,
        axes=("kv",),
    )


def _sum_carrier() -> Monoid:
    # A pure reduction: state (acc,), partial (x,) — the simplest non-twisted carrier.
    return Monoid(state=("acc",), partial=("x",), merge=(Assign("acc", "add", ("acc", "x")),), identity=(Literal(0.0),))


def _sig(a: Assign) -> tuple:
    return (a.name, a.op.name, tuple(a.args))


def test_scalar_phases_are_all_scalar_assigns() -> None:
    # No fragment / row-distributed nodes: every emitted stmt is a plain Assign or an Init seed.
    fs = ScalarCombiner().combine(_carrier())
    assert all(isinstance(s, Init) for s in fs.init)
    for phase in (fs.merge, fs.rescale, fs.consume, fs.epilogue):
        assert all(isinstance(s, Assign) for s in phase)
    assert fs.update == ()


def test_projection_is_faithful_to_the_merge_program() -> None:
    # The identity projection: merge + rescale + consume together reproduce the carrier's merge
    # assignments exactly (same name / op / args multiset) — the scalar tier neither drops nor
    # invents work relative to render_merge_program(merge).
    carrier = _carrier()
    fs = ScalarCombiner().combine(carrier)
    realized = Counter(_sig(a) for a in (*fs.merge, *fs.rescale, *fs.consume))
    expected = Counter(_sig(a) for a in carrier.merge)
    assert realized == expected


def test_init_seeds_states_and_declares_accumulator() -> None:
    # init declares all three carried states with their fold-identity op: m (max → −inf),
    # l (add → 0), O (add → 0). The accumulator O is the last (declare_accum after the seeds).
    fs = _carrier_combine_init()
    seeds = {s.name: s.op.name for s in fs.init}
    assert seeds == {"m_i": "maximum", "l_i": "add", "O_i": "add"}
    assert fs.init[-1].name == "O_i"  # the accumulator declared after the stats-state seeds


def _carrier_combine_init():
    return ScalarCombiner().combine(_carrier())


def test_epilogue_normalizes_accumulator_by_denominator() -> None:
    # O / l — the add-fold stats state l is the denominator.
    fs = ScalarCombiner().combine(_carrier())
    assert len(fs.epilogue) == 1
    assert _sig(fs.epilogue[0]) == ("O_i", "divide", ("O_i", "l_i"))


def test_statement_order_is_dependency_valid() -> None:
    # Walking init → merge → rescale → consume in order, every arg is already defined (a prior
    # stmt, a carried state, or a partial input) — the emitted order is a valid schedule.
    carrier = _carrier()
    fs = ScalarCombiner().combine(carrier)
    defined = set(carrier.partial)  # s, v arrive as the per-step partial
    for s in (*fs.init, *fs.merge, *fs.rescale, *fs.consume):
        for arg in s.args if isinstance(s, Assign) else ():
            assert arg in defined, f"{arg} used before definition in {_sig(s) if isinstance(s, Assign) else s}"
        defined.add(s.name)


# --- non-twisted carriers (online softmax / pure reduce): combine() degenerates ---


def test_online_softmax_carrier_folds_without_accumulator() -> None:
    # A non-twisted carrier (state (m, d), partial (s), no value) runs through the SAME combine():
    # the whole merge projects as stats, the states seed, and there is NO accumulator — so rescale /
    # consume / epilogue are empty. The (m, d) softmax stats are exactly flash's embedded stats half.
    carrier = _softmax_carrier()
    fs = ScalarCombiner().combine(carrier)
    assert fs.rescale == () and fs.consume == () and fs.epilogue == ()
    # init seeds both states with their fold identity: m (max → −inf), d (add → 0).
    assert {s.name: s.op.name for s in fs.init} == {"m": "maximum", "d": "add"}
    # merge reproduces the carrier's fold algebra exactly (identity projection at the scalar tier).
    assert Counter(_sig(a) for a in fs.merge) == Counter(_sig(a) for a in carrier.merge)


def test_pure_reduction_carrier_is_a_single_fold() -> None:
    # The simplest non-twisted carrier: acc += x. One seed (add → 0), one fold, nothing else.
    fs = ScalarCombiner().combine(_sum_carrier())
    assert fs.rescale == () and fs.consume == () and fs.epilogue == ()
    assert [_sig(a) for a in fs.merge] == [("acc", "add", ("acc", "x"))]
    assert [(s.name, s.op.name) for s in fs.init] == [("acc", "add")]
