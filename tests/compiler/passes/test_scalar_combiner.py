"""The scalar combiner (``ir/twist.ScalarCombiner``) — CPU-only, the structural oracle for
"realize the carrier merge at the scalar (undistributed) tier".

Asserts that ``ScalarCombiner.combine`` projects the ``flash_combine`` online-softmax ``Monoid`` onto
loose scalar statements — the identity projection — reproducing the carrier's ``merge`` algebra (the
same faithfulness ``render_merge_program`` has, but as IR split into the carrier-generic
``CombinePhases`` it shares with the fragment ``MmaTwist``). Because these are loose statements (not a
``Monoid`` rendered via ``render_merge_program``'s ``state_names``), a carried-state update is a
``Reassign`` (``Assign`` always declares, which would shadow the ``Init``'d carried value). No GPU, no
fragment ops.
"""

from __future__ import annotations

from collections import Counter

from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.kernel.ir import Reassign
from emmy.compiler.ir.stmt import Assign, Init, Monoid
from emmy.compiler.ir.twist import ScalarCombiner
from emmy.compiler.pipeline.passes.loop.recognize._flash import flash_combine


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


def _normalized(stmts) -> list[tuple]:
    """Inline the temp + ``Reassign`` carried-state rebinds back to ``(state, op, args)`` so the
    realized scalar phases compare against the carrier's ``merge`` algebra: a ``Reassign(state, tmp)``
    over a ``__sc`` rebind temp becomes the temp's ``(op, args)``; a ``Reassign(state, src)`` copy
    becomes ``("copy", (src,))``; the ``__sc`` temps drop; every other ``Assign`` passes through."""
    by_name = {s.name: s for s in stmts if isinstance(s, Assign)}
    out: list[tuple] = []
    for s in stmts:
        if isinstance(s, Reassign):
            if s.value.endswith("__sc"):
                src = by_name[s.value]
                out.append((s.name, src.op.name, tuple(src.args)))
            else:
                out.append((s.name, "copy", (s.value,)))
        elif isinstance(s, Assign) and not s.name.endswith("__sc"):
            out.append(_sig(s))
    return out


def test_scalar_phases_are_scalar_only() -> None:
    # No fragment / row-distributed nodes: every emitted stmt is a plain Assign / Reassign / Init.
    fs = ScalarCombiner().combine(_carrier())
    assert all(isinstance(s, Init) for s in fs.init)
    for phase in (fs.merge, fs.rescale, fs.consume, fs.epilogue):
        assert all(isinstance(s, (Assign, Reassign)) for s in phase)
    assert fs.update == ()


def test_projection_is_faithful_to_the_merge_program() -> None:
    # The identity projection: merge + rescale + consume (with the Reassign rebinds inlined) reproduce
    # the carrier's merge assignments exactly — the scalar tier neither drops nor invents work.
    carrier = _carrier()
    fs = ScalarCombiner().combine(carrier)
    realized = Counter(_normalized([*fs.merge, *fs.rescale, *fs.consume]))
    expected = Counter(_sig(a) for a in carrier.merge)
    assert realized == expected


def test_carried_states_are_init_then_reassigned_never_declared() -> None:
    # The correctness invariant (guards the loose-Assign shadowing bug): every carried state is
    # declared once by an Init and rebound by a Reassign — never the target of an Assign (which would
    # declare a fresh local, shadowing the Init'd carried value and leaving O / l at their identity).
    carrier = _carrier()
    fs = ScalarCombiner().combine(carrier)
    states = set(carrier.state)  # m_i, l_i, O_i
    body = [*fs.merge, *fs.rescale, *fs.consume]
    assert states <= {s.name for s in fs.init if isinstance(s, Init)}
    assert states <= {s.name for s in body if isinstance(s, Reassign)}
    assert not (states & {s.name for s in body if isinstance(s, Assign)})


def test_init_seeds_states_and_declares_accumulator() -> None:
    # init declares all three carried states with their fold-identity op: m (max → −inf),
    # l (add → 0), O (add → 0). The accumulator O is the last (declare_accum after the seeds).
    fs = ScalarCombiner().combine(_carrier())
    seeds = {s.name: s.op.name for s in fs.init}
    assert seeds == {"m_i": "maximum", "l_i": "add", "O_i": "add"}
    assert fs.init[-1].name == "O_i"  # the accumulator declared after the stats-state seeds


def test_epilogue_normalizes_accumulator_by_denominator() -> None:
    # O / l — the add-fold stats state l is the denominator (a Reassign rebind, inlined here).
    fs = ScalarCombiner().combine(_carrier())
    assert _normalized(list(fs.epilogue)) == [("O_i", "divide", ("O_i", "l_i"))]


def test_statement_order_is_dependency_valid() -> None:
    # Walking init → merge → rescale → consume in order, every read is already defined (a prior
    # stmt, a carried state, or a partial input) — the emitted order is a valid schedule.
    carrier = _carrier()
    fs = ScalarCombiner().combine(carrier)
    defined = set(carrier.partial)  # s, v arrive as the per-step partial
    for s in (*fs.init, *fs.merge, *fs.rescale, *fs.consume):
        if isinstance(s, Assign):
            reads = s.args
        elif isinstance(s, Reassign):
            reads = (s.value,)
        else:
            reads = ()
        for arg in reads:
            assert arg in defined, f"{arg} used before definition in {s}"
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
    assert Counter(_normalized(list(fs.merge))) == Counter(_sig(a) for a in carrier.merge)


def test_pure_reduction_carrier_is_a_single_fold() -> None:
    # The simplest non-twisted carrier: acc += x. One seed (add → 0), one fold, nothing else.
    fs = ScalarCombiner().combine(_sum_carrier())
    assert fs.rescale == () and fs.consume == () and fs.epilogue == ()
    assert _normalized(list(fs.merge)) == [("acc", "add", ("acc", "x"))]
    assert [(s.name, s.op.name) for s in fs.init] == [("acc", "add")]
