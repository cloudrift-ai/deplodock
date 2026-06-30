"""The fragment-tier combiner (``ir/twist.MmaTwist``) — CPU-only, the structural oracle for
"generate the m16n8 phases + masks from the carrier algebra".

Asserts that ``MmaTwist.combine`` projects the ``flash_combine`` ``Monoid``'s ``merge`` onto the
right fragment ops + row-distributed scalars (the analog of the cooperative path's ``emit_combine``)
and ``MmaTwist.mask`` builds the coordinate-predicated masks, without any GPU compile. The
``MmaTwist`` is built from the produce / consume ``Mma`` cells — it derives the fragment register
roles + layout off them (``partial_frags`` = ``<c>_frag``, ``accum_frags`` = ``c``).
"""

from __future__ import annotations

from emmy.compiler.dtype import F16, F32
from emmy.compiler.ir.kernel.ir import FRAG, ROW, UNIFORM, FragmentApply, FragmentMask, FragmentRowReduce, Reassign, RegFragment
from emmy.compiler.ir.stmt import Assign, Init, Mma
from emmy.compiler.ir.tile.ir import Atom
from emmy.compiler.ir.twist import MmaTwist
from emmy.compiler.pipeline.passes.loop.recognize._flash import flash_combine

_ATOM = Atom(name="mma_m16n8k16_f16", shape=(16, 8, 16), operand_dtypes=(("a", F16), ("b", F16), ("c", F32)), group_size=32)


def _twist(nd: int = 4) -> MmaTwist:
    produce = (
        Mma(c="Sf0", a="qa0", b="kb0", atom=_ATOM, b_trans=True),
        Mma(c="Sf1", a="qa1", b="kb1", atom=_ATOM, b_trans=True),
    )
    consume = tuple(Mma(c=f"Of{n}", a="pa", b=f"vb{n}", atom=_ATOM) for n in range(nd))
    return MmaTwist(produce=produce, consume=consume)


def _carrier():
    return flash_combine("m_i", "l_i", "O_i", "s", "v")


def test_init_seeds_the_fold_identities():
    """Two stats states × 2 rows: m is a max-fold (-inf identity), l an add-fold (0)."""
    fs = _twist().combine(_carrier())
    inits = [s for s in fs.init if isinstance(s, Init)]
    assert len(inits) == 4, [s.pretty()[0] for s in fs.init]
    by_op = {s.name: s.op.name for s in inits}
    assert by_op == {"m_i0": "maximum", "m_i1": "maximum", "l_i0": "add", "l_i1": "add"}
    # combine also declares the accum C fragments (the consume Mma cells) in init.
    accum_decls = [s for s in fs.init if isinstance(s, RegFragment)]
    assert [r.name for r in accum_decls] == ["Of0", "Of1", "Of2", "Of3"]


def test_merge_emits_two_rowreduces_then_the_exp_map():
    """rowmax over the score frags (max), the P = exp(S - m) map, rowsum over the prob frags
    (add) — the fragment-distributed online-softmax, recovered from the scalar program. The exp
    map is now two generic ``FragmentApply``s per N-atom (subtract then exp), not the former fused
    ``FragmentExp``."""
    fs = _twist().combine(_carrier())
    reduces = [s for s in fs.merge if isinstance(s, FragmentRowReduce)]
    applies = [s for s in fs.merge if isinstance(s, FragmentApply)]

    assert len(reduces) == 2, [s.pretty()[0] for s in fs.merge]
    rowmax, rowsum = reduces
    # First fold reduces the SCORE fragments by max; second reduces the PROB fragments by add.
    assert rowmax.op.name == "maximum" and rowmax.frags == ("Sf0_frag", "Sf1_frag")
    assert rowsum.op.name == "add" and rowsum.frags == ("Pf0", "Pf1")

    # Per N-atom: subtract(score, m_new) then exp(...) -> prob. The exp reads a FRAG, subtracts a
    # per-row (ROW) scalar; the exp result lands in the geometry's prob fragments.
    subs = [a for a in applies if a.op.name == "subtract"]
    exps = [a for a in applies if a.op.name == "exp"]
    assert len(subs) == 2 and len(exps) == 2
    assert [a.args[0] for a in subs] == ["Sf0_frag", "Sf1_frag"]  # FRAG = the score
    assert all(a.kinds == (FRAG, ROW) for a in subs)
    assert {a.args[1] for a in subs} == {subs[0].args[1]}  # same per-row new-max for every N-atom
    assert [e.out for e in exps] == ["Pf0", "Pf1"] and all(e.kinds == (FRAG,) for e in exps)


def test_state_is_reassigned_in_merge_update_is_empty():
    """m and l are loop-carried: reassigned in-place at the end of merge (after every read of
    their old value), so the update phase is empty."""
    fs = _twist().combine(_carrier())
    assert fs.update == ()
    reassigned = {s.name for s in fs.merge if isinstance(s, Reassign)}
    assert reassigned == {"m_i0", "m_i1", "l_i0", "l_i1"}


def test_rescale_scales_every_accumulator_by_alpha():
    """The twist O *= alpha is one in-place ``FragmentApply`` (multiply) per D-atom; the per-row
    (ROW) alpha is shared across accumulators (the rescale is the accum carrier's only realized
    step — p·v + O += … are the P@V Mma)."""
    for nd in (2, 4, 8):
        fs = _twist(nd).combine(_carrier())
        scales = [s for s in fs.rescale if isinstance(s, FragmentApply)]
        assert len(scales) == nd
        assert all(s.op.name == "multiply" and s.in_place and s.kinds == (FRAG, ROW) for s in scales)
        assert [s.out for s in scales] == [f"Of{n}" for n in range(nd)]
        assert [s.args[0] for s in scales] == [f"Of{n}" for n in range(nd)]  # in-place: out is the first FRAG arg
        assert {s.args[1] for s in scales} == {scales[0].args[1]}  # one alpha, broadcast to every accumulator


def test_epilogue_normalizes_every_accumulator_by_the_denominator():
    """O /= l per D-atom — the add-fold state (l) is the denominator; an in-place ``FragmentApply``
    divide by the per-row denom (the former FragmentScale-by-1/l)."""
    fs = _twist(nd=4).combine(_carrier())
    scales = [s for s in fs.epilogue if isinstance(s, FragmentApply)]
    assert len(scales) == 4
    assert all(s.op.name == "divide" and s.in_place and s.kinds == (FRAG, ROW) and s.args[1] == ("l_i0", "l_i1") for s in scales)


def test_scalar_updates_are_row_distributed():
    """Every per-row scalar stat is two scalars (rows g / g+8) — the 0/1 suffixing."""
    fs = _twist().combine(_carrier())
    scalar_names = {s.name for s in fs.merge if isinstance(s, Assign)}
    # the subtract (m - m_new) and the exp (alpha) both appear suffixed 0 and 1
    assert any(n.endswith("0") for n in scalar_names) and any(n.endswith("1") for n in scalar_names)
    assert all(n.endswith("0") or n.endswith("1") for n in scalar_names), scalar_names


def test_fragment_mask_builder_masks_each_partial_twist():
    """The generic ``fragment_mask`` builder emits one ``FragmentMask`` per distributed partial
    fragment for an arbitrary coordinate predicate — no causal / boundary / softmax naming."""
    from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
    from emmy.compiler.ir.kernel.ir import FRAG_COL, FRAG_ROW

    causal = BinaryExpr(">", Var(FRAG_COL), Var(FRAG_ROW))
    masks = _twist().mask(mask_when=causal, col_bases=(Literal(0, "int"), Literal(8, "int")), row_base=Var("qb"))
    assert [m.frag for m in masks] == ["Sf0_frag", "Sf1_frag"]
    assert all(isinstance(m, FragmentMask) and m.mask_when is causal for m in masks)


def test_fragment_mask_is_one_generic_node_for_causal_and_boundary():
    """The generic ``FragmentMask`` covers causal + boundary (and any coordinate predicate): the
    render substitutes each element's absolute (row, col) into ``mask_when`` and guards the fill —
    ONE node, not two (the former FragmentCausalMask / FragmentBoundaryMask)."""
    from emmy.compiler.ir.expr import BinaryExpr, Var
    from emmy.compiler.ir.kernel.ir import FRAG_COL, FRAG_ROW, FragmentMask
    from emmy.compiler.ir.stmt.base import RenderCtx

    causal = FragmentMask(frag="S", mask_when=BinaryExpr(">", Var(FRAG_COL), Var(FRAG_ROW)), col_base=Var("kb"), row_base=Var("qb"))
    src = "\n".join(causal.render(RenderCtx(indent=1)))
    assert "if (kb + (_t * 2 + 0) > qb + _g) S[0]" in src  # elem 0: row _g, col _t*2+0
    assert "if (kb + (_t * 2 + 1) > qb + (_g + 8)) S[3]" in src  # elem 3: row _g+8, col _t*2+1

    boundary = FragmentMask(frag="S", mask_when=BinaryExpr(">=", Var(FRAG_COL), Var("seq_len")), col_base=Var("kb"))
    src2 = "\n".join(boundary.render(RenderCtx(indent=1)))
    assert "if (kb + (_t * 2 + 0) >= seq_len) S[0]" in src2  # column-only — no row term


# --- FragmentApply: the one generic pointwise node (subsumes the former FragmentExp/Scale) ---


def test_fragment_apply_renders_frag_row_and_uniform_args():
    """A FRAG arg indexes per element; a ROW arg broadcasts by row (0/1 suffix); a UNIFORM arg is
    verbatim; ``in_place`` reassigns ``out`` (no ``float`` decl). The three arg kinds are what let
    FragmentApply express every former FragmentExp / FragmentScale shape."""
    from emmy.compiler.ir.elementwise import ElementwiseImpl
    from emmy.compiler.ir.stmt.base import RenderCtx

    mul_row = FragmentApply(out="z", op=ElementwiseImpl("multiply"), args=("x", ("a0", "a1")), kinds=(FRAG, ROW))
    src = "\n".join(mul_row.render(RenderCtx(indent=1)))
    assert "float z[4];" in src
    assert "z[0] = x[0] * a0;" in src and "z[2] = x[2] * a1;" in src
    assert mul_row.deps() == ("x", "a0", "a1")  # ROW arg exposes both per-row scalars (rename-safe)

    scale = FragmentApply(out="O", op=ElementwiseImpl("multiply"), args=("O", "0.25f"), kinds=(FRAG, UNIFORM), in_place=True)
    src2 = "\n".join(scale.render(RenderCtx(indent=1)))
    assert "float O[4];" not in src2  # in-place: no declaration
    assert "O[0] = O[0] * 0.25f;" in src2 and "O[3] = O[3] * 0.25f;" in src2


def test_monoid_project_dispatches_by_distribution_role():
    """``Monoid.project`` — the generic projection driver — dispatches each merge op by its role
    under the distribution (fold = reduce over the distributed axis, pointwise = elementwise,
    scalar, carried-state), with no op cap and no shape knowledge. A recording backend captures
    the calls; ``tanh`` (never in the softmax vocabulary) flows through as a plain pointwise."""
    from emmy.compiler.ir.stmt import Assign, Monoid

    class _Rec:
        def __init__(self):
            self.calls = []

        def fold(self, name, op, src, scalar, *, is_state):
            self.calls.append(("fold", name, op.name, is_state))

        def pointwise(self, name, op, args, distributed):
            self.calls.append(("pointwise", name, op.name))

        def scalar(self, name, op, args):
            self.calls.append(("scalar", name, op.name))

        def state(self, name, op, args):
            self.calls.append(("state", name, op.name))

    # state (l,), partial (s,); merge: sq = tanh(s) [a generic distributed pointwise], then
    # l = add(l, sq) [a reduce over the distributed axis that updates the carried state].
    carrier = Monoid(state=("l",), partial=("s",), merge=(Assign("sq", "tanh", ("s",)), Assign("l", "add", ("l", "sq"))))
    rec = _Rec()
    carrier.project(carrier.merge, distributed_inputs={"s"}, dist=rec)
    assert rec.calls == [("pointwise", "sq", "tanh"), ("fold", "l", "add", True)]


def _carrier_with_generic_op():
    """A twisted ``(m, l, O)`` carrier whose STATS merge applies a generic fragment op (``relu``) to
    the score before the fold — to exercise the realizer's ``frag_apply`` branch with a real
    carrier (the algebra is contrived, only the op vocabulary matters)."""
    from emmy.compiler.ir.expr import Literal
    from emmy.compiler.ir.stmt import Assign, Monoid

    merge = (
        Assign("sq", "relu", ("s",)),  # a generic fragment op (not exp / not a fold)
        Assign("mx", "maximum", ("m_i", "sq")),  # rowmax over the relu'd score
        Assign("ds", "subtract", ("sq", "mx")),
        Assign("p", "exp", ("ds",)),  # fused exp -> prob
        Assign("l_i", "add", ("l_i", "p")),  # denom (add-fold state)
        Assign("pv", "multiply", ("p", "v")),
        Assign("O_i", "add", ("O_i", "pv")),  # value-dependent accumulation
        Assign("m_i", "copy", ("mx",)),  # state, last
    )
    return Monoid(
        state=("m_i", "l_i", "O_i"),
        partial=("s", "v"),
        merge=merge,
        identity=(Literal(-1e30), Literal(0.0), Literal(0.0)),
        commutative=True,
        axes=("kv",),
    )


def test_frag_layout_is_the_per_atom_geometry_source():
    """The per-atom layout descriptor: ``frag_layout`` is the single source the fragment nodes read
    their geometry from (n_elems, elem→row, reduce group, coord codegen); an unmodeled atom raises
    rather than miscompiling. m16n8 is 4 regs / lane, 2 rows / lane."""
    import pytest

    from emmy.compiler.ir.kernel.ir import M16N8, frag_layout

    lay = frag_layout(16, 8)
    assert lay is M16N8
    assert lay.n_elems == 4 and lay.rows_per_lane == 2 and lay.elem_row == (0, 0, 1, 1) and lay.reduce_group == 4
    with pytest.raises(NotImplementedError):
        frag_layout(8, 8)  # an atom with no modeled C-layout fails loudly


def test_realizer_emits_frag_apply_for_a_generic_carrier_op():
    """The realizer turns a generic stats-merge fragment op (``relu`` of the score, one per N-atom)
    into a ``FragmentApply`` — carrier-vocabulary generality reaching the tensor-core tier, the
    same realizer that handles softmax's exp/fold."""
    fs = _twist().combine(_carrier_with_generic_op())
    relus = [s for s in fs.merge if isinstance(s, FragmentApply) and s.op.name == "relu"]
    assert [r.out for r in relus] == ["sq_0", "sq_1"]  # one per N-atom (the 2 score frags)
    assert [r.args for r in relus] == [("Sf0_frag",), ("Sf1_frag",)]
    assert all(r.kinds == (FRAG,) for r in relus)
