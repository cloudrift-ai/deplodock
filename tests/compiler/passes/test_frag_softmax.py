"""The fragment-tier flash-softmax realizer (``assembly/_frag_softmax.py``) — CPU-only,
the structural oracle for "generate the m16n8 softmax phases from the carrier algebra".

Asserts that ``realize_fragment_softmax`` maps the ``flash_combine`` ``Monoid``'s ``merge``
program onto the right fragment ops + row-distributed scalars (the analog of the cooperative
path's ``emit_combine``), without any GPU compile.
"""

from __future__ import annotations

from deplodock.compiler.ir.kernel.ir import FragmentCausalMask, FragmentExp, FragmentRowReduce, FragmentScale, Reassign
from deplodock.compiler.ir.stmt import Assign, Init
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._frag_softmax import (
    FragGeom,
    realize_fragment_softmax,
    realize_score_mask,
)
from deplodock.compiler.pipeline.passes.loop.recognize._flash import flash_combine


def _geom(nd: int = 4) -> FragGeom:
    return FragGeom(
        atom_m=16,
        atom_n=8,
        score_frags=("Sf0_frag", "Sf1_frag"),
        prob_frags=("Pf0", "Pf1"),
        accum_frags=tuple(f"Of{n}" for n in range(nd)),
    )


def _carrier():
    return flash_combine("m_i", "l_i", "O_i", "s", "v")


def test_init_seeds_the_fold_identities():
    """Two stats states × 2 rows: m is a max-fold (-inf identity), l an add-fold (0)."""
    fs = realize_fragment_softmax(_carrier(), geom=_geom())
    inits = [s for s in fs.init if isinstance(s, Init)]
    assert len(inits) == 4, [s.pretty()[0] for s in fs.init]
    by_op = {s.name: s.op.name for s in inits}
    assert by_op == {"m_i0": "maximum", "m_i1": "maximum", "l_i0": "add", "l_i1": "add"}


def test_merge_emits_two_rowreduces_then_the_exp_map():
    """rowmax over the score frags (max), the P = exp(S - m) map, rowsum over the prob frags
    (add) — the fragment-distributed online-softmax, recovered from the scalar program."""
    fs = realize_fragment_softmax(_carrier(), geom=_geom())
    reduces = [s for s in fs.merge if isinstance(s, FragmentRowReduce)]
    exps = [s for s in fs.merge if isinstance(s, FragmentExp)]

    assert len(reduces) == 2, [s.pretty()[0] for s in fs.merge]
    rowmax, rowsum = reduces
    # First fold reduces the SCORE fragments by max; second reduces the PROB fragments by add.
    assert rowmax.op.name == "maximum" and rowmax.frags == ("Sf0_frag", "Sf1_frag")
    assert rowsum.op.name == "add" and rowsum.frags == ("Pf0", "Pf1")

    # One FragmentExp per N-atom: P = exp(score - m_new), reading the score frags into the prob frags.
    assert len(exps) == 2
    assert [e.src for e in exps] == ["Sf0_frag", "Sf1_frag"]
    assert [e.out for e in exps] == ["Pf0", "Pf1"]
    # both rows subtract the per-row new-max (same scalar pair feeds top/bot of every N-atom).
    assert {e.top_sub for e in exps} == {exps[0].top_sub} and {e.bot_sub for e in exps} == {exps[0].bot_sub}


def test_state_is_reassigned_in_merge_update_is_empty():
    """m and l are loop-carried: reassigned in-place at the end of merge (after every read of
    their old value), so the update phase is empty."""
    fs = realize_fragment_softmax(_carrier(), geom=_geom())
    assert fs.update == ()
    reassigned = {s.name for s in fs.merge if isinstance(s, Reassign)}
    assert reassigned == {"m_i0", "m_i1", "l_i0", "l_i1"}


def test_rescale_scales_every_accumulator_by_alpha():
    """The twist O *= alpha is one FragmentScale per D-atom; top/bot share the same alpha pair
    (the rescale is the accum carrier's only realized step — p·v + O += … are the P@V Mma)."""
    for nd in (2, 4, 8):
        fs = realize_fragment_softmax(_carrier(), geom=_geom(nd))
        scales = [s for s in fs.rescale if isinstance(s, FragmentScale)]
        assert len(scales) == nd
        assert [s.frag for s in scales] == [f"Of{n}" for n in range(nd)]
        tops = {s.top for s in scales}
        bots = {s.bot for s in scales}
        assert len(tops) == 1 and len(bots) == 1  # one alpha pair, broadcast to every accumulator


def test_epilogue_normalizes_every_accumulator_by_the_denominator():
    """O /= l per D-atom — the add-fold state (l) is the denominator; recovered as 1/l."""
    fs = realize_fragment_softmax(_carrier(), geom=_geom(nd=4))
    scales = [s for s in fs.epilogue if isinstance(s, FragmentScale)]
    assert len(scales) == 4
    assert all(s.top == "(1.0f/l_i0)" and s.bot == "(1.0f/l_i1)" for s in scales)


def test_scalar_updates_are_row_distributed():
    """Every per-row scalar stat is two scalars (rows g / g+8) — the 0/1 suffixing."""
    fs = realize_fragment_softmax(_carrier(), geom=_geom())
    scalar_names = {s.name for s in fs.merge if isinstance(s, Assign)}
    # the subtract (m - m_new) and the exp (alpha) both appear suffixed 0 and 1
    assert any(n.endswith("0") for n in scalar_names) and any(n.endswith("1") for n in scalar_names)
    assert all(n.endswith("0") or n.endswith("1") for n in scalar_names), scalar_names


def test_score_mask_is_one_causal_mask_per_score_frag():
    fs_geom = _geom()
    from deplodock.compiler.ir.expr import Literal, Var

    masks = realize_score_mask(fs_geom, q_row_base=Var("qb"), kv_col_bases=(Literal(0, "int"), Literal(8, "int")))
    assert [m.frag for m in masks] == ["Sf0_frag", "Sf1_frag"]
    assert all(isinstance(m, FragmentCausalMask) for m in masks)
