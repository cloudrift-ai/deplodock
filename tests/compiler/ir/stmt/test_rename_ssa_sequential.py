"""Regression tests for ``rename_ssa_sequential`` SSA renumbering.

The renumber assigns Load names ``in0, in1, ...`` in definition order. A
gather (a Load whose *index* references another Load's SSA name) must keep
pointing at the producing Load after the renumber — even when the producer's
new name collides with a *surviving* old name elsewhere in the body.

This guards the ``in24``-undefined embedding-lookup bug: the renumber used to
publish each Load rename into both the SSA channel (``rename``) and the axis
channel (``sigma``). The Load/Write rewriter applies both to index exprs, so an
indirect index Var was substituted twice — and a rename chain (``in2_3 → in5``
while a pre-existing ``in5`` → ``in26``) collapsed transitively, wiring the
gather to the wrong row.
"""

from __future__ import annotations

from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.stmt.leaves import Load
from deplodock.compiler.ir.stmt.normalize import rename_ssa_sequential


def _find_load(body, input_buf: str) -> Load:
    for s in Body.coerce(body).iter():
        if isinstance(s, Load) and s.input == input_buf:
            return s
    raise AssertionError(f"no Load from {input_buf!r} in body")


def test_gather_index_survives_rename_chain_collision() -> None:
    # Definition order fixes the renumber: ids loads → in0/in1, gather → in2,
    # then a Load *literally named* ``in1`` (a surviving old name) → in3. The
    # gather's index references the second ids load (old name ``idx1``), whose
    # new name ``in1`` collides with that surviving ``in1`` — the exact shape
    # of the embedding + RMSNorm-weight kernel that triggered the bug.
    body = Body(
        (
            Load(name="idx0", input="ids", index=(Var("a0"),)),
            Load(name="idx1", input="ids", index=(Var("a0"),)),
            Load(name="gather", input="w", index=(Var("idx1"), Var("a0"))),
            Load(name="in1", input="weight", index=(Var("a0"),)),
        )
    )

    out = rename_ssa_sequential(body)

    idx1_new = _find_load(out, "ids").names  # first ids load → in0
    # The second ids load renumbers to in1; the gather's indirect index must
    # resolve to *that* in1, not to the renamed weight load (in3).
    gather = _find_load(out, "w")
    weight = _find_load(out, "weight")

    assert weight.names == ("in3",)
    assert gather.index[0] == Var("in1"), gather.index[0]
    # And the gather index must NOT have been double-substituted onto the weight.
    assert gather.index[0] != Var(weight.names[0])
    # Sanity: idx0 is the very first Load → in0.
    assert idx1_new == ("in0",)
