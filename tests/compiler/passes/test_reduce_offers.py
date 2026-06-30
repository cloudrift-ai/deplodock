"""Reduce-axis offer invariants — the move-composer ``reduce_offers`` /
``coop_reduce_offers`` / ``reduce_reg_offers`` generators.

Ports the still-valid invariants of the deleted ``test_fk_reduce_enumeration.py``
to the rebuilt central enumeration module
(``passes/lowering/tile/enumeration/_moves.py``). The old test pinned the legacy
``enumerate_cartesian`` generator (gone — invalid under the move architecture);
these pin the public offer functions on a real reduce ``IterDag`` derived from a
fused ``LoopOp`` (the same build path ``test_cut_offers.py`` uses).

The invariants the current offer functions guarantee:

- **divisor-cleanliness** — a static-K offer's factor product (``bk·fk·splitk`` /
  ``bk·fk·br``) divides the K extent, so the per-thread serial extent is whole;
- **register-budget bound** — a reduce register tile (``fk·reg_n·reg_m``) stays
  within the per-thread cell cap;
- **FK=1-first ordering** — the greedy (option-0) default leads with ``FK=1`` per
  thread-geometry group, so a non-tuned ``compile`` keeps the single-accumulator
  reduce;
- **split-K soundness** — a bare global reduce admits cross-CTA split-K, a fused /
  multi-accumulator one does not.

All CPU — no CUDA, no lowering.
"""

from __future__ import annotations

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.tensor.ir import ReduceOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import IterDag, iter_dag
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._knobs import MAX_CELLS_PER_THREAD
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._moves import (
    Budget,
    coop_reduce_offers,
    reduce_offers,
    reduce_reg_offers,
)

_CC = (12, 0)


def _reduce_graph(shape: tuple) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", shape), node_id="x")
    out_shape = (*shape[:-1], 1)
    g.add_node(ReduceOp(op="sum", axis=-1), ["x"], Tensor("o", out_shape), node_id="o")
    g.inputs, g.outputs = ["x"], ["o"]
    return g


def _reduce_dag(shape: tuple) -> IterDag:
    """The reduce ``IterDag`` of the single fused ``LoopOp`` for ``shape`` — the
    view the offer functions read."""
    out = Pipeline.build(LOOP_PASSES).run(_reduce_graph(shape), ctx=Context.from_target(_CC))
    lo = next(n.op for n in out.nodes.values() if type(n.op).__name__ == "LoopOp")
    return iter_dag(lo)


# --- reduce_offers (split-K K-tiling) --------------------------------


def test_reduce_offers_divide_static_k():
    """Every ``(bk, fk, splitk)`` offer's product divides the static K extent
    (so ``K_o = K/(splitk·bk·fk)`` is whole)."""
    dag = _reduce_dag((128, 2048))
    offers = reduce_offers(dag)
    assert offers, "reduce_offers produced no K-tiling"
    for bk, fk, sk in offers:
        assert 2048 % (bk * fk * sk) == 0, f"bk·fk·splitk={bk}·{fk}·{sk} does not divide K=2048"


def test_reduce_offers_fk_one_ranks_first_per_group():
    """For the greedy (option-0) default the FK=1 offer leads its FK>1 siblings
    sharing the same (bk, splitk) group — emission order is the ranking, so a
    non-tuned ``compile`` stays single-accumulator."""
    seen: set[tuple[int, int]] = set()
    for bk, fk, sk in reduce_offers(_reduce_dag((128, 2048))):
        key = (bk, sk)
        if key in seen:
            continue
        seen.add(key)
        assert fk == 1, f"FK={fk} outranked FK=1 for (bk={bk}, splitk={sk}): greedy default would change"


def test_reduce_offers_global_admits_splitk():
    """A bare global reduce (no epilogue, single accumulator) admits a cross-CTA
    split — the associative+commutative carrier licenses it."""
    splitks = {sk for _, _, sk in reduce_offers(_reduce_dag((128, 2048)))}
    assert splitks - {1}, f"global reduce produced no split-K offer; saw {sorted(splitks)}"


# --- reduce_reg_offers (register-cell budget) ------------------------


def test_reduce_reg_offers_respect_cell_budget():
    """A reduce register tile (``fk·reg_n·reg_m``) stays within the per-thread
    cell cap for every FK the K-tiling offers."""
    dag = _reduce_dag((128, 2048))
    budget = Budget()
    assert budget.max_cells == MAX_CELLS_PER_THREAD
    for _, fk, _ in reduce_offers(dag):
        reg = reduce_reg_offers(dag, budget, fk)
        assert reg, f"no register tile for fk={fk}"
        for reg_n, reg_m in reg:
            assert fk * reg_n * reg_m <= MAX_CELLS_PER_THREAD, f"fk·reg_n·reg_m={fk}·{reg_n}·{reg_m} exceeds cell cap"


# --- coop_reduce_offers (cooperative-K thread tiling) ----------------


def test_coop_reduce_offers_divide_static_k():
    """Cooperative ``(bk, fk, br, cta)`` offers: ``br·bk·fk·cta`` divides the static K
    extent, and the cooperative thread count is in range (1 ≤ br ≤ 1024)."""
    offers = coop_reduce_offers(_reduce_dag((128, 2048)))
    assert offers, "coop_reduce_offers produced no cooperative K-tiling"
    for bk, fk, br, cta in offers:
        assert 2048 % (bk * fk * br * cta) == 0, f"bk·fk·br·cta={bk}·{fk}·{br}·{cta} does not divide K=2048"
        assert 1 <= br <= 1024, f"cooperative BR out of range: {br}"


def test_coop_reduce_offers_fk_one_ranks_first_per_group():
    """The greedy cooperative default leads with FK=1 per (bk, br, cta) group, and cta=1 first."""
    assert coop_reduce_offers(_reduce_dag((128, 2048)))[0][3] == 1, "greedy default must lead with cta=1 (no split-K)"
    seen: set[tuple[int, int, int]] = set()
    for bk, fk, br, cta in coop_reduce_offers(_reduce_dag((128, 2048))):
        key = (bk, br, cta)
        if key in seen:
            continue
        seen.add(key)
        assert fk == 1, f"FK={fk} outranked FK=1 for (bk={bk}, br={br}, cta={cta})"
