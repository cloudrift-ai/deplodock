"""Per-atom *eligibility* — the tensorize fork's gate for each matmul atom.

``plans/tile-ir-block-dag.md`` R4 (``atomize``): the warp-tier MMA fork
(``020_tensorize``) offers each :class:`~deplodock.compiler.ir.tile.ir.Atom`
the kernel admits. The eligibility predicate is the gate — a pure query over the
iteration DAG (the derived view) + operand dtypes + device compute capability:
does this ``LoopOp`` admit this atom?

It mirrors the legacy ``tile/_atom.py`` (deleted in the block-DAG demolition),
re-expressed against :class:`IterDag` instead of the raw ``LoopOp`` body: the
matmul reduces are ``dag.reduce`` nodes, the free extents come off
``dag.parallel``, and dtypes resolve through a ``dtype_of`` lookup (the seed
``Buffer``s) rather than ``graph.nodes``. :func:`classify_matmul_operands` is the
ONE A/B layout decision shared by the gate and the ``atomize`` body move, so a
cell the move can't classify is never offered the warp tier (an untagged
``AtomTile`` would survive unconsumed to render and crash).

Prefixed ``_`` so the pipeline rule loader skips it.
"""

from __future__ import annotations

from collections.abc import Callable

from deplodock.compiler.ir.stmt import Accum, Assign, Load, Write
from deplodock.compiler.ir.stmt.blocks import Body
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY, Atom
from deplodock.compiler.pipeline.passes.lowering._predicates import (
    classify_fragment_epilogue,
    is_matmul_reduce,
    segmentable_k_extent,
)
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import IterDag


def classify_matmul_operands(loads, k_name: str, *, out_index=None):
    """Identify the A (M×K) / B (K×N) operand ``Load``s of a canonical matmul
    cell by where the reduce axis ``k_name`` sits in each load's index.

    Primary tests: K in the LAST index dim (and not the first) ⇒ A; K in the
    FIRST dim (and not the last) ⇒ B. Fallback for K in a *middle* dim (the
    batched P@V split-consumer ``xnb[head, k, n]``): a load whose single K dim
    sits AFTER every other var-carrying dim ⇒ A, with exactly one var dim after
    K ⇒ B.

    **Transposed-B (Q @ K^T):** when BOTH operands carry K in their last dim the
    primary test tags both as A; with the output coordinates supplied via
    ``out_index`` we disambiguate by which output var each operand shares (M ⇒ A,
    N ⇒ B — the native ``mma.row.col`` layout). Without ``out_index`` it stays
    unclassified (returns ``None`` for B). Returns ``(a_load, b_load)``, either
    possibly ``None``."""
    a_load = None
    b_load = None
    k_last: list = []  # transposed-B candidates (K in the LAST dim, like A)
    for ld in loads:
        if not ld.index:
            continue
        k_in_first = k_name in ld.index[0].free_vars()
        k_in_last = k_name in ld.index[-1].free_vars()
        if k_in_last and not k_in_first:
            if a_load is None:
                a_load = ld
            k_last.append(ld)
            continue
        if k_in_first and not k_in_last:
            b_load = ld
            continue
        k_dims = [d for d, e in enumerate(ld.index) if k_name in e.free_vars()]
        var_dims = [d for d, e in enumerate(ld.index) if d not in k_dims and e.free_vars()]
        if len(k_dims) > 1:
            # Folded K across >1 dim. Segmentable (innermost ``% C``) ⇒ A.
            if segmentable_k_extent(ld, k_name) is not None and a_load is None:
                a_load = ld
            continue
        if len(k_dims) != 1 or not var_dims:
            continue
        after_k = [d for d in var_dims if d > k_dims[0]]
        if not after_k:
            a_load = ld
        elif len(after_k) == 1:
            b_load = ld
    if b_load is None and len(k_last) == 2 and out_index is not None:
        a2, b2 = _classify_transposed_b(k_last, k_name, out_index)
        if a2 is not None and b2 is not None:
            return a2, b2
    return a_load, b_load


def _classify_transposed_b(loads, k_name: str, out_index):
    """Split two K-in-last operands into (A, B) by which output coordinate each
    shares: A carries the M (row) output var, B carries N (col)."""
    out_vars = [e for e in out_index if e.free_vars()]
    if len(out_vars) < 2:
        return None, None
    m_vars = set(out_vars[-2].free_vars()) - {k_name}
    n_vars = set(out_vars[-1].free_vars()) - {k_name}
    a_load = None
    b_load = None
    for ld in loads:
        non_k: set = set()
        for d, e in enumerate(ld.index):
            if d == len(ld.index) - 1 and k_name in e.free_vars():
                continue  # the trailing K dim
            non_k |= set(e.free_vars())
        non_k.discard(k_name)
        if (non_k & m_vars) and not (non_k & n_vars):
            a_load = ld
        elif (non_k & n_vars) and not (non_k & m_vars):
            b_load = ld
    return a_load, b_load


def _matmul_out_index(dag: IterDag):
    """The matmul output ``Write``'s index (≥2 var-bearing dims) — the M / N
    coordinates the transposed-B A/B disambiguation reads."""
    for w in Body.coerce(dag.inner_body).iter_of_type(Write):
        if sum(1 for e in w.index if e.free_vars()) >= 2:
            return w.index
    return None


def _atom_eligible(atom: Atom, dag: IterDag, *, compute_capability: tuple[int, int], dtype_of: Callable[[str], object]) -> bool:
    """True iff ``dag`` admits ``atom`` on this device — the warp-tier gate.

    Checks (mirroring the legacy ``_mma_eligible_factory``): cc ≥ (8,0); ≥1
    matmul-reduce; K extent % cell_k == 0; each K-indexed operand a single-K-dim
    (or segmentable) gmem input of the atom's operand dtype; the canonical
    ``[Load,Load,Assign(mul),Accum]`` cell with ``mul.distributes_over(accum)``
    and A/B classifiable; a foldable pointwise epilogue; output extents % cell."""
    cell_m, cell_n, cell_k = atom.shape
    operand_dtype = atom.operand_dtype("a")
    if compute_capability < (8, 0):
        return False
    matmul_reduces = [n.loop for n in dag.reduce if is_matmul_reduce(n.loop)]
    if not matmul_reduces:
        return False
    out_index = _matmul_out_index(dag)
    produced = {w.output for w in Body.coerce(dag.inner_body).iter_of_type(Write)}
    accum_names: set[str] = set()
    for k_loop in matmul_reduces:
        k_name = k_loop.axis.name
        if k_loop.axis.extent.is_static and k_loop.axis.extent.as_static() % cell_k != 0:
            return False
        for load in Body.coerce(k_loop.body).iter_of_type(Load):
            k_dims = [d for d, e in enumerate(load.index) if k_name in e.free_vars()]
            if not k_dims:
                continue
            seg_c = segmentable_k_extent(load, k_name)
            segmentable = seg_c is not None and seg_c % cell_k == 0
            if (len(k_dims) > 1 and not segmentable) or load.input in produced:
                return False
            dt = dtype_of(load.input)
            if dt != operand_dtype:
                return False
        top_level = list(k_loop.body)
        loads = [s for s in top_level if isinstance(s, Load)]
        assigns = [s for s in top_level if isinstance(s, Assign)]
        accums = [s for s in top_level if isinstance(s, Accum)]
        if not (len(loads) == 2 and len(assigns) == 1 and len(accums) == 1):
            return False
        multiply, accum = assigns[0], accums[0]
        load_names = {ld.names[0] for ld in loads if ld.names}
        if set(multiply.args) != load_names or accum.value != multiply.name:
            return False
        if not multiply.op.distributes_over(accum.op):
            return False
        a_ld, b_ld = classify_matmul_operands(loads, k_name, out_index=out_index)
        if a_ld is None or b_ld is None:
            return False
        accum_names.add(accum.name)
    # Loop-invariant leaf Loads (a fused per-CTA / per-row scale / mask) the
    # frontend hoisted above the free chain land in ``dag.leading`` / ``dag.mid``,
    # not ``inner_body`` — pass them as ``outer_loads`` so the fold can resolve an
    # epilogue operand defined there (the causal-mask scale / fill), mirroring
    # ``kernel/005``'s ``_collect_outer_loads`` over the materialized tower.
    outer_loads = {ld.names[0]: ld for ld in (*dag.leading, *dag.mid) if isinstance(ld, Load) and ld.names}
    _slice, blocker = classify_fragment_epilogue(
        Body.coerce(dag.inner_body),
        accum_names,
        produced=produced,
        leaf_dtype=lambda buf: dt.name if (dt := dtype_of(buf)) is not None else None,
        outer_loads=outer_loads,
    )
    if blocker is not None:
        return False
    free_extents = [n.extent for n in dag.parallel if n.loop.axis.extent.is_static]
    cell_dims = (cell_m, cell_n)
    for ext, cell in zip(reversed(free_extents), reversed(cell_dims), strict=False):
        if ext > 1 and ext % cell != 0:
            return False
    return True


def eligible_atoms(dag: IterDag, *, compute_capability: tuple[int, int], dtype_of: Callable[[str], object]) -> list[Atom]:
    """The atoms ``dag`` admits, in ``ATOM_REGISTRY`` priority order (f16 first).
    Empty for a non-matmul / scalar-only kernel."""
    return [a for a in ATOM_REGISTRY.values() if _atom_eligible(a, dag, compute_capability=compute_capability, dtype_of=dtype_of)]
