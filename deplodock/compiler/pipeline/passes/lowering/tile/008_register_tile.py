"""Register-tile matmul-shaped reduce kernels (axis-aware).

Each thread in the post-blockify state owns one output element of the
CTA's M×N tile (PAT × PAT threads — default ``PAT=16`` so 256 threads,
1 output / thread). ``PAT`` (per-axis thread tile width) and ``F``
(per-thread output factor) are both supplied by ``tuning`` via
``detect_pat`` / ``register_tile_factor`` and are paired through the
``_PAT_TO_FACTOR`` table.

This pass splits each of the two ``BIND_THREAD`` axes ``a:PAT`` into
outer ``a_o:PAT/F`` (still ``BIND_THREAD``) plus a serial ``a_i:F``
dimension, and replicates the matmul reduce body + epilogue per
``(a_i, a_j)`` cell. Each replicated cell carries its own SSA
accumulator (``acc0_<i>_<j>``), giving F² independent partial sums per
thread that nvcc can schedule in parallel registers. With ``PAT=16``
and ``F=2`` this is per-thread output 4 (4× more FMAs per smem-load
round-trip) at 64 threads per CTA.

**Axis-aware replication.** Each stmt's value transitively depends on
some subset of {m_axis, n_axis}; the stmt is replicated F^|subset|
times with σ-substituted indices and SSA names suffixed by the cell
coordinate(s):

==============================  ============  =============================
Stmt's thread-axis dependence   Replicas      Per-cell name suffix
==============================  ============  =============================
``∅`` (constant / batch only)   1             ``""``  (shared across cells)
``{m_axis}`` (per-row)          F             ``"_<i>"``
``{n_axis}`` (per-col)          F             ``"_<j>"``
``{m_axis, n_axis}``            F²            ``"_<i>_<j>"``
==============================  ============  =============================

This is computed via :meth:`Body.fold` over the def-use DAG with a
bound-axis filter — see :func:`replicate_along_axis`.

**Stages stay singleton across F.** Because ``stage_inputs`` runs
*before* this pass, the body already references staged smem via
cache-local Loads. F-replication σ-substitutes the cache-axis Vars
inside those Loads (e.g. ``Var(n_axis) → Var(n_o)*F + i``) so the F²
output cells decode different cache-local offsets into the *same*
slab. The Stage stmt itself is held singleton: its cache-axis names
are excluded from its dependency contribution so the fold doesn't mark
it for replication. Only the consumer Loads multiply.

Idempotence: triggers only when exactly two ``BIND_THREAD`` axes have
an extent that's a key in ``_PAT_TO_FACTOR`` (currently
``{16: 2, 32: 4, 64: 8}``). After firing, the split THREAD axis has
extent ``pat/F`` — for every entry in the table that's exactly ``8``,
which is deliberately *not* a candidate. So ``detect_pat`` returns
``None`` on a second pass and the rule skips at its first gate.

Trigger conditions:

- ``TileOp.body`` contains exactly one ``Tile`` whose ``block_axes`` is empty.
- ``detect_pat`` matches a PAT, and ``register_tile_factor`` is
  ``F > 1`` with ``pat % F == 0``.
- The Tile has exactly two ``BIND_THREAD`` axes with extent equal to PAT.
- The Tile body contains a matmul-shape reduce Loop (profitability gate —
  without it F²-replicating gives no reuse benefit).
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Loop, Stmt, Tile
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce, single_tile
from deplodock.compiler.tuning import detect_pat, register_tile_factor

PATTERN = [Pattern("root", TileOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: Body) -> Body | None:
    idx, tile = single_tile(body)

    # ``detect_pat`` reads whichever PAT 005_blockify_launch landed on
    # by inspecting THREAD axis extents; ``register_tile_factor`` pairs
    # F with that PAT through the centralized table in ``tuning``.
    pat = detect_pat(tile)
    if pat is None:
        raise RuleSkipped("detect_pat returned None — Tile axes don't match a known PAT")
    factor = register_tile_factor(tile)
    if factor <= 1 or pat % factor != 0:
        raise RuleSkipped(f"register-tile factor F={factor} disabled or doesn't divide PAT={pat}")

    target_axes = [ba.axis.name for ba in tile.axes if ba.bind == BIND_THREAD and int(ba.axis.extent) == pat]
    if len(target_axes) != 2:
        raise RuleSkipped(f"need exactly 2 THREAD axes with extent {pat}, found {len(target_axes)}")

    # Profitability gate: register tiling pays off when ≥2 reduce-Loop bodies
    # share operand Loads across the M / N axes (matmul-shape reuse —
    # F² FMAs amortize each smem-load round-trip). Without that signature
    # the rewrite F²-replicates work for no reuse benefit; skip rather
    # than regress pointwise / reduce-only kernels.
    if not any(is_matmul_reduce(s) for s in tile.body.iter() if isinstance(s, Loop)):
        raise RuleSkipped("no matmul-shaped reduce in the Tile body — register tiling unprofitable")

    rewritten = _register_tile(tile, target_axes[0], target_axes[1], factor)
    if rewritten is None:
        raise RuleSkipped("_register_tile bailed (unsupported shape)")
    return body[:idx] + (rewritten,) + body[idx + 1 :]


def _split_axis(axes: tuple[BoundAxis, ...], target: str, factor: int) -> tuple[tuple[BoundAxis, ...], Axis]:
    """Replace ``BoundAxis(target:E, THREAD)`` with ``BoundAxis(target_o:E/F, THREAD)``.
    Returns (new_axes, outer_axis)."""
    new_axes: list[BoundAxis] = []
    outer: Axis | None = None
    for ba in axes:
        if ba.axis.name == target:
            ext = int(ba.axis.extent)
            outer = Axis(f"{target}_o", ext // factor)
            new_axes.append(BoundAxis(axis=outer, bind=BIND_THREAD))
        else:
            new_axes.append(ba)
    assert outer is not None
    return tuple(new_axes), outer


def _register_tile(tile: Tile, m_axis: str, n_axis: str, factor: int) -> Tile | None:
    """Register-tile by composing two per-axis F× replications.

    Each :func:`replicate_along_axis` pass walks the body, determines per
    stmt whether its value transitively depends on the pass's axis (via
    :meth:`Body.fold` over the def-use DAG with a bound-axis filter),
    emits ``factor`` σ-substituted copies for axis-dependent stmts, and
    leaves axis-independent stmts singleton. Composing the two passes
    gives F replicas for stmts depending on a single thread-axis and
    F² for stmts depending on both — same outcome as the previous
    one-shot ``CellRewriter`` flow, derived from per-stmt dependency
    analysis instead of a region-by-region cell-axes policy."""
    new_axes, m_o = _split_axis(tile.axes, m_axis, factor)
    new_axes, n_o = _split_axis(new_axes, n_axis, factor)

    body = replicate_along_axis(tile.body, m_axis, factor, m_o)
    body = replicate_along_axis(body, n_axis, factor, n_o)
    return Tile(axes=new_axes, body=body)


def replicate_along_axis(body: Body, axis: str, factor: int, axis_o: Axis) -> Body:
    """F× replicate every stmt whose value transitively depends on ``axis``.

    Each such stmt is emitted ``factor`` times with σ substituting
    ``axis → axis_o * factor + i`` and SSA names suffixed ``_<i>``.
    Stmts not depending on ``axis`` pass through unchanged. Block stmts
    (Loop / StridedLoop / Tile / Cond) recurse into their bodies and
    rebuild via :meth:`Stmt.with_bodies`; the wrapper itself isn't
    replicated even when it shadows ``axis`` (the fold's bound-axis
    filter keeps shadowed references from leaking into the outer
    dependency).

    The dependency analysis is one :meth:`Body.fold` over the def-use
    DAG with bound-axis filtering. ``keep[name]`` records whether each
    SSA name's defining stmt depends on ``axis`` so the rename closure
    knows which names must carry the suffix (locals defined in this
    region) vs. pass through unchanged (Tile-input buffer names,
    constants, axis-free producers)."""

    def fn(s: Stmt, child_T: tuple[frozenset[str] | None, ...], bound: frozenset[str]) -> frozenset[str]:
        # Stages own their cache axes — those Vars are smem-local
        # coordinates that materialization decodes from the cooperative
        # thread layout, not values that change per F² output cell. Treat
        # cache-axis names as ``bound``-like so the Stage doesn't get
        # marked as F-axis-dependent and replicated. Only its consumer
        # Loads (which σ-rewrite the cache-axis Var) multiply across F.
        local_bound = bound | frozenset(ax.name for ax in s.axes) if isinstance(s, Stage) else bound
        own: frozenset[str] = frozenset()
        for e in s.exprs():
            own = own | frozenset(v for v in e.free_vars() if v not in local_bound)
        for c in child_T:
            if c is not None:
                own = own | c
        return own

    deps = body.fold(fn)
    keep: dict[str, bool] = {n: axis in deps[id(s)] for s in body.iter() for n in s.defines()}

    def rename_for(i: int):
        def _rename(name: str) -> str:
            return f"{name}_{i}" if keep.get(name, False) else name

        return _rename

    def sigma_for(i: int) -> Sigma:
        return Sigma({axis: Var(axis_o.name) * Literal(factor, "int") + Literal(i, "int")})

    def go(b: Body) -> Body:
        out: list[Stmt] = []
        for s in b:
            nested = s.nested()
            if nested:
                out.append(s.with_bodies(tuple(go(child) for child in nested)))
            elif axis in deps.get(id(s), frozenset()):
                for i in range(factor):
                    out.append(s.rewrite(rename_for(i), sigma_for(i)))
            else:
                out.append(s)
        return Body(out)

    return go(body)
