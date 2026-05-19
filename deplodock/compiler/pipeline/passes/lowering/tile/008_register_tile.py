"""Register-tile matmul-shaped reduce kernels (axis-aware).

Each thread in the post-blockify state owns one output element of the
CTA's M×N tile. ``tuning.register_tile_shape`` returns the per-thread
``(FM, FN)`` cell shape — currently fixed at ``(8, 4)`` for matmul
(paired with the ``005_blockify_launch`` ``(BN=128, BM=64)`` thread
tile), or ``(1, 1)`` to skip register tiling (small matmul or
non-matmul body).

This pass splits each ``BIND_THREAD`` axis by its factor and
replicates the matmul reduce body + epilogue per ``(a_i, a_j)`` cell.
Each replicated cell carries its own SSA accumulator
(``acc0_<i>_<j>``), giving ``FM × FN`` independent partial sums per
thread that nvcc can schedule in parallel registers.

**Axis-aware replication.** Each stmt's value transitively depends on
some subset of {m_axis, n_axis}; the stmt is replicated
``FM^|{m_axis}∩deps| × FN^|{n_axis}∩deps|`` times with σ-substituted
indices and SSA names suffixed by the cell coordinate(s):

==============================  =====================  =============================
Stmt's thread-axis dependence   Replicas               Per-cell name suffix
==============================  =====================  =============================
``∅`` (constant / batch only)   1                      ``""``  (shared across cells)
``{m_axis}`` (per-row)          ``FM``                ``"_<i>"``
``{n_axis}`` (per-col)          ``FN``                ``"_<j>"``
``{m_axis, n_axis}``            ``FM × FN``          ``"_<i>_<j>"``
==============================  =====================  =============================

Computed via :meth:`Body.fold` over the def-use DAG with a bound-axis
filter — see :func:`replicate_along_axis`.

**Stages stay singleton across F.** Because ``stage_inputs`` runs
*before* this pass, the body already references staged smem via
cache-local Loads. F-replication σ-substitutes the cache-axis Vars
inside those Loads so the per-thread cells decode different
cache-local offsets into the *same* slab. The Stage stmt itself is
held singleton: cache-axis names are excluded from its dependency
contribution so the fold doesn't mark it for replication.

The two THREAD axes are identified by sorting all ``BIND_THREAD`` axes
by extent — smaller is M, larger is N. Symmetric-mode tiles where both
extents match make this ordering arbitrary, but ``_register_tile`` is
symmetric in axis names when factors are equal, so it doesn't matter.

Idempotence: after firing, the split THREAD axis has extent
``post_split = orig_extent / F``. The default symmetric flow's PAT
table picks F such that ``post_split=8`` (not a known PAT), so a
second-pass ``register_tile_shape`` query produces ``F=2`` paired with
PAT=16, but the THREAD axes have extent 8 — divisibility check fails
and the pass skips. The asymmetric TMA flow leaves THREAD axes at
``(8, 32)`` after split — neither matches ``BM=64`` or ``BN=128``, so
``register_tile_shape``'s post-blockify PAT detection misses and falls
through to ``(F=2, F=2)`` which fails the same divisibility check.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Loop, Stmt, Tile
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce, single_tile
from deplodock.compiler.tuning import register_tile_shape

PATTERN = [Pattern("root", TileOp)]

# Candidate per-thread cell factors to fork over. Cell shape is
# ``(FM, FN)`` with each F dividing its corresponding THREAD axis
# extent.
_TUNE_F_CHOICES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
# Cap on per-thread output cells. FM × FN replicates the matmul reduce
# body that many times per thread; beyond this NVRTC compile time
# explodes on the unrolled body. ``TileOp.validate`` is the second-line
# filter (post-register-tile thread count); this is the first-line cap
# in the rule itself.
_MAX_CELLS_PER_THREAD = 128


def rewrite(root: Node) -> Graph | None | list[TileOp]:
    body = root.op.body
    idx, tile = single_tile(body)

    # Explicit idempotence: once we've register-tiled this op, don't
    # re-fork. Without this, ``F=1`` in one axis leaves that THREAD
    # extent unchanged so ``register_tile_shape``'s gate still passes,
    # and 008 cascades into nested forks across every subsequent rule
    # pass. The ``register_tile`` knob is stamped by ``_variants`` on
    # every emitted variant.
    if root.op.knobs.get("register_tile"):
        raise RuleSkipped("register-tile already applied (knobs['register_tile'] set)")

    # Profitability gate — register tiling only helps matmul-shape
    # bodies (≥2 reduce-Loops sharing operand Loads across M / N).
    if not any(is_matmul_reduce(s) for s in tile.body.iter() if isinstance(s, Loop)):
        raise RuleSkipped("no matmul-shaped reduce in the Tile body — register tiling unprofitable")

    # Heuristic says "don't bother" → don't fork. Keeps the small-matmul
    # path (BN, BM < 16) emitting one un-tiled kernel and avoids burning
    # autotune cycles on tiles where the heuristic's own ``(1, 1)`` reflects
    # a real profitability gate (e.g., post-blockify extents too small to
    # divide cleanly by F).
    heuristic = register_tile_shape(tile)
    if heuristic[0] <= 1 and heuristic[1] <= 1:
        raise RuleSkipped(f"heuristic register-tile factor {heuristic} disabled — skipping fork")

    # Identify M / N THREAD axes by sorting by extent (smaller = M).
    thread_axes = [ba for ba in tile.axes if ba.bind == BIND_THREAD]
    if len(thread_axes) != 2:
        raise RuleSkipped(f"register-tile needs exactly 2 THREAD axes, found {len(thread_axes)}")
    sorted_ba = sorted(thread_axes, key=lambda ba: int(ba.axis.extent))
    m_axis_name, n_axis_name = sorted_ba[0].axis.name, sorted_ba[1].axis.name
    m_ext, n_ext = int(sorted_ba[0].axis.extent), int(sorted_ba[1].axis.extent)

    variants = _variants(body, idx, tile, root.op.name, m_axis_name, n_axis_name, m_ext, n_ext, heuristic)
    if not variants:
        raise RuleSkipped("no viable (FM, FN) candidate")
    if len(variants) == 1:
        return variants[0]
    return variants


def _variants(
    body: Body,
    idx: int,
    tile: Tile,
    name: str,
    m_axis_name: str,
    n_axis_name: str,
    m_ext: int,
    n_ext: int,
    heuristic: tuple[int, int],
) -> list[TileOp]:
    """Enumerate viable ``(FM, FN)`` cell shapes. Heuristic shape goes
    first so deterministic compiles pick it; alternative shapes are
    appended in a fixed order so the trace's ``choice_idx`` is stable
    across runs."""
    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []

    def _add(shape: tuple[int, int], *, cap_cells: bool = True) -> None:
        f_m, f_n = shape
        if shape in seen:
            return
        if f_m <= 1 and f_n <= 1:
            return  # both ≤ 1 means no register tile — handled by the skip path
        if cap_cells and f_m * f_n > _MAX_CELLS_PER_THREAD:
            return  # too many per-thread cells; NVRTC compile bombs
        if m_ext % f_m != 0 or n_ext % f_n != 0:
            return
        seen.add(shape)
        ordered.append(shape)

    # Heuristic first — non-tune compiles pick option 0. Exempt from the
    # cell-count cap so class-tuned defaults like ``(FM=8, FN=8)`` keep
    # working; the cap only narrows the *autotune* alternatives. Search
    # re-ranks unvisited variants via ``TileOp.score`` so rule emission
    # order doesn't determine MCTS bootstrap order.
    _add((int(heuristic[0]), int(heuristic[1])), cap_cells=False)
    for f_m in _TUNE_F_CHOICES:
        for f_n in _TUNE_F_CHOICES:
            _add((f_m, f_n))

    variants: list[TileOp] = []
    for f_m, f_n in ordered:
        rewritten = _register_tile(tile, m_axis_name, n_axis_name, f_m, f_n)
        if rewritten is None:
            continue
        variants.append(
            TileOp(
                body=body[:idx] + (rewritten,) + body[idx + 1 :],
                name=name,
                knobs={"FM": f_m, "FN": f_n, "register_tile": True},
            )
        )
    return variants


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


def _register_tile(tile: Tile, m_axis: str, n_axis: str, factor_m: int, factor_n: int) -> Tile | None:
    """Register-tile by composing two per-axis replications, ``factor_m``
    on the M-axis and ``factor_n`` on the N-axis. Symmetric mode passes
    ``factor_m == factor_n``; asymmetric (cuBLAS-style) passes them
    independently (e.g. ``FM=8, FN=4``)."""
    new_axes, m_o = _split_axis(tile.axes, m_axis, factor_m)
    new_axes, n_o = _split_axis(new_axes, n_axis, factor_n)

    body = replicate_along_axis(tile.body, m_axis, factor_m, m_o)
    body = replicate_along_axis(body, n_axis, factor_n, n_o)
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
