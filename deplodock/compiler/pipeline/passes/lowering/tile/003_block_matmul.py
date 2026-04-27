"""Block-tiled matmul strategy — splits M/N/K into outer/inner tiles
and reshapes the ``Tile`` so each CUDA block owns a BM_BLOCK·BN_BLOCK
output region walked cooperatively across threads, with the K reduction
chunked into BK-sized slices and an optional per-thread output sub-tile
(TM × TN cells per thread) materialized as register accumulators.

Mirrors the cooperative-reduce pattern from ``002_cooperative_reduce.py``:
the rule splits the M / N / K axes, builds the nested per-thread output
body, and (when TM == TN == 1) defers operand staging to
``004_stage_inputs.py``. Matmul-aware materialization
(``passes/lowering/kernel/001_materialize_tile.py``) consumes the
``BIND_THREAD`` output axes for the launch geometry.

Per-thread output sub-tiling. When ``BM_BLOCK = BM_TG * TM`` and
``BN_BLOCK = BN_TG * TN`` with TM, TN > 1, each thread accumulates a
TM·TN block of outputs. The rule unrolls the sub-tile at IR generation
time — TM·TN distinct ``Accum`` SSA names, TM·TN distinct ``Write``
stmts. The renderer treats each Accum as a register, so the per-thread
accumulator is a register tile; smem load count per FMA drops from 2 to
``2/(TM*TN)``. Sub-tile selection is adaptive: the rule picks the largest
``(TM, TN)`` from ``_SUBTILE_CHOICES`` that divides ``(M, N)``,
falling through to ``(1, 1)`` for the original one-output-per-thread
shape.

Operand staging is delegated to ``004_stage_inputs.py``, which is
affine-aware (sizes the cache as ``axis.extent * F`` for the
``axis*F + literal`` patterns sub-tiling emits) and layout-aware
(reorders cache axes so the warp-varying axis is innermost in smem,
avoiding bank conflicts on the hot read path).

Pre-rewrite (post ``lower_naive`` of fused matmul ``C = A @ B``)::

    Tile(axes=(m THREAD, n THREAD)):
      Loop(k):
        a = Load("A", (m, k))
        b = Load("B", (k, n))
        t = Assign(a * b)
        Accum(acc, op=add, value=t)
      Write("C", (m, n), acc)

Post-rewrite::

    Tile(axes=(m_o BLOCK, n_o BLOCK, m_i THREAD, n_i THREAD)):
      Loop(m_i):
        Loop(n_i):
          Loop(k_o):
            Loop(k_i):
              a = Load("A", (m_o*BM + m_i, k_o*BK + k_i))
              b = Load("B", (k_o*BK + k_i, n_o*BN + n_i))
              t = Assign(a * b)
              Accum(acc, op=add, value=t)
          Write("C", (m_o*BM + m_i, n_o*BN + n_i), acc)

No ``Combine`` — every output element is owned by exactly one thread.

Trigger conditions:

- ``TileOp.body`` contains exactly one ``Tile``.
- ``Tile.thread_axes`` is exactly 2D, ``block_axes`` is empty (idempotence).
- ``Tile.body`` is the canonical fused-matmul shape: one reduce ``Loop``
  (Load · Load · Assign-mul · Accum-add, with two distinct source bufs)
  followed by a ``Write`` over the two output axes.
- M, N, K extents are each divisible by the configured tile size.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import (
    BIND_BLOCK,
    BIND_THREAD,
    Axis,
    BoundAxis,
)
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Load, Loop, Stmt, Write
from deplodock.compiler.ir.tile.ir import Tile, TileOp
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", TileOp)]

# Thread-grid dimensions: ``BM_TG·BN_TG == BLOCK_SIZE`` (16·16 = 256).
# These are the launch-geometry axes — one thread per (m_i_tg, n_i_tg).
BM_TG = 16
BN_TG = 16
BK = 16

# Per-thread output sub-tile shapes to try, ordered by preference within
# the target block-count window. ``TM >= TN`` is preferred — microbenchmarks
# (scripts/sweep_subtile.py) on the 5090 show that tall tiles outperform
# wide tiles at equal block count and equal TM*TN (e.g. (4,2) beats (2,4)
# by 25% on the gate_proj shape). ``(1, 1)`` is the always-applicable
# fallback.
_SUBTILE_CHOICES = ((4, 2), (2, 1), (4, 4), (2, 2), (2, 4), (1, 2), (1, 1))

# Choices used when the M-side load has a non-affine index (collapsed
# reshape — typical of o_proj fused with attn-output). Such loads bypass
# the staging rule (``004_stage_inputs``), so each unstaged global read
# is reused only ``TN`` times; ``TN=1`` means no reuse and tanks perf.
# We bias toward ``TN >= 2`` and accept fewer blocks (each block has more
# work). Microbench: (4,2)@128 blocks beats (2,1)@512 by 12% on the
# attn-out → o_proj+residual shape.
_SUBTILE_CHOICES_UNSTAGED_M = ((4, 2), (2, 4), (4, 4), (2, 2), (1, 2), (2, 1), (1, 1))

# Sub-tiling reduces block count by TM*TN; we want enough blocks to keep
# the SMs busy but not so many that per-thread work drops below the point
# where staging overhead dominates. RTX 5090 has ~170 SMs; the sweep shows
# 2-4 blocks/SM (~340-700) is the sweet spot. Below ~256 blocks SMs sit
# idle; above ~700 the per-thread arithmetic gain is gone and we pay
# launch + smem-staging overhead.
_TARGET_MIN_BLOCKS = 340  # ~2 blocks per SM on a 5090
_TARGET_MAX_BLOCKS = 700  # ~4 blocks per SM on a 5090
_FALLBACK_MIN_BLOCKS = 170  # ~one wave; only used when target window is empty
# Unstaged-M kernels can run with as few as 128 blocks — each block does
# more work per output element since the M load isn't reused via smem.
_UNSTAGED_FALLBACK_MIN_BLOCKS = 128


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: tuple) -> tuple | None:
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        return None
    idx, tile = tiles[0]
    if tile.block_axes:
        return None  # idempotence — already block-bound
    new_tile = _rewrite_tile(tile)
    if new_tile is None:
        return None
    return body[:idx] + (new_tile,) + body[idx + 1 :]


def _rewrite_tile(tile: Tile) -> Tile | None:
    if len(tile.thread_axes) != 2:
        return None
    m, n = tile.thread_axes

    parsed = _match_matmul_body(tile.body, m_name=m.name, n_name=n.name)
    if parsed is None:
        return None
    prefix, reduce_loop, suffix, write = parsed
    k = reduce_loop.axis
    if k.extent % BK:
        return None

    extracted = _extract_inner(reduce_loop)
    if extracted is None:
        return None
    accum, raw_a, raw_b, mul = extracted
    # Reduce-loop loads with ``/`` or ``%`` (from collapsed-reshape
    # views) bypass the affine staging path in ``004_stage_inputs``.
    # `_extract_inner` returns the two Loads in body order, which is
    # arbitrary w.r.t. which one carries M vs N. Identify the M-load
    # (index has m.name but not n.name) and the N-load (vice versa).
    a_free = _index_free_vars_set(raw_a.index)
    b_free = _index_free_vars_set(raw_b.index)
    if m.name in a_free and n.name in b_free:
        load_a, load_b = raw_a, raw_b
    elif m.name in b_free and n.name in a_free:
        load_a, load_b = raw_b, raw_a
    else:
        return None

    # Sub-tile choice depends on whether the M-load will be staged: a
    # non-affine index (mod/div) means 004 will skip staging, so each
    # unstaged global read is reused only TN times — bias toward TN>=2.
    m_unstaged = _index_has_div_or_mod(load_a.index)
    subtile = _pick_subtile(m.extent, n.extent, m_unstaged=m_unstaged)
    if subtile is None:
        return None
    tm, tn = subtile
    bm_block = BM_TG * tm
    bn_block = BN_TG * tn
    # Update mul.args order to match (load_a, load_b) as well so the
    # downstream Assign(mul + tag) names line up.
    if set(mul.args) != {load_a.name, load_b.name}:
        return None
    mul = Assign(name=mul.name, op=mul.op, args=(load_a.name, load_b.name))

    # Two-level decomposition: M = m_o · BM_BLOCK + m_i_tg · TM + m_t.
    # ``m.split(factor)`` makes the inner axis have extent == factor, so
    # using it would give m_i_tg.extent == BM_BLOCK (e.g. 32), not BM_TG
    # (16) — and the thread decoder would launch BM_BLOCK·BN_BLOCK threads
    # per block, blowing past 1024. Construct the axes directly so
    # m_i_tg.extent == BM_TG and the m_t dim is unrolled into the body
    # via literals.
    m_o = Axis(f"{m.name}_o", m.extent // bm_block)
    m_i_tg = Axis(f"{m.name}_i_tg", BM_TG)
    n_o = Axis(f"{n.name}_o", n.extent // bn_block)
    n_i_tg = Axis(f"{n.name}_i_tg", BN_TG)
    k_o, k_i = k.split(BK)

    # Per (m_t, n_t) sub-tile cell: emit prefix stmts, then per-cell
    # reduce-loop body, then suffix stmts, then write. Each cell's SSA
    # names get a ``_m_t_n_t`` tag so prefix/suffix references to
    # downstream values (including ``acc0`` from the reduce) line up.
    # Prefix executes once per cell *before* the K_o loop; suffix and
    # write execute once per cell *after*. ``produced_names`` excludes
    # SSA names defined OUTSIDE the matched body (e.g. ``Load`` stmts
    # at TileOp body scope feeding the Tile) — those refs stay un-tagged.
    produced_names = _collect_produced(prefix, reduce_loop, suffix)
    unrolled_prefix: list[Stmt] = []
    inner_compute: list[Stmt] = []
    unrolled_suffix: list[Stmt] = []
    writes: list[Stmt] = []

    for m_t in range(tm):
        for n_t in range(tn):
            tag = f"_{m_t}_{n_t}"
            sigma = _subtile_sigma(m, n, k, m_o, m_i_tg, n_o, n_i_tg, k_o, k_i, bm_block, bn_block, tm, tn, m_t, n_t)
            rename = _make_renamer(tag, produced_names)

            for s in prefix:
                unrolled_prefix.append(s.rewrite(rename, sigma))

            la = Load(name=load_a.name + tag, input=load_a.input, index=tuple(sigma.apply(e) for e in load_a.index))
            lb = Load(name=load_b.name + tag, input=load_b.input, index=tuple(sigma.apply(e) for e in load_b.index))
            mul_t = Assign(name=mul.name + tag, op=mul.op, args=(la.name, lb.name))
            ac = Accum(name=accum.name + tag, op=accum.op, value=mul_t.name)
            inner_compute.extend([la, lb, mul_t, ac])

            for s in suffix:
                unrolled_suffix.append(s.rewrite(rename, sigma))

            writes.append(write.rewrite(rename, sigma))

    new_body: tuple[Stmt, ...] = (
        *unrolled_prefix,
        Loop(axis=k_o, body=(Loop(axis=k_i, body=tuple(inner_compute)),)),
        *unrolled_suffix,
        *writes,
    )

    new_axes = (
        BoundAxis(axis=m_i_tg, bind=BIND_THREAD),
        BoundAxis(axis=n_i_tg, bind=BIND_THREAD),
        BoundAxis(axis=m_o, bind=BIND_BLOCK),
        BoundAxis(axis=n_o, bind=BIND_BLOCK),
    )
    return Tile(axes=new_axes, body=new_body)


def _pick_subtile(m_extent: int, n_extent: int, m_unstaged: bool = False) -> tuple[int, int] | None:
    """Pick (TM, TN) targeting ~2-4 blocks per SM on a 5090, preferring
    ``TM >= TN`` (tall tiles) at equal block count.

    Three-pass selection:
    1. First choice in the [_TARGET_MIN_BLOCKS, _TARGET_MAX_BLOCKS] window
       (the order in ``_SUBTILE_CHOICES`` encodes the preference).
    2. Fallback to any choice with ``block_count >= _TARGET_MIN_BLOCKS``
       (over-decomposed but still valid).
    3. Fallback to ``block_count >= _FALLBACK_MIN_BLOCKS`` (one full wave)
       so small-M shapes like seq=32 still get a tile.

    When the M-side load has a non-affine index (``m_unstaged=True``) the
    staging rule will skip it; we use ``_SUBTILE_CHOICES_UNSTAGED_M`` to
    bias toward TN >= 2 (so each unstaged global load amortizes across
    multiple FMAs) and accept the lower fallback block count.
    """
    choices = _SUBTILE_CHOICES_UNSTAGED_M if m_unstaged else _SUBTILE_CHOICES
    fallback_min = _UNSTAGED_FALLBACK_MIN_BLOCKS if m_unstaged else _FALLBACK_MIN_BLOCKS

    def candidates() -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []
        for tm, tn in choices:
            bm = BM_TG * tm
            bn = BN_TG * tn
            if m_extent % bm or n_extent % bn:
                continue
            out.append((tm, tn, (m_extent // bm) * (n_extent // bn)))
        return out

    cands = candidates()
    if m_unstaged:
        # Without staging, each M-load is reused only TN times. Pick the
        # first choice (in priority order, TN>=2 favored) with enough
        # blocks for one wave — don't apply the upper window cap, since
        # over-decomposition (more blocks, smaller per-thread tile) is
        # exactly the mode that hurts unstaged kernels.
        for tm, tn, b in cands:
            if b >= fallback_min:
                return tm, tn
        return (1, 1) if any(c[:2] == (1, 1) for c in cands) else None
    for tm, tn, b in cands:
        if _TARGET_MIN_BLOCKS <= b <= _TARGET_MAX_BLOCKS:
            return tm, tn
    for tm, tn, b in cands:
        if b >= _TARGET_MIN_BLOCKS:
            return tm, tn
    for tm, tn, b in cands:
        if b >= fallback_min:
            return tm, tn
    return (1, 1) if any(c[:2] == (1, 1) for c in cands) else None


def _has_div_or_mod(e: Expr) -> bool:
    """Recursively detect ``/`` or ``%`` in an Expr — signature of a
    layout-transform like a collapsed-reshape view. Mirrors the same
    helper in ``004_stage_inputs.py`` since loads with this signature
    bypass the affine staging path."""
    if isinstance(e, BinaryExpr):
        if e.op in ("/", "%"):
            return True
        return _has_div_or_mod(e.left) or _has_div_or_mod(e.right)
    return False


def _index_has_div_or_mod(index) -> bool:
    return any(_has_div_or_mod(e) for e in index)


def _index_free_vars_set(index) -> set[str]:
    out: set[str] = set()
    for e in index:
        out |= e.free_vars()
    return out


def _make_renamer(tag: str, produced: set[str]):
    """SSA-rename function used by ``Stmt.rewrite``. Tags only names
    produced *inside* the matched body — refs to TileOp-scope SSA names
    (e.g. a scalar ``Load`` hoisted above the Tile) stay un-tagged so
    the per-cell unrolled body still resolves them correctly."""

    def rename(name: str) -> str:
        return name + tag if name in produced else name

    return rename


def _collect_produced(prefix: tuple, reduce_loop: Loop, suffix: tuple) -> set[str]:
    """Names produced by Load / Assign / Accum / Select inside the
    matched body. Used by the renamer to leave outer-scope refs alone."""
    out: set[str] = set()
    for s in (*prefix, *reduce_loop.body, *suffix):
        if hasattr(s, "name") and isinstance(s.name, str):
            out.add(s.name)
    return out


def _subtile_sigma(m, n, k, m_o, m_i_tg, n_o, n_i_tg, k_o, k_i, bm_block, bn_block, tm, tn, m_t, n_t) -> Sigma:
    """Per-(m_t, n_t) substitution: maps the original m/n/k axis Vars to
    the unrolled thread-and-sub-tile coordinate expression.

    Sub-tile cells are *interleaved across threads* (not contiguous):

      M-coord = m_o * BM_BLOCK + m_t * BM_TG + m_i_tg
      N-coord = n_o * BN_BLOCK + n_t * BN_TG + n_i_tg
      K-coord = k_o * BK + k_i

    Adjacent threads in a warp differ in the innermost thread axis
    (``n_i_tg``), giving smem stride 1 instead of TN — eliminates bank
    conflicts that the contiguous mapping (``n_i_tg * TN + n_t``)
    introduces. Total output cells covered are identical; only the
    thread→cell ownership pattern changes.
    """
    return Sigma(
        {
            m.name: Var(m_o.name) * Literal(bm_block, "int") + Literal(m_t, "int") * Literal(BM_TG, "int") + Var(m_i_tg.name),
            n.name: Var(n_o.name) * Literal(bn_block, "int") + Literal(n_t, "int") * Literal(BN_TG, "int") + Var(n_i_tg.name),
            k.name: Var(k_o.name) * Literal(BK, "int") + Var(k_i.name),
        }
    )


def _match_matmul_body(body: tuple, m_name: str, n_name: str) -> tuple[tuple, Loop, tuple, Write] | None:
    """Match ``(*prefix, reduce_loop, *suffix, write)`` where ``reduce_loop``
    is the canonical matmul reduce and ``write`` indexes both output axes.

    ``prefix`` runs once per output cell before the K reduce; typical use
    is matmul-with-activation-of-other-input (e.g. ``silu(gate) · up_proj``
    where the silu chain on ``gate`` is in prefix). ``suffix`` runs after
    the K reduce and may consume the accum (e.g. ``+ residual`` or
    ``* silu_value``). Both must be straight-line stmts — no nested
    Loops or Tiles. The simple GEMM case has empty prefix and suffix.
    """
    if not body or not isinstance(body[-1], Write):
        return None
    write = body[-1]
    var_names = {e.name for e in write.index if isinstance(e, Var)}
    if {m_name, n_name} - var_names:
        return None
    reduce_idxs = [i for i, s in enumerate(body[:-1]) if isinstance(s, Loop) and s.is_reduce]
    if len(reduce_idxs) != 1:
        return None
    rl_idx = reduce_idxs[0]
    reduce_loop = body[rl_idx]
    prefix = body[:rl_idx]
    suffix = body[rl_idx + 1 : -1]
    # Prefix and suffix must be plain straight-line stmts (Load / Assign /
    # Accum / Select). Any nested Loop / Tile aborts the match — those
    # need their own handling and the matmul rule isn't the place.
    for s in prefix + suffix:
        if isinstance(s, (Loop, Tile)):
            return None
    return prefix, reduce_loop, suffix, write


def _extract_inner(reduce_loop: Loop) -> tuple[Accum, Load, Load, Assign] | None:
    """Inner body must be exactly ``(Load, Load, Assign-mul, Accum-add)``
    with two distinct source buffers and the Assign feeding the Accum."""
    inner = reduce_loop.body
    if len(inner) != 4:
        return None
    load_a, load_b, mul, accum = inner
    if not (isinstance(load_a, Load) and isinstance(load_b, Load)):
        return None
    if load_a.input == load_b.input:
        return None
    if not isinstance(mul, Assign) or mul.op.name != "multiply":
        return None
    if set(mul.args) != {load_a.name, load_b.name}:
        return None
    if not isinstance(accum, Accum) or accum.op.name != "add":
        return None
    if accum.value != mul.name:
        return None
    return accum, load_a, load_b, mul


def _id(name: str) -> str:
    return name
