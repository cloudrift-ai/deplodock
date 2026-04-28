"""Register-tile matmul-shaped reduce kernels.

Each thread in the post-blockify state owns one output element of the
CTA's M×N tile (16×16 = 256 threads, 1 output / thread). This pass
splits each ``BIND_THREAD`` axis ``a:16`` into outer ``a_o:16/F`` (still
``BIND_THREAD``) plus a serial ``a_i:F`` dimension, and replicates the
matmul reduce body + epilogue per ``(a_i, a_j)`` cell. Each replicated
cell carries its own SSA accumulator (``acc0_<i>_<j>``), giving F²
independent partial sums per thread that nvcc can schedule in parallel
registers. With F=2 this is per-thread output 4 (4× more FMAs per
smem-load round-trip) at 64 threads per CTA.

Idempotence: triggers only when both THREAD axes have extent equal to
``_PER_AXIS_THREADS = 16`` (the post-blockify size). After firing,
extents are ``16/F``, so the rule won't re-match.

The replicated cells use distinct SSA names so the resulting kernel
needs no nvcc-side scalarization — the per-cell accumulator chains are
already independent at the IR level. ``place_inits`` (kernel pass) emits
one ``Init`` per cell at the right scope (Tile body head, since the
free K-outer loop is reduce-passthrough).

Trigger conditions:

- ``TileOp.body`` contains exactly one ``Tile``.
- The Tile has 2+ ``BIND_THREAD`` axes, of which exactly two have
  extent ``_PER_AXIS_THREADS``.
- The Tile body has a top-level free ``Loop`` (the K-outer chunk loop)
  whose body contains a single reduce ``Loop`` (the K-inner reduce)
  with at least two distinct buffer Loads — i.e. a matmul shape.
- The chosen factor ``F`` divides ``_PER_AXIS_THREADS``.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Init, Load, Loop, Select, Stmt, Tile, iter_body
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", TileOp)]

_PER_AXIS_THREADS = 16  # must match ``005_blockify_launch._PER_AXIS_THREADS``
_FACTOR = 2  # per-thread tile = F × F


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: tuple[Stmt, ...]) -> tuple[Stmt, ...] | None:
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        return None
    idx, tile = tiles[0]

    target_axes = [ba.axis.name for ba in tile.axes if ba.bind == BIND_THREAD and int(ba.axis.extent) == _PER_AXIS_THREADS]
    if len(target_axes) != 2:
        return None
    if _PER_AXIS_THREADS % _FACTOR != 0:
        return None

    matmul_loc = _find_matmul(tile.body)
    if matmul_loc is None:
        return None

    rewritten = _register_tile(tile, target_axes[0], target_axes[1], _FACTOR)
    if rewritten is None:
        return None
    return body[:idx] + (rewritten,) + body[idx + 1 :]


def _find_matmul(body: tuple[Stmt, ...]) -> int | None:
    """Return the index of a top-level free Loop that wraps a *pure* matmul
    reduce, or None. Pure means:

    - the K-outer's body has exactly one reduce Loop (the K-inner);
    - the reduce body has Loads of 2+ distinct buffers indexed by the
      reduce axis;
    - the reduce body has exactly one Accum;
    - no stmt in the reduce body reads an Accum target (excludes online-
      softmax fusions where the running max / sum is folded back in).
    """
    for i, s in enumerate(body):
        if not (isinstance(s, Loop) and not s.is_reduce):
            continue
        reduces = [c for c in s.loops if c.is_reduce]
        if len(reduces) != 1:
            continue
        rl = reduces[0]
        bufs = {ld.input for ld in rl.loads if rl.axis.name in {v for e in ld.index for v in e.free_vars()}}
        if len(bufs) < 2:
            continue
        if sum(1 for c in rl.body if isinstance(c, Accum)) != 1:
            continue
        # Pure matmul shape only: Load / Assign / Accum stmts allowed.
        # Reject Select / Cond / nested Loops / staged-buffer rereads etc.
        if not all(isinstance(c, (Load, Assign, Accum)) for c in rl.body):
            continue
        # Reject if any SSA name read by the body is not defined locally —
        # that signals a cross-loop dependency (e.g. SDPA online softmax
        # reads running max / sum produced by prior reduce loops). Such
        # accumulators are per-M-row, but our replication multiplies the
        # M coordinate, so every cell would need its own max/sum — which
        # the prior loops haven't computed.
        local_defs = {c.name for c in rl.body if isinstance(c, (Load, Assign, Accum))}
        if any(d not in local_defs for c in rl.body for d in c.deps()):
            continue
        return i
    return None


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
    new_axes, m_o = _split_axis(tile.axes, m_axis, factor)
    new_axes, n_o = _split_axis(new_axes, n_axis, factor)

    k_outer_idx = _find_matmul(tile.body)
    if k_outer_idx is None:
        return None
    k_outer = tile.body[k_outer_idx]

    cells = [(i, j) for i in range(factor) for j in range(factor)]

    # K-outer body: stages stay CTA-scoped; inner reduce body replicates per cell.
    new_outer_body: list[Stmt] = []
    k_inner: Loop | None = None
    for s in k_outer.body:
        if isinstance(s, Stage):
            new_outer_body.append(s)
        elif isinstance(s, Loop) and s.is_reduce:
            k_inner = s
        else:
            return None  # unsupported shape — bail rather than corrupt
    assert k_inner is not None

    # The replicated region spans pre-K-outer stmts (the Tile-body
    # preamble) + K-inner body + post-K-outer epilogue. Preamble stmts
    # that reference the thread axes (e.g. ``v4 = silu(linear_4[a1, a3])``
    # in a SiLU+matmul kernel) MUST be replicated per cell with σ so each
    # cell sees its own (a1*F+i, a3*F+j) coords. Replicating preamble
    # stmts that don't depend on thread axes is wasted but harmless —
    # nvcc CSEs the resulting redundant scalar loads.
    pre_outer = tile.body[:k_outer_idx]
    post_outer = tile.body[k_outer_idx + 1 :]
    replicated = (*pre_outer, *k_inner.body, *post_outer)
    local_ssa = _collect_ssa_defs(replicated)

    new_k_inner_body: list[Stmt] = []
    for i, j in cells:
        sigma = _cell_sigma(m_axis, m_o, i, n_axis, n_o, j, factor)
        rename = _cell_rename(i, j, local_ssa)
        for s in k_inner.body:
            new_k_inner_body.append(s.rewrite(rename, sigma))
    new_outer_body.append(Loop(axis=k_inner.axis, body=tuple(new_k_inner_body), unroll=k_inner.unroll))

    new_body: list[Stmt] = []
    # Preamble: replicate per cell so any thread-axis-indexed Loads or
    # Assigns produce per-cell values matching the epilogue's writes.
    for i, j in cells:
        sigma = _cell_sigma(m_axis, m_o, i, n_axis, n_o, j, factor)
        rename = _cell_rename(i, j, local_ssa)
        for s in pre_outer:
            new_body.append(s.rewrite(rename, sigma))

    new_body.append(Loop(axis=k_outer.axis, body=tuple(new_outer_body), unroll=k_outer.unroll))

    # Epilogue: replicate post-K-outer stmts per cell.
    for i, j in cells:
        sigma = _cell_sigma(m_axis, m_o, i, n_axis, n_o, j, factor)
        rename = _cell_rename(i, j, local_ssa)
        for s in post_outer:
            new_body.append(s.rewrite(rename, sigma))

    return Tile(axes=new_axes, body=tuple(new_body))


def _collect_ssa_defs(stmts: tuple[Stmt, ...]) -> set[str]:
    """SSA names bound somewhere inside ``stmts``. Used to limit cell
    renaming to locally-defined names so external references stay intact."""
    out: set[str] = set()
    for s in iter_body(stmts):
        if isinstance(s, (Load, Assign, Accum, Select, Init)):
            out.add(s.name)
    return out


def _cell_sigma(m_axis: str, m_o: Axis, i: int, n_axis: str, n_o: Axis, j: int, factor: int) -> Sigma:
    return Sigma(
        {
            m_axis: Var(m_o.name) * Literal(factor, "int") + Literal(i, "int"),
            n_axis: Var(n_o.name) * Literal(factor, "int") + Literal(j, "int"),
        }
    )


def _cell_rename(i: int, j: int, locals_set: set[str]):
    suffix = f"_{i}_{j}"

    def rename(name: str) -> str:
        return f"{name}{suffix}" if name in locals_set else name

    return rename
