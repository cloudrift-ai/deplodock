"""Rewrite the warp-tier matmul cell into tensor-core form — runs right after
``010_partition_loops``, before any staging pass.

``010_partition_loops`` emits the matmul cell as the scalar shape
``AtomTile > … > SerialTile(K_i, reduce) > [Load a, Load b, Assign(multiply),
Accum]`` (operands still gmem-direct — staging hasn't run yet). This pass
collapses the ``Assign(multiply) + Accum`` pair into a single :class:`Mma` op
(``c += a @ b``) that names its A (M×K) / B (K×N) operands by SSA value and
carries the atom kind + reduce axes. The ``Mma`` keeps the reduce loop
``is_reduce`` (so the staging gates still fire).

The two operand ``Load``s stay **plain** — no tensor-core tag. They flow
through the staging passes as ordinary ``Load``s, and the late
``kernel/005_lower_atom_tile`` pass recovers each one's A/B role from the
co-located ``Mma`` (which names them), so nothing needs to ride a side channel.
Everything else (the ``AtomTile`` wrapper, the K_o / K_i ``SerialTile``s, the
``Write``) is preserved structurally; ``080_pipeline_stages`` duplicates the
whole ``Load + Load + Mma`` cell into the prologue / epilogue via ``rewrite``.

Eligibility: an ``AtomTile`` in the body (its ``.atom`` is the spec used to
build the ``Mma`` — no ``ATOM_KIND`` knob lookup). Idempotence: after this pass
the cell holds an ``Mma`` (no ``Assign``/``Accum`` to collapse), so a second
visit rewrites nothing and the pass skips.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load, Mma, Stmt, Write
from deplodock.compiler.ir.tile.ir import Atom, AtomTile, SerialTile, TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._atom import classify_matmul_operands

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> Graph | None:
    op = root.op
    new_body, found = _tag_atom_tiles(op.body)
    if not found:
        raise RuleSkipped("no AtomTile matmul cell to tag (scalar, or already an Mma)")
    return TileOp(body=new_body, name=op.name, knobs=op.knobs)


def _tag_atom_tiles(body: Body) -> tuple[Body, bool]:
    """Walk ``body``; for each ``AtomTile``, tag the matmul cell inside it,
    using the ``Atom`` spec carried on the tile itself (no ``ATOM_KIND`` knob).
    Recurses into other block stmts so a deep-nested AtomTile is reached."""
    out: list[Stmt] = []
    found = False
    for s in body:
        if isinstance(s, AtomTile):
            new_cell_body, cell_found = _tag_cell(s.body, atom=s.atom, k_name=None, write=None)
            out.append(s.with_bodies((new_cell_body,)))
            found = found or cell_found
            continue
        if s.nested():
            new_bodies: list[Body] = []
            any_found = False
            for sub in s.nested():
                ns, sf = _tag_atom_tiles(sub)
                new_bodies.append(ns)
                any_found = any_found or sf
            out.append(s.with_bodies(tuple(new_bodies)))
            found = found or any_found
            continue
        out.append(s)
    return Body(out), found


def _tag_cell(body: Body, *, atom: Atom, k_name: str | None, write: Write | None) -> tuple[Body, bool]:
    """Recursively rewrite the AtomTile body. A reduce ``SerialTile`` body
    holding the canonical ``[Load, Load, Assign(multiply), Accum]`` cell
    (shapes A/B/D) becomes ``[Load a*, Load b*, Mma]``; a bare inline cell at
    the AtomTile-body level (shape C, K filtered) is tagged the same way using
    the ``Write``'s M/N free vars to classify. Returns ``(new_body, found)``."""
    # Track the enclosing Write so shape C can classify by output index.
    write = next((s for s in body if isinstance(s, Write)), write)

    tagged = _try_tag_here(body, atom=atom, k_name=k_name, write=write)
    if tagged is not None:
        return tagged, True

    out: list[Stmt] = []
    found = False
    for s in body:
        if isinstance(s, SerialTile) and s.is_reduce:
            new_inner, inner_found = _tag_cell(s.body, atom=atom, k_name=s.axis.name, write=write)
            out.append(s.with_bodies((new_inner,)))
            found = found or inner_found
            continue
        if s.nested():
            new_bodies = []
            any_found = False
            for sub in s.nested():
                ns, sf = _tag_cell(sub, atom=atom, k_name=k_name, write=write)
                new_bodies.append(ns)
                any_found = any_found or sf
            out.append(s.with_bodies(tuple(new_bodies)))
            found = found or any_found
            continue
        out.append(s)
    return Body(out), found


def _try_tag_here(body: Body, *, atom: Atom, k_name: str | None, write: Write | None) -> Body | None:
    """If ``body`` is the canonical matmul cell (2 Loads + one
    ``Assign(multiply)`` + one ``Accum``), return the rewritten cell — the two
    operand ``Load``s kept **plain**, the ``Assign + Accum`` replaced by one
    ``Mma`` that names its A/B operands by SSA value. Else ``None``."""
    loads = [s for s in body if isinstance(s, Load)]
    assigns = [s for s in body if isinstance(s, Assign)]
    accums = [s for s in body if isinstance(s, Accum)]
    if not (len(loads) == 2 and len(assigns) == 1 and len(accums) == 1):
        return None
    mul, accum = assigns[0], accums[0]
    if mul.op.name != "multiply" or accum.value != mul.name or set(mul.args) != {ld.names[0] for ld in loads if ld.names}:
        return None
    a_load, b_load = _classify_ab(loads, k_name=k_name, write=write)
    if a_load is None or b_load is None:
        return None
    # Transposed-B (Q @ K^T): B stored N×K (K in its last dim) is the native
    # mma.row.col col-major B — flag it so kernel/005 loads it without ``.trans``.
    b_trans = _is_transposed_b(b_load, k_name=k_name, write=write)
    # The Mma carries the Atom spec (taken straight off the AtomTile) + the A/B
    # operand identity (by SSA name); the operand Loads stay plain —
    # kernel/005_lower_atom_tile reads the spec off the Mma and recovers each
    # load's role from it. Emit the (plain) loads in A,B order then the Mma;
    # the multiply + Accum are dropped.
    mma = Mma(c=accum.name, a=a_load.names[0], b=b_load.names[0], atom=atom, axes=accum.axes, b_trans=b_trans)
    rest = [s for s in body if s not in (a_load, b_load, mul, accum)]
    return Body((*rest, a_load, b_load, mma))


def _classify_ab(loads: list[Load], *, k_name: str | None, write: Write | None) -> tuple[Load | None, Load | None]:
    """Identify A (M×K) vs B (K×N) by the reduce axis position (gmem-direct,
    pre-staging) via the shared :func:`classify_matmul_operands` — the same
    classifier the ``is_atom_eligible`` mma gate runs, so any cell that
    reaches this tagger is classifiable by construction. Shape C
    (``k_name is None``) falls back to the Write's M/N free vars."""
    a_load: Load | None = None
    b_load: Load | None = None
    if k_name is not None:
        out_index = write.index if (write is not None and write.index) else None
        return classify_matmul_operands(loads, k_name, out_index=out_index)
    if write is None or not write.index:
        return None, None
    w_m_vars = set(write.index[0].free_vars())
    w_n_vars = set(write.index[-1].free_vars())
    for ld in loads:
        ld_outer = set(ld.index[0].free_vars()) if ld.index else set()
        ld_inner = set(ld.index[-1].free_vars()) if ld.index else set()
        if w_m_vars & ld_outer and not (w_n_vars & ld_inner):
            a_load = ld
        elif w_n_vars & ld_inner and not (w_m_vars & ld_outer):
            b_load = ld
    return a_load, b_load


def _is_transposed_b(b_load: Load, *, k_name: str | None, write: Write | None) -> bool:
    """The B operand is transposed (stored N×K, K contiguous in its last dim) — a
    Q @ K^T cell — so kernel/005 reads it gmem-direct without ``.trans``. With a
    reduce axis: K is the contiguous (last) gmem dim AND nowhere else. The
    ``and-nowhere-else`` guard is load-bearing for a *collapsed-reshape* B (the
    SDPA P@V's ``V[seq, kv_head, head_dim]`` flattened to ``[seq, 1024]``): the
    delinearized index spreads the reduce var ``a10`` across BOTH the seq (K,
    outer) dim and the channel (N, contiguous-last) dim, so the bare ``k in
    last`` test wrongly flags canonical V as transposed and the consumer reads
    V with K/N swapped (garbage). K appearing in an earlier dim too ⇒ it's the
    outer (row) axis ⇒ canonical B. For the shape-C (filtered-K) fallback it's
    B carrying the N (col) output var in its FIRST dim."""
    if not b_load.index:
        return False
    if k_name is not None:
        in_last = k_name in b_load.index[-1].free_vars()
        in_earlier = any(k_name in e.free_vars() for e in b_load.index[:-1])
        return in_last and not in_earlier
    if write is None or not write.index:
        return False
    n_vars = set(write.index[-1].free_vars())
    return bool(n_vars & set(b_load.index[0].free_vars()))
