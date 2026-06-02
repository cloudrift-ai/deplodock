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

Eligibility: ``op.knobs["ATOM_KIND"]`` set (warp-tier matmul rows only).
Idempotence: after this pass the cell holds an ``Mma`` (no ``Assign``/``Accum``
to collapse), so a second visit rewrites nothing and the pass skips.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load, Mma, Stmt, Write
from deplodock.compiler.ir.tile.ir import AtomTile, SerialTile, TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> Graph | None:
    op = root.op
    kind = op.knobs.get("ATOM_KIND")
    if not kind:
        raise RuleSkipped("not an MMA TileOp (no ATOM_KIND knob)")
    new_body, found = _tag_atom_tiles(op.body, kind=kind)
    if not found:
        raise RuleSkipped("no scalar matmul cell to tag (already an Mma, or none)")
    return TileOp(body=new_body, name=op.name, knobs=op.knobs)


def _tag_atom_tiles(body: Body, *, kind: str) -> tuple[Body, bool]:
    """Walk ``body``; for each ``AtomTile``, tag the matmul cell inside it.
    Recurses into other block stmts so a deep-nested AtomTile is reached."""
    out: list[Stmt] = []
    found = False
    for s in body:
        if isinstance(s, AtomTile):
            new_cell_body, cell_found = _tag_cell(s.body, kind=kind, k_name=None, write=None)
            out.append(s.with_bodies((new_cell_body,)))
            found = found or cell_found
            continue
        if s.nested():
            new_bodies: list[Body] = []
            any_found = False
            for sub in s.nested():
                ns, sf = _tag_atom_tiles(sub, kind=kind)
                new_bodies.append(ns)
                any_found = any_found or sf
            out.append(s.with_bodies(tuple(new_bodies)))
            found = found or any_found
            continue
        out.append(s)
    return Body(out), found


def _tag_cell(body: Body, *, kind: str, k_name: str | None, write: Write | None) -> tuple[Body, bool]:
    """Recursively rewrite the AtomTile body. A reduce ``SerialTile`` body
    holding the canonical ``[Load, Load, Assign(multiply), Accum]`` cell
    (shapes A/B/D) becomes ``[Load a*, Load b*, Mma]``; a bare inline cell at
    the AtomTile-body level (shape C, K filtered) is tagged the same way using
    the ``Write``'s M/N free vars to classify. Returns ``(new_body, found)``."""
    # Track the enclosing Write so shape C can classify by output index.
    write = next((s for s in body if isinstance(s, Write)), write)

    tagged = _try_tag_here(body, kind=kind, k_name=k_name, write=write)
    if tagged is not None:
        return tagged, True

    out: list[Stmt] = []
    found = False
    for s in body:
        if isinstance(s, SerialTile) and s.is_reduce:
            new_inner, inner_found = _tag_cell(s.body, kind=kind, k_name=s.axis.name, write=write)
            out.append(s.with_bodies((new_inner,)))
            found = found or inner_found
            continue
        if s.nested():
            new_bodies = []
            any_found = False
            for sub in s.nested():
                ns, sf = _tag_cell(sub, kind=kind, k_name=k_name, write=write)
                new_bodies.append(ns)
                any_found = any_found or sf
            out.append(s.with_bodies(tuple(new_bodies)))
            found = found or any_found
            continue
        out.append(s)
    return Body(out), found


def _try_tag_here(body: Body, *, kind: str, k_name: str | None, write: Write | None) -> Body | None:
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
    # The Mma carries the atom kind + the A/B operand identity (by SSA name);
    # the operand Loads stay plain — kernel/005_lower_atom_tile recovers each
    # one's role from this Mma. Emit the (plain) loads in A,B order then the
    # Mma; the multiply + Accum are dropped.
    mma = Mma(c=accum.name, a=a_load.names[0], b=b_load.names[0], atom=kind, axes=accum.axes)
    rest = [s for s in body if s not in (a_load, b_load, mul, accum)]
    return Body((*rest, a_load, b_load, mma))


def _classify_ab(loads: list[Load], *, k_name: str | None, write: Write | None) -> tuple[Load | None, Load | None]:
    """Identify A (M×K) vs B (K×N) by the reduce axis position (gmem-direct,
    pre-staging): K in the last index dim ⇒ A, K in the first ⇒ B. Shape C
    (``k_name is None``) falls back to the Write's M/N free vars."""
    a_load: Load | None = None
    b_load: Load | None = None
    if k_name is not None:
        for ld in loads:
            k_in_last = k_name in {v for e in ld.index[-1:] for v in e.free_vars()}
            k_in_first = k_name in {v for e in ld.index[:1] for v in e.free_vars()}
            if k_in_last and not k_in_first:
                a_load = ld
            elif k_in_first and not k_in_last:
                b_load = ld
        return a_load, b_load
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
