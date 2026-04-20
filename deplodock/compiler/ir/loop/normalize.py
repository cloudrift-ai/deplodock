"""Structural normalization passes for ``LoopOp`` bodies.

Invoked from ``LoopOp.__post_init__`` so every constructed ``LoopOp`` —
including intermediate results produced by fusion rules — lands in a
canonical shape before validation. Pure ``body → body`` transforms,
applied in order by :func:`normalize_body`:

- :func:`drop_size_one_free_axes` — inline ``Loop(axis, extent=1)`` free
  loops, rewriting ``Var(axis.name)`` to ``Literal(0, "int")`` throughout
  the body. Reduce loops (bodies with ``Accum``) are left alone; dropping
  them would strip the accumulator.
- :func:`canonicalize_free_axis_order` — sort the outer chain of free
  ``Loop`` blocks alphabetically by axis name. Free loops commute, so
  reordering is safe; the chain terminates at a reduce Loop or a
  branching body.
- :func:`eliminate_copy_aliases` — treat ``y = copy(x)`` Assigns as
  aliases, rewire downstream references to the alias root, and drop the
  copies. Merge chains leave stacks of identity copies; this collapses
  them so the IR prints / compares cleanly.
- :func:`unify_sibling_reduce_axes` — at every scope, find sibling reduce
  ``Loop``s whose reduce axes index the same ``(source, dim)`` position and
  rename them to a single canonical axis name. Softmax's max-over-K /
  sum-over-K end up sharing one reduce axis; likewise
  ``sum(x, -1) + max(x, -1)`` after fusion.
- :func:`rename_ssa_sequential` — after the other passes settle, rename
  Assign / Select SSA names to ``v0``, ``v1``, ... in definition order.
  Accum and Load names are preserved (Accum names carry semantic roles;
  Load names are structural binding sites).
"""

from __future__ import annotations

from deplodock.compiler.ir.expr import Expr, Literal, Var, substitute
from deplodock.compiler.ir.loop.ir import (
    Accum,
    Assign,
    Axis,
    Load,
    Loop,
    Select,
    SelectBranch,
    Stmt,
    Write,
    flatten_body,
)
from deplodock.compiler.ir.tensor_ir import ElementwiseOp

__all__ = [
    "normalize_body",
    "drop_size_one_free_axes",
    "canonicalize_free_axis_order",
    "eliminate_copy_aliases",
    "unify_sibling_reduce_axes",
    "rename_ssa_sequential",
]


# ---------------------------------------------------------------------------
# Top-level composition
# ---------------------------------------------------------------------------


def normalize_body(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Apply the structural and cosmetic normalization passes in order."""
    stmts = drop_size_one_free_axes(stmts)
    stmts = canonicalize_free_axis_order(stmts)
    stmts = eliminate_copy_aliases(stmts)
    stmts = unify_sibling_reduce_axes(stmts)
    stmts = rename_ssa_sequential(stmts)
    return stmts


# ---------------------------------------------------------------------------
# Body predicates
# ---------------------------------------------------------------------------


def _immediate_has_accum(stmts: tuple[Stmt, ...]) -> bool:
    """True when the immediate sequence contains an ``Accum`` — marks the
    enclosing Loop as a reduce loop."""
    return any(isinstance(s, Accum) for s in stmts)


# ---------------------------------------------------------------------------
# Pass 1: drop size-1 free axes
# ---------------------------------------------------------------------------


def _substitute_vars_in_body(stmts: tuple[Stmt, ...], mapping: dict[str, Expr]) -> tuple[Stmt, ...]:
    """Rewrite ``Var(name)`` occurrences in every Expr subtree of ``stmts``."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop):
            out.append(Loop(axis=s.axis, body=_substitute_vars_in_body(s.body, mapping)))
        elif isinstance(s, Load):
            out.append(Load(name=s.name, source=s.source, index=tuple(substitute(e, mapping) for e in s.index)))
        elif isinstance(s, Write):
            out.append(Write(output=s.output, index=tuple(substitute(e, mapping) for e in s.index), value=s.value))
        elif isinstance(s, Select):
            out.append(
                Select(
                    name=s.name,
                    branches=tuple(SelectBranch(value=b.value, select=substitute(b.select, mapping)) for b in s.branches),
                )
            )
        else:
            out.append(s)
    return tuple(out)


def drop_size_one_free_axes(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Inline every free ``Loop(axis, extent=1)``: replace it with its body
    after substituting ``Var(axis.name) → Literal(0, "int")``. Reduce Loops
    keep their wrappers because dropping them would remove the accumulator."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop):
            inner = drop_size_one_free_axes(s.body)
            is_reduce = _immediate_has_accum(inner)
            if int(s.axis.extent) == 1 and not is_reduce:
                sub = {s.axis.name: Literal(0, "int")}
                out.extend(_substitute_vars_in_body(inner, sub))
            else:
                out.append(Loop(axis=s.axis, body=inner))
        else:
            out.append(s)
    return tuple(out)


# ---------------------------------------------------------------------------
# Pass 2: canonical free-axis ordering
# ---------------------------------------------------------------------------


def canonicalize_free_axis_order(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Sort the outer chain of free ``Loop`` blocks alphabetically by axis
    name. The chain is the sequence of single-child free Loops at the top of
    ``stmts``; it terminates at a reduce Loop or a branching body. Recursion
    continues into that terminal body."""
    chain_axes: list[Axis] = []
    current = stmts
    while len(current) == 1 and isinstance(current[0], Loop):
        loop = current[0]
        if _immediate_has_accum(loop.body):
            break
        chain_axes.append(loop.axis)
        current = loop.body

    terminal = tuple(Loop(axis=s.axis, body=canonicalize_free_axis_order(s.body)) if isinstance(s, Loop) else s for s in current)

    chain_axes_sorted = sorted(chain_axes, key=lambda a: a.name)
    result: tuple[Stmt, ...] = terminal
    for axis in reversed(chain_axes_sorted):
        result = (Loop(axis=axis, body=result),)
    return result


# ---------------------------------------------------------------------------
# Pass 3: eliminate `y = copy(x)` identity aliases
# ---------------------------------------------------------------------------


def _is_copy_assign(stmt: Stmt) -> bool:
    return isinstance(stmt, Assign) and isinstance(stmt.op, ElementwiseOp) and stmt.op.fn == "copy" and len(stmt.args) == 1


def eliminate_copy_aliases(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Collapse ``y = copy(x)`` Assigns. The merge rule plants identity
    copies as bridges between producer writes and consumer reads; a long
    chain stacks them (``y = copy(copy(copy(x)))``). Every such Assign is
    dropped and downstream references to ``y`` are rewired to the alias
    root. Semantically a no-op — pure IR hygiene."""
    alias: dict[str, str] = {}

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    def walk(body: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
        out: list[Stmt] = []
        for stmt in body:
            if isinstance(stmt, Loop):
                out.append(Loop(axis=stmt.axis, body=walk(stmt.body)))
            elif _is_copy_assign(stmt):
                alias[stmt.name] = stmt.args[0]
            elif isinstance(stmt, Assign):
                out.append(Assign(name=stmt.name, op=stmt.op, args=tuple(resolve(a) for a in stmt.args)))
            elif isinstance(stmt, Accum):
                out.append(Accum(name=stmt.name, value=resolve(stmt.value), op=stmt.op))
            elif isinstance(stmt, Write):
                out.append(Write(output=stmt.output, index=stmt.index, value=resolve(stmt.value)))
            elif isinstance(stmt, Select):
                out.append(
                    Select(
                        name=stmt.name,
                        branches=tuple(SelectBranch(value=resolve(br.value), select=br.select) for br in stmt.branches),
                    )
                )
            else:
                out.append(stmt)
        return tuple(out)

    return walk(stmts)


# ---------------------------------------------------------------------------
# Pass 5: unify sibling reduce-loop axis names
# ---------------------------------------------------------------------------


def unify_sibling_reduce_axes(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """At every scope, find sibling reduce ``Loop``s whose reduce axes index
    the same ``(Load.source, dim)`` position and rename them to a single
    canonical axis name.

    Fusion commonly plants sibling reduces that sweep the same buffer at the
    same dim — softmax's max-over-K followed by sum-over-K, or
    ``sum(x, -1) + max(x, -1)`` after fusion. Sharing an axis name makes
    ``LoopOp.axes`` (which dedupes by name) report one reduce axis rather
    than N, matching the schedule the emitter would otherwise produce.

    Self-contained within a single body: two Loads with the same ``source``
    index are provably the same external buffer, so no caller-supplied
    input-name list is needed. Idempotent."""

    def walk(body: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
        # Recurse into nested scopes first.
        new_body: list[Stmt] = []
        for s in body:
            if isinstance(s, Loop):
                new_body.append(Loop(axis=s.axis, body=walk(s.body)))
            else:
                new_body.append(s)

        # Group sibling reduce Loops by their reduce axis's (source, dim) pattern.
        groups: dict[frozenset[tuple[int, int]], list[int]] = {}
        for i, s in enumerate(new_body):
            if isinstance(s, Loop) and _immediate_has_accum(s.body):
                positions = _reduce_axis_source_positions(s.body, s.axis.name)
                if positions:
                    groups.setdefault(frozenset(positions), []).append(i)

        for indices in groups.values():
            if len(indices) < 2:
                continue
            first = new_body[indices[0]]
            assert isinstance(first, Loop)
            canonical = first.axis.name
            canonical_extent = int(first.axis.extent)
            for idx in indices[1:]:
                loop = new_body[idx]
                assert isinstance(loop, Loop)
                if int(loop.axis.extent) != canonical_extent:
                    continue
                if loop.axis.name == canonical:
                    continue
                renamed = _rename_axis_in_body(loop.body, loop.axis.name, canonical)
                new_body[idx] = Loop(axis=Axis(name=canonical, extent=canonical_extent), body=renamed)

        return tuple(new_body)

    return walk(stmts)


def _reduce_axis_source_positions(body: tuple[Stmt, ...], reduce_axis_name: str) -> set[tuple[int, int]]:
    """Collect ``(source, dim)`` positions where ``Var(reduce_axis_name)``
    appears bare in a Load index within ``body`` (recursing into nested
    Loops — important when the reduce body contains a nested free loop
    that does the actual load)."""
    positions: set[tuple[int, int]] = set()

    def go(stmts: tuple[Stmt, ...]) -> None:
        for s in stmts:
            if isinstance(s, Load):
                for dim, e in enumerate(s.index):
                    if isinstance(e, Var) and e.name == reduce_axis_name:
                        positions.add((s.source, dim))
            elif isinstance(s, Loop):
                go(s.body)

    go(body)
    return positions


def _rename_axis_in_body(body: tuple[Stmt, ...], old_name: str, new_name: str) -> tuple[Stmt, ...]:
    """Substitute ``Var(old_name) → Var(new_name)`` in every Expr field of
    ``body``, and rename any nested Loop whose axis name is ``old_name``."""
    mapping: dict[str, Expr] = {old_name: Var(new_name)}
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, Loop):
            inner = _rename_axis_in_body(s.body, old_name, new_name)
            if s.axis.name == old_name:
                out.append(Loop(axis=Axis(name=new_name, extent=s.axis.extent), body=inner))
            else:
                out.append(Loop(axis=s.axis, body=inner))
        elif isinstance(s, Load):
            out.append(Load(name=s.name, source=s.source, index=tuple(substitute(e, mapping) for e in s.index)))
        elif isinstance(s, Write):
            out.append(Write(output=s.output, index=tuple(substitute(e, mapping) for e in s.index), value=s.value))
        elif isinstance(s, Select):
            out.append(
                Select(
                    name=s.name,
                    branches=tuple(SelectBranch(value=br.value, select=substitute(br.select, mapping)) for br in s.branches),
                )
            )
        else:
            out.append(s)
    return tuple(out)


# ---------------------------------------------------------------------------
# Pass 6: canonicalize SSA names to sequential v0, v1, ...
# ---------------------------------------------------------------------------


def rename_ssa_sequential(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Canonicalize names in a fused body:

    - Axes renamed to ``a0, a1, ...`` in pre-order of first declaration.
      Sibling ``Loop``s with the same axis name (post-unification reduce
      axes) share one renumbering entry.
    - Load SSA names renamed to ``in0, in1, ...`` in definition order.
    - Accum names renamed to ``acc0, acc1, ...`` in definition order.
    - Assign / Select SSA names renamed to ``v0, v1, ...`` in definition
      order.

    Idempotent: bodies already in canonical form round-trip unchanged."""
    ssa_rename: dict[str, str] = {}
    v_counter = 0
    in_counter = 0
    acc_counter = 0
    for stmt in flatten_body(stmts):
        if isinstance(stmt, Load):
            if stmt.name in ssa_rename:
                continue
            ssa_rename[stmt.name] = f"in{in_counter}"
            in_counter += 1
        elif isinstance(stmt, Accum):
            if stmt.name in ssa_rename:
                continue
            ssa_rename[stmt.name] = f"acc{acc_counter}"
            acc_counter += 1
        elif isinstance(stmt, (Assign, Select)):
            if stmt.name in ssa_rename:
                continue
            ssa_rename[stmt.name] = f"v{v_counter}"
            v_counter += 1

    axis_rename: dict[str, str] = {}
    a_counter = 0

    def collect_axes(body: tuple[Stmt, ...]) -> None:
        nonlocal a_counter
        for s in body:
            if isinstance(s, Loop):
                if s.axis.name not in axis_rename:
                    axis_rename[s.axis.name] = f"a{a_counter}"
                    a_counter += 1
                collect_axes(s.body)

    collect_axes(stmts)

    ssa_noop = all(old == new for old, new in ssa_rename.items())
    axis_noop = all(old == new for old, new in axis_rename.items())
    if ssa_noop and axis_noop:
        return stmts

    # Expr-level substitution covers axis renames AND Load renames — a Load's
    # SSA name can appear as ``Var(load_name)`` in another Load's index (the
    # gather pattern: ``data[..., Var("idx"), ...]`` where ``idx`` is another
    # Load's name). Assign/Select names never surface in Exprs so they stay
    # out of this map.
    expr_sub: dict[str, Expr] = {old: Var(new) for old, new in axis_rename.items() if old != new}
    for stmt in flatten_body(stmts):
        if isinstance(stmt, Load):
            old = stmt.name
            new = ssa_rename.get(old, old)
            if old != new:
                expr_sub[old] = Var(new)

    def rn(name: str) -> str:
        return ssa_rename.get(name, name)

    def walk(body: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
        out: list[Stmt] = []
        for stmt in body:
            if isinstance(stmt, Loop):
                new_axis_name = axis_rename.get(stmt.axis.name, stmt.axis.name)
                out.append(Loop(axis=Axis(name=new_axis_name, extent=stmt.axis.extent), body=walk(stmt.body)))
            elif isinstance(stmt, Load):
                out.append(
                    Load(
                        name=rn(stmt.name),
                        source=stmt.source,
                        index=tuple(substitute(e, expr_sub) for e in stmt.index),
                    )
                )
            elif isinstance(stmt, Assign):
                out.append(Assign(name=rn(stmt.name), op=stmt.op, args=tuple(rn(a) for a in stmt.args)))
            elif isinstance(stmt, Accum):
                out.append(Accum(name=rn(stmt.name), value=rn(stmt.value), op=stmt.op))
            elif isinstance(stmt, Write):
                out.append(
                    Write(
                        output=stmt.output,
                        index=tuple(substitute(e, expr_sub) for e in stmt.index),
                        value=rn(stmt.value),
                    )
                )
            elif isinstance(stmt, Select):
                out.append(
                    Select(
                        name=rn(stmt.name),
                        branches=tuple(SelectBranch(value=rn(br.value), select=substitute(br.select, expr_sub)) for br in stmt.branches),
                    )
                )
            else:
                out.append(stmt)
        return tuple(out)

    return walk(stmts)
