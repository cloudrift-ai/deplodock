"""Body-level normalization passes.

Pure ``body → body`` transforms applied via :func:`normalize_body` from
``LoopOp.__post_init__`` and ``TileOp.__post_init__`` so every constructed
Op lands in canonical form. The passes operate on the shared Stmt
vocabulary (``Loop``, ``Load``, ``Assign``, ``Accum``, ``Select``,
``Write``) and recurse through every block-structured Stmt
(``Loop`` / ``StridedLoop`` / ``Tile`` / ``Cond``) so they apply uniformly
to Loop IR and Tile IR bodies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Expr, Interval, Literal, SimplifyCtx, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop, Tile
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.stmt.leaves import Accum, Assign, Load, Select, SelectBranch, Write

# ---------------------------------------------------------------------------
# Visitor helpers shared by every pass below
# ---------------------------------------------------------------------------


def _identity_rename(n: str) -> str:
    return n


def _make_axis_renamer(old: str, new: Axis) -> Callable[[Axis], Axis]:
    return lambda a: new if a.name == old else a


def normalize_body(stmts: Body, *, hoist: bool = True) -> Body:
    """Apply the structural and cosmetic normalization passes in order.

    Used by both ``LoopOp.__post_init__`` and ``TileOp.__post_init__`` so
    bodies — Loop-IR and Tile-IR — land in a canonical shape before
    validation.

    ``hoist=False`` skips :func:`hoist_loop_invariants`. TileOp bodies turn
    it off because a Stage binding is scoped to the Loop where it's
    declared — hoisting Loads that read from a staged buffer above the
    Stage decl would leave the read referencing an undeclared name."""
    stmts = drop_size_one_free_axes(stmts)
    stmts = canonicalize_free_axis_order(stmts)
    stmts = eliminate_copy_aliases(stmts)
    stmts = unify_sibling_reduce_axes(stmts)
    if hoist:
        stmts = hoist_loop_invariants(stmts)
    stmts = simplify_body(stmts)
    stmts = rename_ssa_sequential(stmts)
    return stmts


# ---------------------------------------------------------------------------
# Pass 1: drop size-1 free axes
# ---------------------------------------------------------------------------


def drop_size_one_free_axes(stmts: Body) -> Body:
    """Inline every free ``Loop(axis, extent=1)``: replace it with its body
    after substituting ``Var(axis.name) → Literal(0, "int")``. Reduce Loops
    keep their wrappers because dropping them would remove the accumulator.
    Recurses through StridedLoop / Tile / Cond bodies without rewriting
    those wrappers (their iteration semantics aren't a free Loop)."""
    stmts = Body.coerce(stmts)

    def fn(s: Stmt) -> Stmt | Body:
        # Body.map post-order: ``s.body`` is already recursively mapped.
        if isinstance(s, Loop) and int(s.axis.extent) == 1 and not s.is_reduce:
            sub = Sigma({s.axis.name: Literal(0, "int")})
            return tuple(c.rewrite(_identity_rename, sub) for c in s.body)
        return s

    return stmts.map(fn)


# ---------------------------------------------------------------------------
# Pass 2: canonical free-axis ordering
# ---------------------------------------------------------------------------


def _recurse_canonicalize(s: Stmt) -> Stmt:
    if isinstance(s, Loop):
        return replace(s, body=canonicalize_free_axis_order(s.body))
    if isinstance(s, StridedLoop):
        return replace(s, body=canonicalize_free_axis_order(s.body))
    if isinstance(s, Tile):
        return Tile(axes=s.axes, body=canonicalize_free_axis_order(s.body))
    if isinstance(s, Cond):
        return Cond(
            cond=s.cond,
            body=canonicalize_free_axis_order(s.body),
            else_body=canonicalize_free_axis_order(s.else_body),
        )
    return s


def canonicalize_free_axis_order(stmts: Body) -> Body:
    """Sort the outer chain of free ``Loop`` blocks alphabetically by axis
    name. The chain is the sequence of single-child free Loops at the top of
    ``stmts``; it terminates at a reduce Loop or a branching body. Recursion
    continues into terminal block bodies (Loop / StridedLoop / Tile / Cond)."""
    stmts = Body.coerce(stmts)
    chain_axes: list[Axis] = []
    current = stmts
    while len(current) == 1 and isinstance(current[0], Loop):
        loop = current[0]
        if loop.is_reduce:
            break
        chain_axes.append(loop.axis)
        current = loop.body

    terminal = tuple(_recurse_canonicalize(s) for s in current)

    chain_axes_sorted = sorted(chain_axes, key=lambda a: a.name)
    result: Body = terminal
    for axis in reversed(chain_axes_sorted):
        result = (Loop(axis=axis, body=result),)
    return result


# ---------------------------------------------------------------------------
# Pass 3: eliminate `y = copy(x)` identity aliases
# ---------------------------------------------------------------------------


def eliminate_copy_aliases(stmts: Body) -> Body:
    """Collapse ``y = copy(x)`` Assigns. The merge rule plants identity
    copies as bridges between producer writes and consumer reads; a long
    chain stacks them. Every such Assign is dropped and downstream
    references to ``y`` are rewired to the alias root. Pure IR hygiene."""
    stmts = Body.coerce(stmts)
    alias: dict[str, str] = {}

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    def fn(s: Stmt) -> Stmt | None:
        # Body.map post-order: block bodies already recursed; only handle leaves.
        if isinstance(s, (Loop, StridedLoop, Tile, Cond)):
            return s
        if isinstance(s, Assign) and s.op.name == "copy" and len(s.args) == 1:
            alias[s.name] = s.args[0]
            return None
        return s.rewrite(resolve)

    return stmts.map(fn)


# ---------------------------------------------------------------------------
# Pass 4: unify sibling reduce-loop axis names
# ---------------------------------------------------------------------------


def unify_sibling_reduce_axes(stmts: Body) -> Body:
    """At every scope, find sibling reduce ``Loop``s whose reduce axes index
    the same ``(Load.source, dim)`` position and rename them to a single
    canonical axis name. Recurses through every block-structured Stmt
    (Loop / StridedLoop / Tile / Cond) to find nested scopes."""
    stmts = Body.coerce(stmts)

    def walk(body: Body) -> Body:
        new_body: list[Stmt] = []
        for s in body:
            if isinstance(s, Loop):
                new_body.append(replace(s, body=walk(s.body)))
            elif isinstance(s, StridedLoop):
                new_body.append(replace(s, body=walk(s.body)))
            elif isinstance(s, Tile):
                new_body.append(Tile(axes=s.axes, body=walk(s.body)))
            elif isinstance(s, Cond):
                new_body.append(Cond(cond=s.cond, body=walk(s.body), else_body=walk(s.else_body)))
            else:
                new_body.append(s)

        groups: dict[frozenset[tuple[str, int]], list[int]] = {}
        for i, s in enumerate(new_body):
            if isinstance(s, Loop) and s.is_reduce:
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
                if int(loop.axis.extent) != canonical_extent or loop.axis.name == canonical:
                    continue
                new_axis = Axis(name=canonical, extent=canonical_extent)
                sub = Sigma({loop.axis.name: Var(canonical)})
                rename_axis = _make_axis_renamer(loop.axis.name, new_axis)
                renamed = tuple(s.rewrite(_identity_rename, sub, rename_axis) for s in loop.body)
                new_body[idx] = replace(loop, axis=new_axis, body=renamed)

        return tuple(new_body)

    return walk(stmts)


def _reduce_axis_source_positions(body: Body, reduce_axis_name: str) -> set[tuple[str, int]]:
    """Collect ``(source, dim)`` positions where ``Var(reduce_axis_name)``
    appears bare in a Load index within ``body`` (recursing into nested
    blocks)."""
    return {
        (s.input, dim)
        for s in body.iter()
        if isinstance(s, Load)
        for dim, e in enumerate(s.index)
        if isinstance(e, Var) and e.name == reduce_axis_name
    }


# ---------------------------------------------------------------------------
# Pass 5: loop-invariant code motion
# ---------------------------------------------------------------------------


def hoist_loop_invariants(stmts: Body) -> Body:
    """Move ``Load`` / ``Assign`` / ``Select`` stmts out of ``Loop``s whose
    axis their value doesn't depend on. Hoisting only crosses ``Loop``
    boundaries; ``StridedLoop`` / ``Tile`` / ``Cond`` are barriers but are
    recursed into so inner Loops still get the optimization."""
    stmts = Body.coerce(stmts)
    axes_of: dict[str, frozenset[str]] = {}
    ssa_names: set[str] = set()

    def _load_index_parts(index: tuple[Expr, ...]) -> tuple[set[str], set[str]]:
        deps: set[str] = set()
        axes: set[str] = set()
        for e in index:
            for v in e.free_vars():
                (deps if v in ssa_names else axes).add(v)
        return deps, axes

    def _stmt_live(s: Stmt) -> frozenset[str]:
        if isinstance(s, Load):
            deps, axes = _load_index_parts(s.index)
            result = set(axes)
            for d in deps:
                result |= axes_of.get(d, frozenset())
            return frozenset(result)
        if isinstance(s, Assign):
            result = set()
            for name in s.args:
                result |= axes_of.get(name, frozenset())
            return frozenset(result)
        if isinstance(s, Select):
            result = set()
            for b in s.branches:
                result |= b.select.free_vars()
                result |= axes_of.get(b.value, frozenset())
            return frozenset(result)
        return frozenset()

    def _deps(s: Stmt) -> tuple[str, ...]:
        if isinstance(s, Load):
            deps, _ = _load_index_parts(s.index)
            return tuple(deps)
        if isinstance(s, Assign):
            return s.args
        if isinstance(s, Select):
            return tuple(b.value for b in s.branches)
        return ()

    def _record(c: Stmt, bindings: set[str]) -> None:
        defined = c.defines()
        if not defined:
            return
        name = defined[0]
        if isinstance(c, Accum):
            axes_of[name] = axes_of.get(c.value, frozenset())
        else:
            axes_of[name] = _stmt_live(c)
        ssa_names.add(name)
        bindings.add(name)

    def walk(body: Body, outer_bindings: set[str]) -> list[Stmt]:
        new_body: list[Stmt] = []
        bindings = set(outer_bindings)

        for s in body:
            if isinstance(s, Loop):
                inner = walk(s.body, bindings)
                axis = s.axis.name
                hoisted_names: set[str] = set()
                hoisted: list[Stmt] = []
                stay: list[Stmt] = []
                for c in inner:
                    if isinstance(c, (Load, Assign, Select)):
                        live = axes_of.get(c.name, frozenset())
                        if axis not in live and all(d in bindings or d in hoisted_names for d in _deps(c)):
                            hoisted.append(c)
                            hoisted_names.add(c.name)
                            continue
                    stay.append(c)

                for h in hoisted:
                    new_body.append(h)
                    bindings.add(h.name)
                new_body.append(replace(s, body=tuple(stay)))

                for c in inner:
                    if isinstance(c, Accum):
                        axes_of[c.name] = axes_of.get(c.value, frozenset()) - {axis}
                        ssa_names.add(c.name)
                        bindings.add(c.name)
            elif isinstance(s, StridedLoop):
                inner = walk(s.body, bindings)
                new_body.append(replace(s, body=tuple(inner)))
                for c in inner:
                    if isinstance(c, Accum):
                        ssa_names.add(c.name)
                        bindings.add(c.name)
            elif isinstance(s, Tile):
                inner = walk(s.body, bindings)
                new_body.append(Tile(axes=s.axes, body=tuple(inner)))
            elif isinstance(s, Cond):
                inner_b = walk(s.body, bindings)
                inner_e = walk(s.else_body, bindings)
                new_body.append(Cond(cond=s.cond, body=tuple(inner_b), else_body=tuple(inner_e)))
            else:
                new_body.append(s)
                _record(s, bindings)

        return new_body

    return tuple(walk(stmts, set()))


# ---------------------------------------------------------------------------
# Pass 6: simplify Exprs inside body Stmts (constant folding, identity collapse,
# range-based comparison folding). The per-Expr rewrite logic lives on each
# ``Expr`` subclass as ``simplify(ctx)``; this pass just walks the body and
# threads ``SimplifyCtx`` ranges as it descends.
# ---------------------------------------------------------------------------


def _simplify_expr_tuple(xs: tuple[Expr, ...], ctx: SimplifyCtx) -> tuple[Expr, ...]:
    return tuple(e.simplify(ctx) for e in xs)


def _simplify_stmt(stmt: Stmt, ctx: SimplifyCtx) -> Stmt:
    if isinstance(stmt, Loop):
        inner = ctx.extend(stmt.axis.name, Interval(0, stmt.axis.extent - 1))
        return replace(stmt, body=tuple(_simplify_stmt(s, inner) for s in stmt.body))
    if isinstance(stmt, StridedLoop):
        inner = ctx.extend(stmt.axis.name, Interval(0, stmt.axis.extent - 1))
        return replace(
            stmt,
            start=stmt.start.simplify(ctx),
            step=stmt.step.simplify(ctx) if isinstance(stmt.step, Expr) else stmt.step,
            body=tuple(_simplify_stmt(s, inner) for s in stmt.body),
        )
    if isinstance(stmt, Tile):
        inner = ctx
        for ba in stmt.axes:
            inner = inner.extend(ba.axis.name, Interval(0, ba.axis.extent - 1))
        return Tile(axes=stmt.axes, body=tuple(_simplify_stmt(s, inner) for s in stmt.body))
    if isinstance(stmt, Cond):
        return Cond(
            cond=stmt.cond.simplify(ctx),
            body=tuple(_simplify_stmt(s, ctx) for s in stmt.body),
            else_body=tuple(_simplify_stmt(s, ctx) for s in stmt.else_body),
        )
    if isinstance(stmt, Select):
        return Select(stmt.name, tuple(SelectBranch(b.value, b.select.simplify(ctx)) for b in stmt.branches))
    if isinstance(stmt, Write):
        return Write(stmt.output, _simplify_expr_tuple(stmt.index, ctx), stmt.value)
    if isinstance(stmt, Load):
        return Load(stmt.name, stmt.input, _simplify_expr_tuple(stmt.index, ctx))
    # Tile-IR-only ``Stage`` carries Exprs in ``origin`` / ``source_index_template``.
    # Lazy import to avoid a tile→stmt circular at module load time.
    from deplodock.compiler.ir.tile.ir import Stage  # noqa: PLC0415

    if isinstance(stmt, Stage):
        new_template = _simplify_expr_tuple(stmt.source_index_template, ctx) if stmt.source_index_template is not None else None
        return Stage(
            name=stmt.name,
            buf=stmt.buf,
            origin=_simplify_expr_tuple(stmt.origin, ctx),
            axes=stmt.axes,
            slab_dims=stmt.slab_dims,
            source_index_template=new_template,
            pad=stmt.pad,
            buffer_count=stmt.buffer_count,
            phase=stmt.phase.simplify(ctx) if stmt.phase is not None else None,
            async_load=stmt.async_load,
            pipelined=stmt.pipelined,
        )
    # Assign / Accum / Combine carry only SSA names — no Expr field to simplify.
    return stmt


def simplify_body(body: Body) -> Body:
    """Simplify every Expr inside a body. Seeds ``SimplifyCtx`` from
    ``Loop`` / ``StridedLoop`` / ``Tile`` axis extents as the walker descends."""
    body = Body.coerce(body)
    ctx = SimplifyCtx.empty()
    return tuple(_simplify_stmt(s, ctx) for s in body)


# ---------------------------------------------------------------------------
# Pass: deduplicate Load stmts with identical (input, index)
# ---------------------------------------------------------------------------


def dedup_loads(stmts: Body) -> Body:
    """Drop duplicate ``Load`` stmts within nested scopes.

    Two ``Load`` stmts with the same ``(input, index)`` read the same
    value; keep the first and rewire downstream SSA references to its
    name. Operates per-scope: a Load at an outer scope is reused by
    inner siblings (their identical ``index`` doesn't reference any
    inner-axis Var, so the values are equal). Loads inside a nested
    scope are not visible to outer / sibling scopes."""
    stmts = Body.coerce(stmts)

    def walk(
        body: Body,
        env: dict[tuple[str, tuple[str, ...]], str],
        parent_alias: dict[str, str],
    ) -> Body:
        local = dict(env)
        alias = dict(parent_alias)

        def rename(n: str) -> str:
            return alias.get(n, n)

        out: list[Stmt] = []
        for s in body:
            if isinstance(s, Load):
                key = (s.input, tuple(e.pretty() for e in s.index))
                if key in local:
                    alias[s.name] = local[key]
                    continue
                local[key] = s.name
                out.append(s)
            elif isinstance(s, Loop):
                out.append(replace(s, body=walk(s.body, local, alias)))
            elif isinstance(s, StridedLoop):
                out.append(replace(s, body=walk(s.body, local, alias)))
            elif isinstance(s, Tile):
                out.append(Tile(axes=s.axes, body=walk(s.body, local, alias)))
            elif isinstance(s, Cond):
                out.append(Cond(cond=s.cond, body=walk(s.body, local, alias), else_body=walk(s.else_body, local, alias)))
            else:
                out.append(s.rewrite(rename))
        return tuple(out)

    return walk(stmts, {}, {})


# ---------------------------------------------------------------------------
# Pass 7: canonicalize SSA names to sequential v0, v1, ...
# ---------------------------------------------------------------------------


def rename_ssa_sequential(stmts: Body) -> Body:
    """Canonicalize names in a fused body:

    - Axes from every axis-bearing scope (``Loop`` / ``StridedLoop`` /
      ``Tile.axes``) renamed to ``a0, a1, ...`` in pre-order of first
      declaration. All three share one numbering namespace so Tile.axes
      ``a0_o`` and a Loop axis ``a2_o`` don't collide on rename.
    - Load SSA names renamed to ``in0, in1, ...`` in definition order.
    - Accum names renamed to ``acc0, acc1, ...`` in definition order.
    - Assign / Select SSA names renamed to ``v0, v1, ...`` in definition
      order.

    Idempotent: bodies already in canonical form round-trip unchanged."""
    stmts = Body.coerce(stmts)
    ssa_rename: dict[str, str] = {}
    axis_rename: dict[str, str] = {}
    expr_sub: dict[str, Expr] = {}
    counters = {"v": 0, "in": 0, "acc": 0}

    def _rename(name: str, prefix: str) -> str:
        new = f"{prefix}{counters[prefix]}"
        ssa_rename[name] = new
        counters[prefix] += 1
        return new

    def _record_axis(name: str) -> None:
        if name in axis_rename:
            return
        new = f"a{len(axis_rename)}"
        axis_rename[name] = new
        if name != new:
            expr_sub[name] = Var(new)

    for stmt in stmts.iter():
        if isinstance(stmt, Load) and stmt.name not in ssa_rename:
            new = _rename(stmt.name, "in")
            if stmt.name != new:
                expr_sub[stmt.name] = Var(new)
        elif isinstance(stmt, Accum) and stmt.name not in ssa_rename:
            _rename(stmt.name, "acc")
        elif isinstance(stmt, (Assign, Select)) and stmt.name not in ssa_rename:
            _rename(stmt.name, "v")
        elif isinstance(stmt, Tile):
            for ba in stmt.axes:
                _record_axis(ba.axis.name)
        elif isinstance(stmt, (Loop, StridedLoop)):
            _record_axis(stmt.axis.name)

    if all(o == n for o, n in ssa_rename.items()) and all(o == n for o, n in axis_rename.items()):
        return stmts

    sigma = Sigma(expr_sub)

    def rename_ssa(name: str) -> str:
        return ssa_rename.get(name, name)

    def axis_fn(a: Axis) -> Axis:
        new = axis_rename.get(a.name, a.name)
        return Axis(name=new, extent=a.extent) if new != a.name else a

    return tuple(s.rewrite(rename_ssa, sigma, axis_fn) for s in stmts)
