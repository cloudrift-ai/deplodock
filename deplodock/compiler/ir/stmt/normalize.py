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
from deplodock.compiler.ir.stmt.leaves import Accum, Assign, Init, Load, Select, SelectBranch, Write

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
        stmts = split_invariant_divides(stmts)
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
    nested = s.nested()
    if not nested:
        return s
    return s.with_bodies(tuple(canonicalize_free_axis_order(b) for b in nested))


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
        # Recurse into nested bodies first (post-order) via the canonical
        # nested() / with_bodies() descent, then group siblings at this
        # scope. Splitting the recursion from the sibling-grouping keeps
        # this pass's scope-level logic isolated in ``_unify_siblings``.
        recursed: list[Stmt] = []
        for s in body:
            nested = s.nested()
            if nested:
                recursed.append(s.with_bodies(tuple(walk(b) for b in nested)))
            else:
                recursed.append(s)
        return _unify_siblings(Body(recursed))

    return walk(stmts)


def _unify_siblings(body: Body) -> Body:
    """Single-scope sibling grouping: rename reduce-axis vars across
    sibling reduce Loops that index the same ``(source, dim)`` positions
    so they share one canonical axis name."""
    stmts = list(body)
    groups: dict[frozenset[tuple[str, int]], list[int]] = {}
    for i, s in enumerate(stmts):
        if isinstance(s, Loop) and s.is_reduce:
            positions = _reduce_axis_source_positions(s.body, s.axis.name)
            if positions:
                groups.setdefault(frozenset(positions), []).append(i)

    for indices in groups.values():
        if len(indices) < 2:
            continue
        first = stmts[indices[0]]
        assert isinstance(first, Loop)
        canonical = first.axis.name
        canonical_extent = int(first.axis.extent)
        for idx in indices[1:]:
            loop = stmts[idx]
            assert isinstance(loop, Loop)
            if int(loop.axis.extent) != canonical_extent or loop.axis.name == canonical:
                continue
            new_axis = Axis(name=canonical, extent=canonical_extent)
            sub = Sigma({loop.axis.name: Var(canonical)})
            rename_axis = _make_axis_renamer(loop.axis.name, new_axis)
            renamed = tuple(s.rewrite(_identity_rename, sub, rename_axis) for s in loop.body)
            stmts[idx] = replace(loop, axis=new_axis, body=renamed)

    return Body(stmts)


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
# Pass 5a: split loop-invariant divides into reciprocal + multiply.
# ---------------------------------------------------------------------------
#
# ``divide(x, y)`` lowers to a single-precision divide on the XU pipe (the
# same pipe ``exp`` uses). When ``y`` is loop-invariant w.r.t. some
# enclosing Loop and ``x`` is not, the divide can't hoist as-is — its live
# set is the union of x's and y's. Splitting into::
#
#     recip_y = reciprocal(y)        # live = axes_of(y)
#     result  = multiply(x, recip_y) # live = axes_of(x) ∪ {recip_y}
#
# lets the next pass (``hoist_loop_invariants``) move ``recip_y`` out of
# every Loop axis that doesn't appear in ``y``. Inside the loop the
# divide turns into a multiply (FMA pipe), which is typically the
# under-utilized pipe on transcendental-heavy kernels (softmax,
# RMSNorm, attention output). One XU op per outer-axis iteration
# instead of one per inner-axis iteration.
#
# Gate: split iff ``axes_of(y)`` is a strict subset of ``axes_of(x)``.
# That's the precise structural condition for "splitting unblocks at
# least one Loop's worth of hoisting." Skip when y has axes x doesn't
# (no hoisting wins) or when both have identical axes (rcp would stay
# in the same scope as the original divide, no win and slight
# precision drift). When y is a true scalar (axes_of empty), the rcp
# hoists all the way to body root.
# ---------------------------------------------------------------------------


def split_invariant_divides(stmts: Body) -> Body:
    """Rewrite ``divide(x, y)`` → ``reciprocal(y) + multiply(x, recip)``
    when ``y``'s axis-dependency set is a strict subset of ``x``'s.

    Invariance is queried via :meth:`Body.deps_closure` over the
    pre-rewrite body, filtered to axis names. The strict-subset check
    means there's at least one axis ``x`` depends on that ``y``
    doesn't — splitting moves the rcp out of that axis's Loop while
    the multiply stays. Generates fresh SSA names for the rcp; the
    trailing :func:`rename_ssa_sequential` pass renumbers them into
    ``vN`` form.
    """
    from deplodock.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415

    stmts = Body.coerce(stmts)
    closure = stmts.deps_closure
    axes = stmts.axis_names
    ssa_names: set[str] = set(closure.keys())
    fresh_counter = [0]

    def _fresh(prefix: str) -> str:
        while True:
            fresh_counter[0] += 1
            n = f"{prefix}_{fresh_counter[0]}"
            if n not in ssa_names:
                ssa_names.add(n)
                return n

    def _axes_of(name: str) -> frozenset[str]:
        return closure.get(name, frozenset()) & axes

    def walk(body: Body) -> Body:
        out: list[Stmt] = []
        for s in body:
            nested = s.nested()
            if nested:
                # Generic descent — recurse into every nested body, rebuild
                # the wrapper via with_bodies. The closure was built once
                # over the whole body, so post-Loop Accum bookkeeping is
                # already baked in — no per-wrapper update needed here.
                out.append(s.with_bodies(tuple(walk(b) for b in nested)))
                continue
            if isinstance(s, Assign) and s.op == ElementwiseImpl("divide") and len(s.args) == 2:
                x_name, y_name = s.args
                if _axes_of(y_name) < _axes_of(x_name):  # strict subset → splitting unblocks at least one hoist
                    recip_name = _fresh(f"recip_{y_name}")
                    recip = Assign(name=recip_name, op=ElementwiseImpl("reciprocal"), args=(y_name,))
                    mult = Assign(name=s.name, op=ElementwiseImpl("multiply"), args=(x_name, recip_name))
                    # Patch closure for the freshly-introduced rcp so a
                    # later divide reading the same y in the same body
                    # still sees the correct axis set.
                    closure[recip_name] = closure.get(y_name, frozenset())
                    closure[mult.name] = closure.get(x_name, frozenset()) | closure[recip_name]
                    out.append(recip)
                    out.append(mult)
                    continue
            out.append(s)
        return Body(out)

    return walk(stmts)


# ---------------------------------------------------------------------------
# Pass 5b: loop-invariant code motion
# ---------------------------------------------------------------------------


def hoist_loop_invariants(stmts: Body) -> Body:
    """Move stmts out of ``Loop``s whose axis they don't depend on.

    Hoists ``Load`` / ``Assign`` / ``Select`` (SSA values) and entire
    ``Loop`` / ``StridedLoop`` / ``Tile`` / ``Cond`` blocks whose contents
    transitively avoid the outer axis — provided the block contains no
    ``Write`` (a Write hoist would change observable side effects).
    Block-level hoisting is what lets a Loop and its downstream consumer
    move together: hoisting just the consumer would leave it referencing
    an Accum still defined inside the outer Loop body.

    ``Accum`` / ``Init`` / ``Write`` always stay (iteration-tied
    semantics). Loop-invariance is queried via :meth:`Body.depends_on`
    against the body's transitive read closure, so the hoisted set is
    automatically closed under SSA dependencies — no separate ordering
    check is needed.
    """
    stmts = Body.coerce(stmts)

    def _hoistable(s: Stmt, axis: str) -> bool:
        # Accum / Init are scope-bound to their enclosing Loop's reduction — they can't
        # move alone, but the whole enclosing block can. Side-effecting stmts (Write, or
        # any block containing a Write) pin their iteration count and stay put.
        if isinstance(s, (Accum, Init)) or s.has_side_effects():
            return False
        return not stmts.depends_on(s, axis)

    def walk(body: Body) -> list[Stmt]:
        new_body: list[Stmt] = []
        for s in body:
            if isinstance(s, (Loop, StridedLoop)):
                inner = walk(s.body)
                axis = s.axis.name
                hoisted = [c for c in inner if _hoistable(c, axis)]
                hoisted_ids = {id(c) for c in hoisted}
                stay = [c for c in inner if id(c) not in hoisted_ids]
                new_body.extend(hoisted)
                new_body.append(replace(s, body=tuple(stay)))
            elif isinstance(s, Tile):
                new_body.append(Tile(axes=s.axes, body=tuple(walk(s.body))))
            elif isinstance(s, Cond):
                new_body.append(Cond(cond=s.cond, body=tuple(walk(s.body)), else_body=tuple(walk(s.else_body))))
            else:
                new_body.append(s)
        return new_body

    return tuple(walk(stmts))


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
        return Write(stmt.output, _simplify_expr_tuple(stmt.index, ctx), stmt.value, reduce_op=stmt.reduce_op)
    if isinstance(stmt, Load):
        return Load(stmt.name, stmt.input, _simplify_expr_tuple(stmt.index, ctx))
    # Tile-IR-only ``Stage`` (+ BufferedStage / AsyncBufferedStage /
    # TmaBufferedStage subtypes). Each subclass overrides ``_simplify_kwargs``
    # to thread its own Expr fields through ``ctx`` and forward all
    # subtype-specific fields (e.g. ``swizzle``); we just call it and
    # reconstruct via ``type(stmt)(**kwargs)``. Adding a new Stage field
    # only needs the subclass's own override, not a branch here.
    # Lazy import to avoid a tile→stmt circular at module load time.
    from deplodock.compiler.ir.tile.ir import Stage  # noqa: PLC0415

    if isinstance(stmt, Stage):
        return type(stmt)(**stmt._simplify_kwargs(ctx))
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
