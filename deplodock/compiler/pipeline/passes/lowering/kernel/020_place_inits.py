"""Place explicit ``Init`` Stmts at the correct scope for every Accum.

Runs before ``100_materialize_tile`` so materialize sees IR where each
Accum's lifetime is explicit. Removes the need for materialize to make
implicit scoping decisions.

Scope rule: walk up from each Accum, crossing reduce Loops freely;
stop at the nearest enclosing free Loop or at the Tile body. Insert
``Init(name, op)`` at that scope, just before the loop chain that
contains the Accum.

Why this is correct:

- **Matmul** (``Loop(k_o, reduce) > Loop(k_i, reduce) > Accum``):
  walk crosses both reduces, hits Tile. Init goes at Tile body head;
  ``acc`` persists across ``k_o`` iterations as needed.

- **SDPA k_reduce** (``Loop(a3, free) > Loop(a4, reduce) > Accum``):
  walk crosses ``a4`` (reduce ✓), stops at ``a3`` (free ✗). Init goes
  at the start of ``a3``'s body; ``acc`` resets per a3 iteration.

The renderer's ``RenderCtx.explicit_inits`` mechanism suppresses the
per-Loop default init when an explicit ``Init`` exists for that name,
so the standard render path produces the right code unmodified.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.dtype import F16, F32, DataType
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.stmt import Accum, Cond, Init, Stmt, Write
from deplodock.compiler.ir.tile.ir import (
    GridTile,
    RegisterTile,
    SerialTile,
    StageBundle,
    StridedTile,
    ThreadTile,
    TileOp,
    WarpSpecialize,
    WarpTile,
)
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> Graph | None:
    # Two dtype slots for the Init-placement freeze:
    #
    # - ``accumulating_dtype``: for reductions that build up magnitude
    #   (sum, prod). When any input is fp16, we promote to F32 to avoid
    #   precision loss / overflow over the reduction range.
    # - ``selecting_dtype``: for reductions that pick one value (max,
    #   min). No magnitude is built up, so fp16 inputs can stay in fp16.
    #
    # Both fall back to ``F32`` when all inputs are f32 (legacy behavior).
    has_fp16_input = any(match.graph.nodes[inp].output.dtype == F16 for inp in root.inputs)
    accumulating_dtype = F32
    selecting_dtype = F16 if has_fp16_input else F32
    new_body = _place_inits_in_body(root.op.body, accumulating_dtype, selecting_dtype)
    if new_body == root.op.body:
        raise RuleSkipped("no Accum needs an Init placed (already placed or no reduce in body)")
    return TileOp(body=new_body, name=root.op.name)


# Reduction ops that select one input value (no magnitude accumulation)
# and can therefore stay in the input dtype. Everything else (sum, prod)
# uses the accumulating dtype.
_SELECTING_OPS: frozenset[str] = frozenset({"maximum", "amax", "minimum", "max", "min"})


def _decide_dtype(accum: Accum, accumulating_dtype: DataType, selecting_dtype: DataType) -> DataType:
    """Per-Accum dtype choice. Already-stamped Accums keep their dtype."""
    if accum.dtype is not None:
        return accum.dtype
    if accum.op.name in _SELECTING_OPS:
        return selecting_dtype
    return accumulating_dtype


def _place_inits_in_body(body: tuple, accumulating_dtype: DataType, selecting_dtype: DataType) -> tuple:
    """Find the innermost ``ParallelTile`` (or the TileOp body itself when
    none nests one) and place Inits at that scope.

    Under the new tile-flavor IR, the TileOp body wraps a ``GridTile``
    (cooperative) or a standalone ``ThreadTile`` (pointwise / single-CTA
    reduce). For matmul / softmax / RMSNorm the scope where Inits land is
    the ``ThreadTile`` body (the per-thread scope just above the K loop);
    for a flat reduce kernel with no nested ParallelTile, the TileOp body
    itself is the scope.
    """

    def _open_scope(stmts: tuple) -> tuple:
        # Find the deepest inner ParallelTile (ThreadTile or WarpTile) and
        # recurse into its body. WarpTile is treated the same as ThreadTile
        # at this scope: both bind the per-binding-tier coord that scopes
        # Init placement.
        out: list[Stmt] = []
        opened = False
        for s in stmts:
            if isinstance(s, GridTile):
                new_inner = _open_scope(tuple(s.body))
                out.append(s.with_bodies((new_inner,)))
                opened = True
            elif isinstance(s, (ThreadTile, WarpTile)):
                inner = tuple(s.body)
                if len(inner) == 1 and isinstance(inner[0], WarpSpecialize):
                    # Warp-specialized body: the consumer half owns the
                    # reduce loop and its Accums; the producer half only
                    # issues TMA loads and has no accumulator. Place Inits at
                    # the consumer_body head — the per-consumer-thread scope
                    # just above the K loop, exactly where a non-specialized
                    # ThreadTile would get them, shifted into the consumer
                    # arm. Placing them at the WarpTile body head instead
                    # would hoist the Init above the role split (producer
                    # threads run a dead Init) and — because that scope sits
                    # outside the consumer K loop the renderer can't tie the
                    # explicit Init to — the default per-loop init still
                    # fires inside the loop, resetting the accumulators every
                    # K chunk (the WS=1 accuracy bug).
                    ws = inner[0]
                    new_cons = _place_inits_in_scope(tuple(ws.consumer_body), accumulating_dtype, selecting_dtype)
                    out.append(s.with_bodies(((replace(ws, consumer_body=new_cons),),)))
                else:
                    new_inner = _place_inits_in_scope(tuple(s.body), accumulating_dtype, selecting_dtype)
                    out.append(s.with_bodies((new_inner,)))
                opened = True
            else:
                out.append(s)
        if not opened:
            # No ParallelTile in this scope — treat the scope itself.
            return _place_inits_in_scope(stmts, accumulating_dtype, selecting_dtype)
        return tuple(out)

    return _open_scope(body)


def _place_inits_in_scope(body: tuple, accumulating_dtype: DataType, selecting_dtype: DataType) -> tuple:
    """At each scope (Tile body, or a free-Loop body), collect all Accum
    names whose Init belongs HERE (= no free-Loop interposed between this
    scope and the Accum). Insert an Init for each at the start of the
    scope body. Then recurse into any nested free Loops to repeat at
    their scope.

    Decides accumulator dtype at placement time (the "freeze" point). For
    now: always ``F32``. The fp16-promotion policy that picks ``F32`` for
    reductions over fp16 values is a follow-up — once it lands here, the
    matching ``Accum`` Stmts get stamped with the same dtype.

    Idempotent on its own output: Accums whose Init already exists at
    this scope are left alone, so repeated applications converge."""
    existing: set[str] = {s.name for s in body if isinstance(s, Init)}
    inits_here: dict[str, Accum] = {}
    for s in body:
        for a in _accums_under_reduces_only(s):
            if a.name in existing:
                continue
            inits_here.setdefault(a.name, a)

    init_dtypes: dict[str, DataType] = {n: _decide_dtype(a, accumulating_dtype, selecting_dtype) for n, a in inits_here.items()}

    new_body: list[Stmt] = []
    for s in body:
        new_body.append(_recurse(s, init_dtypes, accumulating_dtype, selecting_dtype))

    init_stmts: list[Stmt] = [Init(name=n, op=a.op, dtype=init_dtypes[n]) for n, a in inits_here.items()]
    return tuple(init_stmts + new_body)


def _accums_under_reduces_only(stmt: Stmt) -> list[Accum]:
    """Collect Accums reachable from ``stmt`` through reduce tiles only.
    Stops at free tiles — those Accums will be placed at the free tile's
    own scope when we recurse into it.

    "Reduce tile" here is the recursive notion: a ``SerialTile`` /
    ``StridedTile`` whose body has an Accum directly, *or* a nested
    reduce-tile. This catches the K-chunked matmul shape where ``K_o``'s
    immediate body is ``[Stage, ..., SerialTile(K_i, reduce, body=[...,
    Accum])]`` — the inner Accum accumulates across ``K_o`` too, so
    ``K_o`` is crossable for Init placement and the Init lands at the
    surrounding ``ThreadTile`` body head.

    ``RegisterTile`` is transparent for Init scoping: ``010_split_register_axes``
    will replicate Accums across register cells, but at placement time the inner
    Accums are still single statements. We descend into ``RegisterTile.body``
    unconditionally."""
    out: list[Accum] = []
    if isinstance(stmt, Accum):
        out.append(stmt)
    elif isinstance(stmt, (SerialTile, StridedTile)):
        if _is_reduce_recursive(stmt):
            for child in stmt.body:
                out.extend(_accums_under_reduces_only(child))
    elif isinstance(stmt, RegisterTile):
        for child in stmt.body:
            out.extend(_accums_under_reduces_only(child))
    elif isinstance(stmt, Cond):
        for child in stmt.body:
            out.extend(_accums_under_reduces_only(child))
        for child in stmt.else_body:
            out.extend(_accums_under_reduces_only(child))
    elif isinstance(stmt, StageBundle):
        # StageBundle: consumer subtree (containing the reduce + Accum)
        # lives inside bundle.body. Walk through it so Accums get their Init
        # placed at the enclosing scope (the bundle is transparent for
        # accumulator scoping). The cooperative compute phase could in
        # principle hold an Accum too — descend it as well.
        for child in stmt.body:
            out.extend(_accums_under_reduces_only(child))
        if stmt.compute is not None:
            for child in stmt.compute:
                out.extend(_accums_under_reduces_only(child))
    return out


def _is_reduce_recursive(loop) -> bool:
    """Recursive reduce check (Init-scoping only). A ``SerialTile`` /
    ``StridedTile`` is "crossable" iff its body either has an Accum
    directly OR forms a pure reduce-passthrough — no Write / output-escape
    stmts in body, and at least one nested reduce-tile carries the Accum.

    The Write check is what distinguishes:

    - **Matmul K-chunked** (``K_o`` body = [Stage, Stage, SerialTile(K_i,
      reduce)]): no Write directly in body → crossable, Init lives at
      the surrounding ThreadTile.
    - **SDPA per-output free loop** (``a4`` body = [SerialTile(reduce,
      ...), Write[..., a4]]): the Write escapes per-iteration, so the
      Accum must reset per ``a4`` → not crossable, Init lives inside
      ``a4``.

    ``RegisterTile`` is transparent: descend into its body for the same
    crossability check. ``Cond`` is likewise transparent — its body /
    else_body get the same recursive treatment (the masked register-
    blocked GEMM emits ``RegisterTile(N_r, [Cond(pred, [Load, Assign,
    Accum])])`` per N-dependent tail; the Cond shouldn't hide the Accum
    from the crossability check)."""
    has_inner_reduce = False
    for s in loop.body:
        if isinstance(s, Accum):
            return True
        if isinstance(s, Write):
            return False
        if isinstance(s, (SerialTile, StridedTile)) and _is_reduce_recursive(s):
            has_inner_reduce = True
        elif isinstance(s, RegisterTile) and _is_reduce_recursive(s):
            has_inner_reduce = True
        elif isinstance(s, Cond):
            # Treat Cond as a transparent wrapper: an Accum nested behind
            # a per-cell predicate (the masked-N reg_block path) still
            # accumulates across the enclosing reduce loop. ``Write``
            # inside the Cond escapes per-iter the same way it does at
            # body level — recurse and honor its False-return.
            for branch in (s.body, s.else_body):

                class _Probe:
                    body = branch  # noqa: B023 — bound at iteration

                if _is_reduce_recursive(_Probe()):
                    has_inner_reduce = True
        elif isinstance(s, StageBundle):
            # StageBundle is transparent for reduce-crossing: synthesize a
            # probe-loop carrying the bundle's body so the recursive walk
            # treats the consumer subtree as if it were the loop's body.
            class _Probe:
                body = s.body

            if _is_reduce_recursive(_Probe()):
                has_inner_reduce = True
    return has_inner_reduce


def _recurse(stmt: Stmt, init_dtypes: dict[str, DataType], accumulating_dtype: DataType, selecting_dtype: DataType) -> Stmt:
    """Descend into block-structured stmts so nested free tiles get their
    own Init placement at their own scope. Same recursive-reduce
    distinction as ``_accums_under_reduces_only``: a SerialTile / StridedTile
    / RegisterTile that transitively wraps an Accum (matmul ``K_o``
    chunking) descends without opening a new scope. Free tiles open a
    scope so the inner Accum's Init lands inside (resetting per-iteration).

    Stamps the accumulator dtype on every ``Accum`` whose name we just
    placed an Init for at the enclosing scope, so Init and Accum agree.

    ``ParallelTile`` (Grid/Thread/RegisterTile) are NOT scope-openers —
    the scope is opened once at the outermost ``ThreadTile`` body. We
    pass through them transparently."""
    if isinstance(stmt, Accum) and stmt.name in init_dtypes:
        return replace(stmt, dtype=init_dtypes[stmt.name])
    if isinstance(stmt, (SerialTile, StridedTile)) and not _is_reduce_recursive(stmt):
        # Free serial-tile loop — its body is its own scope; place Inits there.
        return replace(stmt, body=_place_inits_in_scope(stmt.body, accumulating_dtype, selecting_dtype))
    nested = stmt.nested()
    if not nested:
        return stmt
    return stmt.with_bodies(tuple(tuple(_recurse(c, init_dtypes, accumulating_dtype, selecting_dtype) for c in b) for b in nested))
