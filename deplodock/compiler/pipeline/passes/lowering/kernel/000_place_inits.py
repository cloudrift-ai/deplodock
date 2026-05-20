"""Place explicit ``Init`` Stmts at the correct scope for every Accum.

Runs before ``001_materialize_tile`` so materialize sees IR where each
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
from deplodock.compiler.ir.stmt import Accum, Cond, Init, Loop, Stmt, StridedLoop, Tile, Write
from deplodock.compiler.ir.tile.ir import TileOp
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
    """For each nested ``Tile`` in the TileOp body, open a scope and place
    Inits inside it (the matmul / softmax / RMSNorm shape). When the
    TileOp's body has no nested ``Tile`` (the simple-reduction shape —
    body is a flat ``(Loop, Write)`` from the lowering pipeline), treat
    the TileOp body itself as the scope so its Accums still get explicit
    Inits with the chosen dtype."""
    has_nested_tile = any(isinstance(s, Tile) for s in body)
    if not has_nested_tile:
        return _place_inits_in_scope(body, accumulating_dtype, selecting_dtype)
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, Tile):
            new_tile_body = _place_inits_in_scope(s.body, accumulating_dtype, selecting_dtype)
            out.append(Tile(axes=s.axes, body=new_tile_body))
        else:
            out.append(s)
    return tuple(out)


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
    """Collect Accums reachable from ``stmt`` through reduce loops only.
    Stops at free loops — those Accums will be placed at the free loop's
    own scope when we recurse into it.

    "Reduce loop" here is the recursive notion: a Loop whose body has an
    Accum directly, *or* a nested reduce-Loop. This catches the
    K-chunked matmul shape where ``K_o``'s immediate body is ``[Stage,
    ..., Loop(K_i, reduce, body=[..., Accum])]`` — the inner Accum
    accumulates across ``K_o`` too, so ``K_o`` is crossable for Init
    placement and the Init lands at the surrounding Tile body head."""
    out: list[Accum] = []
    if isinstance(stmt, Accum):
        out.append(stmt)
    elif isinstance(stmt, (Loop, StridedLoop)):
        if _is_reduce_recursive(stmt):
            for child in stmt.body:
                out.extend(_accums_under_reduces_only(child))
    elif isinstance(stmt, Cond):
        for child in stmt.body:
            out.extend(_accums_under_reduces_only(child))
        for child in stmt.else_body:
            out.extend(_accums_under_reduces_only(child))
    return out


def _is_reduce_recursive(loop: Loop | StridedLoop) -> bool:
    """Recursive reduce check (Init-scoping only). A Loop is "crossable"
    iff its body either has an Accum directly OR forms a pure
    reduce-passthrough — no Write / output-escape stmts in body, and at
    least one nested reduce-Loop carries the Accum.

    The Write check is what distinguishes:

    - **Matmul K-chunked** (``K_o`` body = [Stage, Stage, Loop(K_i,
      reduce)]): no Write directly in body → crossable, Init lives at
      the surrounding Tile.
    - **SDPA per-output free loop** (``a4`` body = [Loop(reduce, ...),
      Write[..., a4]]): the Write escapes per-iteration, so the Accum
      must reset per ``a4`` → not crossable, Init lives inside ``a4``."""
    has_inner_reduce = False
    for s in loop.body:
        if isinstance(s, Accum):
            return True
        if isinstance(s, Write):
            return False
        if isinstance(s, (Loop, StridedLoop)) and _is_reduce_recursive(s):
            has_inner_reduce = True
    return has_inner_reduce


def _recurse(stmt: Stmt, init_dtypes: dict[str, DataType], accumulating_dtype: DataType, selecting_dtype: DataType) -> Stmt:
    """Descend into block-structured stmts so nested free Loops get their
    own Init placement at their own scope. Same recursive-reduce
    distinction as ``_accums_under_reduces_only``: a Loop that
    transitively wraps an Accum (matmul ``K_o`` chunking) descends
    without opening a new scope. ``StridedLoop`` is treated identically
    — SDPA's per-output free StridedLoop wraps a free Loop chain plus a
    Write, so it's not reduce-crossable and must open a scope so the
    inner Accum's Init lands inside (resetting per-iteration).

    Stamps the accumulator dtype on every ``Accum`` whose name we just
    placed an Init for at the enclosing scope, so Init and Accum agree."""
    if isinstance(stmt, Accum) and stmt.name in init_dtypes:
        return replace(stmt, dtype=init_dtypes[stmt.name])
    if isinstance(stmt, (Loop, StridedLoop)) and not _is_reduce_recursive(stmt):
        # Free loop — its body is its own scope; place Inits there.
        return replace(stmt, body=_place_inits_in_scope(stmt.body, accumulating_dtype, selecting_dtype))
    nested = stmt.nested()
    if not nested:
        return stmt
    return stmt.with_bodies(tuple(tuple(_recurse(c, init_dtypes, accumulating_dtype, selecting_dtype) for c in b) for b in nested))
