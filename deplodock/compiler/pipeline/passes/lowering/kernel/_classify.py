"""Classify a ``LoopOp`` body for the unified emit skeleton.

Recognizes three body shapes and returns a ``UnifiedSig`` describing the
geometry (free axes, reduction axes, live axes during reduction) and the
partitioned stmt sequences (prologue / per-Accum reduce bodies / interlude /
output / epilogue / Write). Returns ``None`` for shapes the unified emitter
doesn't yet support — caller falls back to the legacy emitter.

Supported shapes (after fusion + normalization):

1. **Pointwise** — body has no Accum. After zero or more outer free Loops we
   reach a leaf Write. Live axes = ∅; |live|=0 → no smem, no syncs.

2. **Per-row reduction** — outer free Loop(s) wrap a sequence of
   [reduce Loop+, interlude assigns, *optional* free output Loop, Write].
   Two sub-shapes:
   - With output Loop (RMSNorm / softmax): the output axis is iterated
     serially inside each thread; Write lives in the inner Loop.
     ``output_axis`` is set; live axes = outer free axes.
   - Without output Loop (reduce-to-scalar per row, matmul without
     epilogue): Write sits at the reduce-Loop level. ``output_axis``
     is ``None``; live axes = all free axes (each thread owns one
     output element and walks the reduction serially).

Live axes per Accum must agree (softmax's two reductions both have
live={i}); we reject when they disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.expr import free_vars
from deplodock.compiler.ir.loop import Accum, Assign, Axis, Load, Loop, LoopOp, Select, Stmt, Write


@dataclass(frozen=True)
class ReduceBlock:
    """One reduce Loop's body, parsed.

    ``body_stmts`` are the original reduce-Loop body in body order (Loads +
    Assigns + Selects + the Accum). ``accum`` is the single Accum the body
    folds into. ``staged_loads`` are the Loads whose index references the
    reduce axis (need cooperative staging into smem); ``invariant_loads``
    are the rest (loop-invariant, hoistable).
    """

    axis: Axis
    body_stmts: tuple[Stmt, ...]
    accum: Accum
    staged_loads: tuple[Load, ...]
    invariant_loads: tuple[Load, ...]


@dataclass(frozen=True)
class UnifiedSig:
    """Geometry + partitioned body for the unified emitter.

    Shape kinds:
    - ``"pointwise"``: ``reduce_blocks`` is empty; ``output_chain`` is the
      stmts inside the innermost free Loop ending with Write.
    - ``"per_row_reduce"``: one or more ``reduce_blocks`` (e.g. softmax has
      max + sum), then ``interlude`` (per-row finishing math like rsqrt/
      reciprocal), then ``output_chain`` (the per-element output pass) and
      ``output_axis`` (the free axis the output Loop iterates over).

    ``free_axes`` are the iteration axes the output Write spans (in
    occurrence order). ``live_axes`` ⊆ ``free_axes`` are the axes still
    bound during each reduction (after stripping the reduce axis).
    """

    kind: str  # "pointwise" | "per_row_reduce"
    free_axes: tuple[Axis, ...]
    live_axes: tuple[Axis, ...]
    top_level: tuple[Stmt, ...]  # stmts at LoopOp.body root, before the outer free Loop
    pre_reduce: tuple[Stmt, ...]  # per-row stmts hoisted above the reduce Loop (e.g. bias Load)
    reduce_blocks: tuple[ReduceBlock, ...]
    interlude: tuple[Stmt, ...]  # per-row stmts between last reduce and output Loop / Write
    output_axis: Axis | None  # axis of the output free Loop (per_row_reduce); None for pointwise
    output_chain: tuple[Stmt, ...]  # stmts at the output level, ending with Write
    write: Write


# ---------------------------------------------------------------------------
# Body walking helpers
# ---------------------------------------------------------------------------


def _stmt_axis_refs(stmt: Stmt) -> frozenset[str]:
    """Axis-Var names directly referenced by a stmt's index / predicate exprs."""
    out: set[str] = set()
    if isinstance(stmt, Load):
        for e in stmt.index:
            out |= set(free_vars(e))
    elif isinstance(stmt, Select):
        for b in stmt.branches:
            out |= set(free_vars(b.select))
    return frozenset(out)


def _is_reduce_loop(loop: Loop) -> bool:
    """True iff this Loop's immediate body contains an Accum."""
    return any(isinstance(s, Accum) for s in loop.body)


def _split_top_level(body: tuple[Stmt, ...]) -> tuple[tuple[Stmt, ...], Loop | None, tuple[Stmt, ...]] | None:
    """Split a body into ``(pre_stmts, the_outer_loop, post_stmts)``.

    The outer loop is the unique Loop in this body. Returns ``None`` if there
    are zero or two-or-more sibling Loops at the top — the unified shapes all
    have exactly one outer free Loop.
    """
    pre: list[Stmt] = []
    post: list[Stmt] = []
    the_loop: Loop | None = None
    for s in body:
        if isinstance(s, Loop):
            if the_loop is not None:
                return None
            the_loop = s
        elif the_loop is None:
            pre.append(s)
        else:
            post.append(s)
    return tuple(pre), the_loop, tuple(post)


def _descend_outer_free(body: tuple[Stmt, ...]) -> tuple[list[Axis], tuple[Stmt, ...]] | None:
    """Walk down through nested single-Loop bodies as long as the wrapping
    Loop is a free Loop. Returns ``(free_axes_in_order, leaf_body)``.

    Stops at a level that contains a reduce Loop, multiple sibling Loops, or
    a Write — the leaf body is what's at that level.
    """
    free_axes: list[Axis] = []
    cur = body
    while True:
        # If this level has a Write or any Accum at top, stop here.
        if any(isinstance(s, Write | Accum) for s in cur):
            return free_axes, cur
        loops = [s for s in cur if isinstance(s, Loop)]
        if not loops:
            return free_axes, cur
        if len(loops) > 1:
            return free_axes, cur
        only = loops[0]
        if _is_reduce_loop(only):
            return free_axes, cur
        # Single non-reduce Loop: descend.
        free_axes.append(only.axis)
        cur = only.body


# ---------------------------------------------------------------------------
# Reduce-block parsing
# ---------------------------------------------------------------------------


def _parse_reduce_block(loop: Loop) -> ReduceBlock | None:
    """Validate a reduce Loop and partition its body.

    Body must be Loads / Assigns / Selects / a single Accum. We separate the
    Loads into staged (index references the reduce axis) vs invariant.
    """
    accums = [s for s in loop.body if isinstance(s, Accum)]
    if len(accums) != 1:
        return None
    accum = accums[0]
    body_stmts: list[Stmt] = []
    staged: list[Load] = []
    invariant: list[Load] = []
    axis_name = loop.axis.name
    for s in loop.body:
        if isinstance(s, (Load, Assign, Select)):
            body_stmts.append(s)
            if isinstance(s, Load):
                if axis_name in _stmt_axis_refs(s):
                    staged.append(s)
                else:
                    invariant.append(s)
        elif isinstance(s, Accum):
            body_stmts.append(s)
        else:
            return None
    return ReduceBlock(
        axis=loop.axis,
        body_stmts=tuple(body_stmts),
        accum=accum,
        staged_loads=tuple(staged),
        invariant_loads=tuple(invariant),
    )


def _validate_output_body(body: tuple[Stmt, ...]) -> bool:
    """An output Loop's body may contain Loads / Assigns / Selects / Writes
    plus nested reduce Loops; nothing else (e.g. free Loops are not yet
    supported inside the output Loop). Recursively descends into nested
    reduce Loops to validate them.
    """
    for s in body:
        if isinstance(s, (Load, Assign, Select, Write)):
            continue
        if isinstance(s, Loop):
            if not _is_reduce_loop(s):
                return False
            if _parse_reduce_block(s) is None:
                return False
            continue
        return False
    return True


def _live_axes_of_block(block: ReduceBlock, all_free: frozenset[str]) -> frozenset[str]:
    """Free-axes that are referenced from inside this reduce block.

    Walks every stmt's axis refs (Load.index, Select.select) and intersects
    with ``all_free``. The reduce axis itself is excluded.
    """
    refs: set[str] = set()
    for s in block.body_stmts:
        refs |= _stmt_axis_refs(s)
    return frozenset(refs) & all_free


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def classify(loop_op: LoopOp) -> UnifiedSig | None:
    """Recognize a body shape; return ``None`` if it's not yet supported."""
    # Step 1: peel any top-level non-Loop stmts (typically scalar Loads).
    split = _split_top_level(loop_op.body)
    if split is None:
        # Multiple sibling Loops at root — not a supported shape.
        return None
    top_level, outer_loop, post_outer = split
    if outer_loop is None:
        # Body is just a flat list ending in Write (degenerate scalar case).
        return _classify_leaf(loop_op, top_level=(), free_axes=[], leaf=loop_op.body)
    if post_outer:
        # Stmts after the outer Loop are not supported.
        return None

    # top_level stmts must be Load / Assign / Select with no axis refs
    # (true scalars or broadcasts that don't need any free axis).
    for s in top_level:
        if not isinstance(s, (Load, Assign, Select)):
            return None
        if _stmt_axis_refs(s):
            return None

    # Step 2: descend through outer free Loops.
    descent = _descend_outer_free((outer_loop,))
    if descent is None:
        return None
    free_axes, leaf = descent
    return _classify_leaf(loop_op, top_level=top_level, free_axes=free_axes, leaf=leaf)


def _classify_leaf(
    loop_op: LoopOp,
    *,
    top_level: tuple[Stmt, ...],
    free_axes: list[Axis],
    leaf: tuple[Stmt, ...],
) -> UnifiedSig | None:
    """Parse the leaf body once outer free Loops are stripped.

    Two recognized leaf shapes:

    - **Pointwise leaf**: only Loads / Assigns / Selects + a Write. The leaf
      body is the output chain.
    - **Per-row-reduce leaf**: a sequence of reduce Loops, then per-row
      interlude assigns, then a single free output Loop containing the
      output chain + Write.
    """
    free_names = frozenset(a.name for a in free_axes)

    # --- Pointwise: no Loops in the leaf, ends with Write ---
    if not any(isinstance(s, Loop) for s in leaf):
        if not leaf or not isinstance(leaf[-1], Write):
            return None
        if not all(isinstance(s, (Load, Assign, Select, Write)) for s in leaf):
            return None
        write = leaf[-1]
        return UnifiedSig(
            kind="pointwise",
            free_axes=tuple(free_axes),
            live_axes=(),
            top_level=top_level,
            pre_reduce=(),
            reduce_blocks=(),
            interlude=(),
            output_axis=None,
            output_chain=leaf,
            write=write,
        )

    # --- Per-row-reduce: pre_reduce + reduce Loops + interlude + (output Loop | Write) ---
    pre_reduce: list[Stmt] = []
    reduce_blocks: list[ReduceBlock] = []
    interlude: list[Stmt] = []
    output_axis: Axis | None = None
    output_chain: tuple[Stmt, ...] | None = None
    write: Write | None = None
    seen_output_loop = False
    direct_writes: list[Write] = []
    seen_reduce = False

    for s in leaf:
        if isinstance(s, Loop):
            if seen_output_loop or direct_writes:
                return None
            if _is_reduce_loop(s):
                blk = _parse_reduce_block(s)
                if blk is None:
                    return None
                reduce_blocks.append(blk)
                seen_reduce = True
            else:
                # Free output Loop. Body is a single trailing Write plus any mix
                # of Loads / Assigns / Selects / nested reduce Loops. Each nested
                # reduce Loop becomes a per-element reduction inside the output
                # for-loop (flash-attention-style fused matmul-after-softmax).
                writes = [x for x in s.body if isinstance(x, Write)]
                if len(writes) != 1 or s.body[-1] is not writes[0]:
                    return None
                if not _validate_output_body(s.body):
                    return None
                output_axis = s.axis
                output_chain = s.body
                write = writes[0]
                seen_output_loop = True
        elif isinstance(s, (Load, Assign, Select)):
            if seen_output_loop or direct_writes:
                return None
            if seen_reduce:
                interlude.append(s)
            else:
                pre_reduce.append(s)
        elif isinstance(s, Write):
            if seen_output_loop:
                return None
            direct_writes.append(s)
        else:
            return None

    if not reduce_blocks:
        return None
    if seen_output_loop == bool(direct_writes):
        # Need exactly one of: output Loop, or one direct Write.
        if not direct_writes or len(direct_writes) != 1:
            return None

    if direct_writes:
        # Matmul-like / scalar-per-row shape: every free axis stays live; each
        # thread owns one (i, j, ...) output element and runs the reduction serially.
        write = direct_writes[0]
        output_chain = (write,)
        full_free_axes = tuple(free_axes)
        all_free_set = free_names
    else:
        assert output_axis is not None and write is not None and output_chain is not None
        full_free_axes = tuple(free_axes) + (output_axis,)
        all_free_set = free_names | {output_axis.name}

    # Compute live axes and verify all reduce blocks agree.
    live_sets = [_live_axes_of_block(b, all_free_set) for b in reduce_blocks]
    common_live = live_sets[0]
    for ls in live_sets[1:]:
        if ls != common_live:
            return None

    # When there's no output Loop, every free axis must be live (otherwise some
    # output element would not be uniquely owned by a thread).
    if direct_writes and common_live != all_free_set:
        return None

    live_axes = tuple(a for a in full_free_axes if a.name in common_live)

    return UnifiedSig(
        kind="per_row_reduce",
        free_axes=full_free_axes,
        live_axes=live_axes,
        top_level=top_level,
        pre_reduce=tuple(pre_reduce),
        reduce_blocks=tuple(reduce_blocks),
        interlude=tuple(interlude),
        output_axis=output_axis,
        output_chain=output_chain,
        write=write,
    )


__all__ = ["ReduceBlock", "UnifiedSig", "classify"]
