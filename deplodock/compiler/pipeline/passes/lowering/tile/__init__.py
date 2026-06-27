"""Tile-IR lowering: ``LoopOp`` → ``TileOp``, in two steps.

1. **Recognize** (``010_recognize``) — read the reduce carrier's algebra and
   normalize it to the unified twisted ``Monoid`` (a scalar ``Accum`` becomes its
   degenerate, identity-twist monoid; an online-softmax ``Monoid`` is kept;
   a ``SEMIRING`` contraction is left alone). After this, a plain reduction and
   online softmax share one representation.
2. **Schedule** (``020_schedule``) — choose the per-cell schedule: map the free
   (output) axes onto the thread grid and keep any fold serial in the per-cell
   body.

Recognition is purely about algebraic structure; scheduling is purely geometry —
so neither, nor the downstream lowering, branches on reduction-vs-softmax. The
contraction (``SEMIRING``) schedule and the cooperative / cross-CTA reduce
schedules arrive later, per ``plans/tile-ir-rebuild.md``.
"""
