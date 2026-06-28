"""Tile-IR lowering: ``LoopOp`` → ``TileOp``, in two rules.

1. **Recognize** (``010_recognize``) — the Loop-IR → Tile-IR boundary. ALL recognition
   lives here (no separate flash / softmax rule): fuse flash attention, fuse online
   softmax, normalize plain ``Accum``\\ s to twisted ``Monoid``s, then **lift** the kernel
   to a ``TileOp`` carrying ONE op-tree ``AlgebraNode`` (``Map`` / ``Monoid`` / ``Semiring``)
   with its parallel axes on the node's ``free`` field and an **empty** schedule. After this
   nothing downstream traffics in ``LoopOp``. The ``_flash`` / ``_softmax`` helper modules
   hold the flash / online-softmax pattern matchers the rule calls.
2. **Schedule** (``020_schedule``) — pure geometry: move the lifted node's ``free`` axes
   onto ``TileOp.grid_axes`` (the per-cell, one-thread-per-output-cell tier maps every free
   axis onto the thread grid).

Recognition reads algebraic structure; scheduling is geometry; materialization back to
loop IR happens in ``lowering/kernel`` — so the tile passes work purely with algebra
primitives. The cooperative / cross-CTA reduce and the mma / blocked / split-K contraction
schedules arrive later as richer mappings of the same ``free`` axes
(``plans/tile-ir-rebuild.md``).
"""
