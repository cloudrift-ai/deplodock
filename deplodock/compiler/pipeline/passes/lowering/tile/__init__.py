"""Tile-IR lowering: ``LoopOp`` → ``TileOp``, in three rules.

1. **Recognize** (``010_recognize``) — the Loop-IR → Tile-IR boundary. ALL recognition
   lives here (no separate flash / softmax rule): fuse flash attention, fuse online
   softmax, normalize plain ``Accum``\\ s to twisted ``Monoid``s, then **lift** the kernel
   to a ``TileOp`` carrying ONE op-tree ``AlgebraNode`` (``Map`` / ``Monoid`` / ``Semiring``)
   with an **unmapped** ``Schedule`` (its parallel ``free`` axes, on the ``TileOp`` not the
   node). After this nothing downstream traffics in ``LoopOp``. The ``_flash`` / ``_softmax``
   helper modules hold the flash / online-softmax pattern matchers the rule calls.
2. **Schedule** (``020_schedule``) — geometry + the reduce partition: map the schedule's
   ``free`` axes onto ``grid`` (the per-cell tier) and pick the reduce-axis partition (the
   ``REDUCE`` codec → ``ReducePlan``; a cross-CTA ``g`` split pin flows onto both ``Monoid``
   and ``Semiring`` schedules) and a contraction's output tile (the ``TILE`` codec).
3. **Split** (``030_split``) — consume a cross-CTA ``GRID`` stage (``ReducePlan.needs_split``)
   as a **graph rewrite**: a partial kernel reduces each CTA's slice of the reduce axis and
   either ``atomicAdd``\\ s its (additive) state into the output (one kernel) or writes it to a
   ``__partial`` workspace folded by a sibling finalize kernel (the carrier's
   ``combine_states`` over the split axis — additive ``sum`` / split-K matmul AND the twisted
   flash ``(m, l, O)`` split-KV). The schedule carries the partition; the graph carries the
   kernel count, so ``lowering/kernel`` only ever sees single-launch kernels.

Recognition reads algebraic structure; scheduling is geometry; materialization back to
loop IR happens in ``lowering/kernel`` — so the tile passes work purely with algebra
primitives. The mma / blocked contraction schedules arrive later as richer mappings of the
same ``free`` axes.
"""
