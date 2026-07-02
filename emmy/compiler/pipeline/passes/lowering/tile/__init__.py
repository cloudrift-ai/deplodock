"""Tile-IR lowering: ``LoopOp`` → ``TileOp``, in two rules.

1. **Recognize + Schedule** (``010_recognize``) — the Loop-IR → Tile-IR boundary, both halves
   in ONE rewrite. **Recognition**: fuse flash attention, fuse online softmax, annotate each
   reduce ``Loop`` with its ``AxisRole`` + ``Carrier``, then **lift** the kernel to a ``Kernel``
   carrying ONE op-tree ``Map`` — a thin ``Body`` wrapper over the annotated loop nest — with an
   **unmapped** placement (its parallel ``free`` axes). **Scheduling** (the ``_schedule`` helper, called inline): map the
   ``free`` axes onto ``grid`` and offer the per-axis scheduling forks — the reduce-axis partition
   (the ``REDUCE`` codec → ``ReducePlan``) and a contraction's output tile (the ``TILE`` codec) —
   dispatched on the axes' ``AxisRole``, never a kernel kind. After this nothing downstream
   traffics in ``LoopOp``. (Flash recognition is a *graph rewrite*, so its fused ``TileOp`` is
   scheduled when it re-enters the rule — the rule matches ``LoopOp`` AND an unmapped ``TileOp``.)
   The ``_flash`` / ``_softmax`` helpers hold the pattern matchers; ``_schedule`` holds the
   geometry + reduce-partition logic and the ``REDUCE`` / ``TILE`` / ``STAGE`` / ``WSPEC`` knobs.
2. **Split** (``030_split``) — consume a cross-CTA ``GRID`` stage (``ReducePlan.needs_split``)
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
