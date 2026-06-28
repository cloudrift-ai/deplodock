"""Kernel-IR lowering passes: schedule materialization + post-materialize codegen.

``010_materialize`` binds a scheduled :class:`~deplodock.compiler.ir.tile.TileOp`
to hardware by mapping its ``schedule.grid`` axes onto the thread grid — wrapping
the per-cell body in a :class:`Tile`. The step is algebra-generic — a kernel's
fold rides inside the per-cell body and materializes through its carrier — so
further kernel kinds reuse it once their schedules are enumerated.

The remaining rules are ``KernelOp`` → ``KernelOp`` codegen-quality passes that
walk the materialized body of statements; they are independent of the tile IR
(they only read/rewrite the kernel-IR ``Load`` / ``Assign`` / ``Write`` / ``Sync``
vocabulary), so they were restored from the demolished pipeline ahead of the
tile-tier rebuild:

- ``030_stamp_types`` — stamp Load/Assign/Write dtypes (so the next two read
  them off the IR; overflow-prone fp16 squares promote to f32).
- ``040_demote_to_write_dtype`` — demote native-fp16-able elementwise chains
  feeding an fp16 Write to fp16 (dormant under today's f32 accumulators).
- ``050_vectorize_loads`` / ``080_vectorize_stores`` — fold consecutive scalar
  Loads / Writes into one wide vector access.
- ``095_interleave_loads`` — sink each Load to just before its first consumer.
- ``110_drop_redundant_syncs`` — collapse no-op Sync stmts at the Tile body level.

The tile-tier kernel passes (mma fragment lowering, register-tile split, smem
staging, fp16 K-window packing) stay demolished — they consume tile structures
the current materializer doesn't produce and belong to the tile rebuild.
"""
