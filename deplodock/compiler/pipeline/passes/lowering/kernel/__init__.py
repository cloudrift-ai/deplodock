"""Kernel-IR lowering pass: ``TileOp`` → ``KernelOp`` (schedule materialization).

Binds a scheduled :class:`~deplodock.compiler.ir.tile.TileOp` to hardware by
mapping its ``schedule.grid`` axes onto the thread grid — wrapping the per-cell body in a
:class:`Tile`. The step is algebra-generic — a kernel's fold rides inside the
per-cell body and materializes through its carrier — so further kernel kinds
reuse it once their schedules are enumerated (see ``plans/tile-ir-rebuild.md``).
"""
