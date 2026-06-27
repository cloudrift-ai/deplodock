"""Tile-IR enumeration: ``LoopOp`` → ``TileOp`` (schedule selection).

Reads each loop kernel's algebraic kind (``Loop.algebra_kind``) and chooses a
schedule for it. The skeleton currently schedules the no-fold kind — onto the
thread grid, one thread per output cell. Kernels carrying a combine are left
un-lowered until their schedule is built; they reuse the same op by supplying the
combine, per ``plans/tile-ir-rebuild.md``.
"""
