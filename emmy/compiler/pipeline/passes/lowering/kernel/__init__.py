"""Tile IR → Kernel IR lowering passes.

Each ``TileOp`` in the graph becomes a ``KernelOp`` whose body is the
fully-scheduled kernel form (``Tile`` / ``Tile`` / ``Smem`` /
``Sync`` / ``TreeHalve`` / ``StridedLoop`` + Loop-IR leaves).
"""
