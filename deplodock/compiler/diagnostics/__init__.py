"""Diagnostics over the compiler IR.

:mod:`.bank_conflicts` exposes two layers sharing the
``BankConflictResult`` shape:

- :func:`.bank_conflicts.lane_bank_distribution` — pure oracle over Tile
  IR. Symbolic ``Expr.eval`` per lane against declared smem layouts.
  Fast, GPU-free, used by Tile-IR passes (``070_pad_smem``,
  ``060_permute_lane_accesses``) to score candidate rewrites.
- :func:`.bank_conflicts.simulate_graph` — Kernel-IR static analyzer.
  Lowers the input graph through ``KERNEL_PASSES`` and walks each
  ``KernelOp`` body for smem ``Load``s, computing per-lane addresses,
  per-bank distinct-address counts, LDS.128 chain absorption, and
  per-cell access provenance over the inner-loop sweep. GPU-free —
  used by the visualizer (``scripts/visualize_bank_conflicts.py``).
"""
