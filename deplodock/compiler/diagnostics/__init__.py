"""Diagnostics over the compiler IR.

:mod:`.bank_conflicts` exposes two layers sharing the
``BankConflictResult`` shape:

- :func:`.bank_conflicts.lane_bank_distribution` — pure oracle over Tile
  IR. Symbolic ``Expr.eval`` per lane against declared smem layouts.
  Fast, GPU-free, used by Tile-IR passes (``014_pad_smem``,
  ``009_permute_register_tile``) to score candidate rewrites.
- :func:`.bank_conflicts.simulate_graph` — runtime trace.
  Compiles the graph with smem-Load instrumentation enabled (sets
  ``DEPLODOCK_BANK_TRACE=1`` for the rule
  ``pipeline/passes/lowering/kernel/002_instrument_smem_loads``), runs
  one CTA on the GPU, and decodes the recorded addresses. Ground truth —
  used by the visualizer (``scripts/visualize_bank_conflicts.py``).
"""
