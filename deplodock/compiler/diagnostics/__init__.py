"""Diagnostics over the compiler IR.

Two paths share the ``BankConflictResult`` shape:

- :mod:`.bank_conflicts` — static simulator over Tile IR. Symbolic
  ``Expr.eval`` per lane against declared smem layouts. Fast, GPU-free,
  used by Tile-IR passes (``014_pad_smem``, ``009_permute_register_tile``)
  to score candidate rewrites.
- :mod:`.bank_conflicts_dynamic` — dynamic simulator. Compiles the graph
  with smem-Load instrumentation enabled (sets ``DEPLODOCK_BANK_TRACE=1``
  for the rule
  ``pipeline/passes/lowering/kernel/002_instrument_smem_loads``), runs
  one CTA on the GPU, and decodes the recorded addresses. Ground truth —
  used by the visualizer (``scripts/visualize_bank_conflicts.py``).
"""
