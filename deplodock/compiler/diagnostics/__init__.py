"""Static diagnostics over Tile IR — no GPU required.

These helpers walk a compiled ``Graph`` and report properties that would
otherwise need a profiler (ncu / nsys) to surface. Today's only module is
``bank_conflicts`` (smem bank-conflict simulation per Stage); future
modules might add occupancy estimation, smem footprint heatmaps, etc.
"""
