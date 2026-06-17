"""Liveness-based memory planning for the CUDA runtime allocator.

The backend lowers a whole program (e.g. a 28-layer transformer) to a flat list
of kernel launches in topological order. Naively each graph node's output gets
its own permanently-live device buffer — so every layer's ``[heads, S, S]``
attention scratch stays resident at once (~29 GB for Qwen3-Embedding at S=4096 →
cupy OOM). But a *scratch* buffer is live only from the launch that writes it
until the last launch that reads it; two scratch buffers whose live intervals
don't overlap can share the same memory.

These pure helpers compute those live intervals and pack the scratch buffers into
one slab (offset per buffer) via a greedy-by-size memory planner (the strategy
TFLite's ``GreedyMemoryPlanner`` uses). They take no cupy and no graph — only the
launch list + sizes — so they unit-test on a CPU box. ``program.py`` turns the
offsets into typed views over one persistent ``cupy`` slab.

Input/constant/output buffers are NOT planned here: inputs are uploaded (and read
across the whole call), outputs are sliced out at the end, constants are weights —
all persistent. Only ``role == "scratch"`` buffers are reuse candidates.
"""

from __future__ import annotations


def compute_live_intervals(scratch_names: list[str], launches: list) -> dict[str, tuple[int, int]]:
    """Per scratch buffer, its half-open live interval ``[first_write, free_at)``
    over the topological launch order (``launches`` is already topo-sorted).

    - ``first_write`` = index of the launch whose ``node_id`` is the buffer (one
      buffer is produced by exactly one launch).
    - ``free_at`` = ``last_read + 1``, where ``last_read`` is the highest launch
      index that *reads* the buffer. A launch reads buffer ``b`` when ``b`` is one
      of its ``arg_names`` (kernel params) OR the ``src_buf`` of one of its TMA
      descriptors (a TMA-loaded buffer appears as the descriptor name in
      ``arg_names``, not the buffer name — so its source must be counted
      explicitly or the buffer would look dead and be aliased while still read).

    The half-open ``+1`` (free strictly after last use) is load-bearing: a launch
    at index ``i`` produces its output (interval starts at ``i``) while reading its
    inputs (each input's ``free_at >= i+1``), so the overlap test in
    :func:`plan_offsets` always treats an input live at ``i`` and the output born
    at ``i`` as overlapping — a launch's output can never alias its own live input.

    Raises on a scratch buffer with no producer or no consumer (a lowering-contract
    violation — surfaced loudly rather than silently leaked/mis-aliased).
    """
    scratch = set(scratch_names)
    first_write: dict[str, int] = {}
    last_read: dict[str, int] = {}
    for i, ln in enumerate(launches):
        if ln.node_id in scratch:
            first_write[ln.node_id] = i
        reads = set(ln.arg_names)
        reads.update(d.src_buf for d in ln.tma_descriptors)
        for name in reads:
            if name in scratch and name != ln.node_id:
                last_read[name] = i
    intervals: dict[str, tuple[int, int]] = {}
    for name in scratch_names:
        if name not in first_write:
            raise ValueError(f"scratch buffer {name!r} has no producing launch")
        lr = last_read.get(name)
        if lr is None:
            raise ValueError(f"scratch buffer {name!r} has no consuming launch (dead scratch)")
        intervals[name] = (first_write[name], lr + 1)
    return intervals


def plan_offsets(
    intervals: dict[str, tuple[int, int]],
    sizes: dict[str, int],
    aligns: dict[str, int],
) -> tuple[dict[str, int], int]:
    """Assign each scratch buffer a byte offset into one slab so that buffers with
    overlapping live intervals never share a byte. Greedy-by-size: place the
    largest buffers first (they constrain the layout most), each at the lowest
    aligned offset that collides with no already-placed *overlapping-interval*
    buffer. Returns ``({name: offset}, total_bytes)``.

    Deterministic: buffers are ordered by ``(-size, name)`` so the same graph +
    sizes always yields the same layout (captured graphs bake these offsets).
    """
    order = sorted(intervals, key=lambda n: (-sizes[n], n))
    placed: list[tuple[int, int, int, int]] = []  # (offset, size, first, free)
    offsets: dict[str, int] = {}
    total = 0
    for name in order:
        size = sizes[name]
        first, free = intervals[name]
        align = max(1, aligns.get(name, 1))
        # Byte ranges occupied by buffers whose live interval overlaps this one,
        # sorted by start so we can sweep a candidate offset forward past them.
        occ = sorted((o, o + s) for (o, s, f, fr) in placed if first < fr and f < free)
        off = 0
        for lo, hi in occ:
            if off + size <= lo:
                break  # fits in the gap before this occupied range
            if off < hi:
                off = ((hi + align - 1) // align) * align  # bump past it, realigned
        offsets[name] = off
        placed.append((off, size, first, free))
        total = max(total, off + size)
    return offsets, total
