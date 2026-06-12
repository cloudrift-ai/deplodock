"""Span-splitting for vLLM's packed batches. Pure Python — no torch/vllm imports."""

from __future__ import annotations


def split_spans(positions, max_seq_len: int) -> list[tuple[int, int]]:
    """Split a packed batch into per-sequence ``(start, end)`` spans.

    vLLM V1 hands pooling models one flattened token tensor with per-request
    0-based ``positions``; with chunked prefill disabled every request arrives
    whole, so a new sequence starts exactly where ``positions[i] == 0``.

    Hardened for vLLM's ``_dummy_run`` batches (garbage positions during
    memory profiling): index 0 always starts a span even when
    ``positions[0] != 0``, and any span longer than ``max_seq_len`` is chopped
    into ``max_seq_len``-sized chunks — the output is garbage there, but dummy
    runs only need the forward to not crash.
    """
    n = len(positions)
    if n == 0:
        return []
    starts = [0]
    for i in range(1, n):
        if positions[i] == 0:
            starts.append(i)
    starts.append(n)
    spans: list[tuple[int, int]] = []
    for a, b in zip(starts, starts[1:], strict=False):  # pairwise: zip stops at the shorter
        while b - a > max_seq_len:
            spans.append((a, a + max_seq_len))
            a += max_seq_len
        spans.append((a, b))
    return spans
