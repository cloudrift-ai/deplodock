"""Stream-K MAC-unit work distribution — the pure arithmetic (Phase B2).

The adaptive Stream-K kernel walks **MAC units**, not whole tiles. A MAC unit is
one ``(output tile, K-chunk)`` pair: the total work is
``M_blocks · N_blocks · K_blocks`` units, where ``K_blocks = K / BK`` is the
K-loop trip count. Each of ``num_sms`` CTAs is handed one contiguous
``[start, end)`` slice of that unit space (the same ceil-div partition the
non-adaptive path uses, see ``_streamk_ranges`` in the CUDA backend).

Units are ordered **K-minor within a tile** (``mac = tile_id · K_blocks +
k_chunk``, ``tile_id = m_b · N_blocks + n_b``), so a CTA's contiguous slice maps
to: an optional *head partial* (the tail K-chunks of the tile it starts inside),
zero or more *full tiles*, and an optional *tail partial* (the head K-chunks of
the tile it stops inside). Full tiles are written directly to the output; the ≤ 2
boundary partials per CTA are summed by the atomic-free combine kernel (B4).

This module is the single source of truth for that walk: ``cta_segments`` is the
reference model the kernel codegen (B3) implements as ``Expr`` arithmetic, and
the tests pin its coverage / one-owner / boundary properties. Underscore-prefixed
so the pass loader (globs ``*.py``, skips ``_``) doesn't mistake it for a rule.
"""

from __future__ import annotations

from dataclasses import dataclass


def total_mac_units(m_blocks: int, n_blocks: int, k_blocks: int) -> int:
    """Total MAC work units = output tiles × K-chunks per tile."""
    return m_blocks * n_blocks * k_blocks


def per_cta(total_units: int, num_sms: int) -> int:
    """Units assigned to each CTA — ceil-div so ``num_sms`` CTAs cover the whole
    range in one wave's worth of contiguous slices."""
    if num_sms <= 0:
        return total_units
    return -(-total_units // num_sms)  # ceil-div


def cta_range(cta: int, total_units: int, num_sms: int) -> tuple[int, int]:
    """The ``[start, end)`` MAC-unit slice owned by ``cta``. Clamped to
    ``total_units`` so trailing CTAs past the work get an empty range and idle —
    matching ``_streamk_ranges`` in the backend."""
    pc = per_cta(total_units, num_sms)
    start = min(cta * pc, total_units)
    end = min((cta + 1) * pc, total_units)
    return start, end


def decode_mac(mac: int, n_blocks: int, k_blocks: int) -> tuple[int, int, int]:
    """``mac → (m_b, n_b, k_chunk)``. K-minor within a row-major tile id."""
    tile_id, k_chunk = divmod(mac, k_blocks)
    m_b, n_b = divmod(tile_id, n_blocks)
    return m_b, n_b, k_chunk


@dataclass(frozen=True)
class Segment:
    """One contiguous K-sub-range of one output tile processed by a CTA.

    ``[k_lo, k_hi)`` are K-chunk indices; ``is_full`` ⇒ the CTA owns the tile's
    entire K range (``[0, K_blocks)``) and writes the output directly. Otherwise
    it's a boundary partial that goes to the combine workspace.
    """

    m_b: int
    n_b: int
    k_lo: int
    k_hi: int
    is_full: bool


def cta_segments(cta: int, m_blocks: int, n_blocks: int, k_blocks: int, num_sms: int) -> list[Segment]:
    """Decompose ``cta``'s MAC-unit slice into per-tile K-sub-range segments.

    Walks ``[start, end)`` tile by tile: at each step the current ``mac`` sits at
    K-chunk ``k_lo`` of some tile; the segment runs until the tile's K ends or the
    CTA's budget does, whichever comes first. The reference the kernel emits as
    ``Expr`` arithmetic in B3.
    """
    start, end = cta_range(cta, total_mac_units(m_blocks, n_blocks, k_blocks), num_sms)
    segments: list[Segment] = []
    mac = start
    while mac < end:
        m_b, n_b, k_lo = decode_mac(mac, n_blocks, k_blocks)
        tile_end_mac = mac - k_lo + k_blocks  # mac where this tile's K-chunks end
        seg_end = min(end, tile_end_mac)
        k_hi = k_lo + (seg_end - mac)
        segments.append(Segment(m_b=m_b, n_b=n_b, k_lo=k_lo, k_hi=k_hi, is_full=(k_lo == 0 and k_hi == k_blocks)))
        mac = seg_end
    return segments
