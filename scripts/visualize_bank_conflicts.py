"""Visualize smem bank conflicts under different swizzle strategies.

Simulates a warp's 32 lanes accessing a 2D smem stage at one inner-loop
iteration, applies a chosen swizzle formula to the column index, and
plots three panels comparing:

(a) No swizzle (TMA default).
(b) Hardware swizzle (TMA_SWIZZLE=1 with the hardware-fixed XOR).
(c) Per-lane K-rotation (proposed software fix).

Each panel shows:

- A *lane → bank* matrix (32×32). Cell (l, b) is filled if lane ``l``
  lands in bank ``b``. Multiple lanes in one column = conflict.
- A bank histogram (lanes-per-bank).

Defaults match the matmul_add tile shape we've been profiling:
``(rows=128, cols=16)``, lanes index rows at stride 4 (``F_M=4``).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

WARP_SIZE = 32
BANKS = 32


@dataclass
class AccessPattern:
    """``(lane, k_iter) → (row, col)`` mapping for a 2D smem stage."""

    rows: int
    cols: int
    row_fn: Callable[[int, int], int]
    col_fn: Callable[[int, int], int]


def warp_banks(pat: AccessPattern, k_iter: int, swizzle: Callable[[int, int, int], int]) -> list[int]:
    """Per-lane bank id for the warp at iteration ``k_iter``."""
    banks = []
    for lane in range(WARP_SIZE):
        row = pat.row_fn(lane, k_iter)
        col = pat.col_fn(lane, k_iter)
        col_phys = swizzle(lane, row, col)
        addr = row * pat.cols + col_phys
        banks.append(addr % BANKS)
    return banks


def plot_panel(ax_matrix, ax_hist, banks: list[int], title: str, formula: str) -> None:
    """Two stacked subplots: lane→bank matrix, bank histogram."""
    matrix = np.zeros((WARP_SIZE, BANKS), dtype=int)
    for lane, bank in enumerate(banks):
        matrix[lane, bank] = 1
    ax_matrix.imshow(matrix, cmap="Greens", aspect="equal", vmin=0, vmax=1)
    ax_matrix.set_xticks(range(0, BANKS, 4))
    ax_matrix.set_yticks(range(0, WARP_SIZE, 4))
    ax_matrix.set_xlabel("bank id")
    ax_matrix.set_ylabel("lane id")
    ax_matrix.grid(which="major", color="#cccccc", linewidth=0.3, alpha=0.5)
    ax_matrix.set_axisbelow(True)
    ax_matrix.set_title(f"{title}\n{formula}", fontsize=10)

    bank_counts = [0] * BANKS
    for bank in banks:
        bank_counts[bank] += 1
    max_way = max(bank_counts)
    avg_way = sum(c for c in bank_counts if c > 0) / sum(1 for c in bank_counts if c > 0)
    colors = ["#d62728" if c > 1 else "#2ca02c" if c == 1 else "#dddddd" for c in bank_counts]
    ax_hist.bar(range(BANKS), bank_counts, color=colors)
    ax_hist.set_xlim(-0.5, BANKS - 0.5)
    ax_hist.set_ylim(0, max(max_way, 4) + 0.5)
    ax_hist.axhline(1, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    verdict_color = "#d62728" if max_way > 4 else "#ff7f0e" if max_way > 1 else "#2ca02c"
    ax_hist.set_title(
        f"max-way conflict: {max_way}   (avg {avg_way:.1f} lane/bank)",
        fontsize=10,
        color=verdict_color,
        weight="bold",
    )
    ax_hist.set_xlabel("bank id")
    ax_hist.set_ylabel("# lanes")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=128, help="stage rows (BM or BN)")
    p.add_argument("--cols", type=int, default=16, help="stage cols (BK)")
    p.add_argument(
        "--lane-stride",
        type=int,
        default=4,
        help="lanes index rows at this stride (F_M / F_N register-tile factor)",
    )
    p.add_argument("--k-iter", type=int, default=0, help="K-loop iteration to visualize")
    p.add_argument("--out", default="/tmp/bank_conflicts.png", help="output PNG path")
    p.add_argument(
        "--lane-axis",
        choices=("row", "col"),
        default="row",
        help=(
            "Which slab axis the warp lanes stride over. "
            "'row' = a_smem-style (lane*F_M indexes rows, k_iter is col); "
            "'col' = b_smem-style (lane*F_N indexes cols, k_iter is row)."
        ),
    )
    args = p.parse_args()

    if args.lane_axis == "row":
        pat = AccessPattern(
            rows=args.rows,
            cols=args.cols,
            row_fn=lambda lane, k: (lane * args.lane_stride) % args.rows,
            col_fn=lambda lane, k: k,
        )
    else:
        pat = AccessPattern(
            rows=args.rows,
            cols=args.cols,
            row_fn=lambda lane, k: k,
            col_fn=lambda lane, k: (lane * args.lane_stride) % args.cols,
        )

    def no_swizzle(lane: int, row: int, col: int) -> int:
        return col

    def hw_b64_swizzle(lane: int, row: int, col: int) -> int:
        return col ^ ((row & 6) * 2)

    bk_mask = args.cols - 1

    def k_rotation(lane: int, row: int, col: int) -> int:
        # Per-lane rotate. Useful when lanes stride rows: XOR by lane breaks the
        # uniform-bank pattern caused by lane*stride mod 32 = 0.
        return col ^ (lane & bk_mask)

    def col_xor_high(lane: int, row: int, col: int) -> int:
        # Storage-time XOR depending only on col. Drives the b_smem (lanes-
        # stride-cols) pattern to 1-way for any (BN, F_N) where BN ≥ 32 and
        # F_N · 32 ≤ BN. Key idea: col mod 32 repeats every WARP cols, so
        # cols (lane*F_N+c) and ((lane+WARP/F_N)*F_N+c) collide on banks.
        # XOR-ing in (col >> 5) injects col's high bits into the bank, which
        # differ between the colliding lanes (they live in distinct 32-col
        # strips). Bijective per row, so it's safe to apply symmetrically at
        # store time and load time.
        return col ^ (col >> 5)

    panels = [
        ("(a) No swizzle", "col_phys = col", no_swizzle),
        ("(b) HW B64 swizzle", "col_phys = col ⊕ ((row & 6) · 2)", hw_b64_swizzle),
        (
            "(c) Per-lane K-rotation",
            f"col_phys = col ⊕ (lane & {bk_mask})",
            k_rotation,
        ),
        (
            "(d) Col-only XOR swizzle",
            "col_phys = col ⊕ (col >> 5)",
            col_xor_high,
        ),
    ]

    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), gridspec_kw={"height_ratios": [4, 1]})
    for col_idx, (title, formula, swizzle) in enumerate(panels):
        banks = warp_banks(pat, args.k_iter, swizzle)
        plot_panel(axes[0, col_idx], axes[1, col_idx], banks, title, formula)

    axis_desc = "rows" if args.lane_axis == "row" else "cols"
    fig.suptitle(
        f"smem bank conflicts at stage ({args.rows}×{args.cols}), "
        f"lanes index {axis_desc} at stride {args.lane_stride}, k_iter = {args.k_iter}",
        fontsize=12,
    )
    legend = [
        mpatches.Patch(color="#2ca02c", label="1 lane (no conflict)"),
        mpatches.Patch(color="#d62728", label=">1 lane (conflict)"),
        mpatches.Patch(color="#dddddd", label="0 lanes"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
