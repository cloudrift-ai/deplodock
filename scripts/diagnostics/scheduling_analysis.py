#!/usr/bin/env python3
"""SASS scheduling-quality analysis for SGEMM kernels.

Tier-1 experiments for the article's "PTX is for Noobs: SASS Deep Dive"
section. Given a compiled binary (or cached cubin), parses the SASS for
the matmul kernel and answers three concrete questions:

1. **LDS-to-consumer-FMA distance.** For each `LDS.*` instruction, how
   many FFMAs (and total instructions) sit between the load and the
   first FFMA that consumes the loaded register? This tells you whether
   LDS latency is hidden behind compute. Average ~40+ = good schedule;
   average ~5 = LDS-bound and there's room to improve.

2. **LDS-to-next-LDS spacing.** How many FFMAs between consecutive LDS
   instructions? If they're clumped (all LDS at top of loop), the LDS
   pipeline backs up. If interleaved ~1 LDS per N FFMAs, the pipeline
   stays full.

3. **Inner-loop excerpt.** Extract the hot loop body (the BRA-back
   region with the FFMA cluster) so it can be quoted in the article.

Usage:
    python scripts/diagnostics/scheduling_analysis.py <binary> [--kernel NAME]

The default kernel name is `fused_matmul`. For cuBLAS JIT cubins from
~/.nv/ComputeCache, pass `--kernel cutlass_80_simt_sgemm_256x128_8x4` (or
whatever symbol is in the cubin — the script will fuzzy-match).
"""

from __future__ import annotations

import argparse
import collections
import re
import statistics
import subprocess
import sys
from pathlib import Path

# A SASS line from cuobjdump looks like:
#   /*0e50*/   FFMA R20, R68, R92.reuse, R9 ;     /* 0x0000005c44147223 */
#                                                  /* 0x081fe20000000009 */
# We want to extract: address (0e50), predicate (none here), mnemonic (FFMA),
# and the operands so we can identify dst/src registers.
LINE_RE = re.compile(
    r"/\*\s*([0-9a-fA-F]+)\s*\*/\s+(@!?P[T0-7]\s+)?([A-Z][A-Z0-9_.]*)\s+(.*?);"
)
HEX_RE = re.compile(r"/\*\s*0x([0-9a-fA-F]+)\s*\*/")
# Register references in SASS: R0..R255 (RZ = R255 = always zero). We only
# care about general-purpose Rn here, not predicate or uniform regs.
REG_RE = re.compile(r"\bR(\d+)(?:\.\w+)?")


def _run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {res.stderr}")
    return res.stdout


def extract_kernel_sass(binary: Path, kernel_substring: str) -> list[tuple[int, str, str]]:
    """Return [(address, mnemonic, operands)] for the matched kernel."""
    raw = _run(["cuobjdump", "--dump-sass", str(binary)])
    out: list[tuple[int, str, str]] = []
    in_kernel = False
    for line in raw.splitlines():
        stripped = line.lstrip()
        # Function header lines look like:  Function : _Z12fused_matmul...
        if stripped.startswith("Function :") or stripped.startswith(".text."):
            in_kernel = kernel_substring in line
            continue
        if not in_kernel:
            continue
        m = LINE_RE.search(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        mnem = m.group(3)
        ops = m.group(4).strip()
        out.append((addr, mnem, ops))
    return out


def parse_dst_src(mnem: str, ops: str) -> tuple[set[int], set[int]]:
    """Best-effort parse: return (set of dst regs, set of src regs).

    The dst-vs-src split depends on the mnemonic. We handle the families
    that matter for FMA-pipe scheduling analysis (FFMA, LDS, STG, IMAD,
    IADD, MOV, LEA, etc.); for everything else we conservatively put the
    first register operand in dst and the rest in src.

    LDS.128 / LDS.64 widen the dst set: a `.128` load writes 4 consecutive
    registers, a `.64` writes 2. We approximate this by adding the next
    consecutive register IDs.
    """
    regs = [int(r) for r in REG_RE.findall(ops)]
    if not regs:
        return set(), set()

    family = mnem.split(".")[0]
    width_suffix = mnem.split(".")[-1] if "." in mnem else ""

    if family in {
        "FFMA", "FADD", "FMUL", "FMNMX",
        "IMAD", "IADD", "IADD3", "ISUB",
        "LEA", "SHF", "LOP3",
        "HFMA2",
    }:
        # First reg is dst, rest are sources.
        dst = {regs[0]}
        src = set(regs[1:])
    elif family in {"MOV", "UMOV"}:
        dst = {regs[0]}
        src = set(regs[1:])
    elif family == "LDS":
        # LDS.128 R72, [R8.X4+0x80]   →  dst = R72..R75, src = R8
        dst_base = regs[0]
        if width_suffix == "128":
            dst = {dst_base, dst_base + 1, dst_base + 2, dst_base + 3}
        elif width_suffix == "64":
            dst = {dst_base, dst_base + 1}
        else:
            dst = {dst_base}
        src = set(regs[1:])
    elif family == "LDC":
        # LDC R10, c[0x0][0x37c]
        dst = {regs[0]}
        src = set()
    elif family in {"LDG", "LD", "UTMALDG"}:
        # We don't track these — TMA writes smem directly, no register dst.
        dst = set()
        src = set(regs)
    elif family in {"STG", "STS", "ST"}:
        # All registers are sources (stored values + addresses).
        dst = set()
        src = set(regs)
    elif family in {"ISETP", "FSETP", "PLOP3"}:
        # Predicate dst (we don't track), reg srcs only.
        dst = set()
        src = set(regs)
    elif family in {"BRA", "EXIT", "BAR", "BSYNC", "BSSY", "RET", "CALL", "FENCE", "MEMBAR"}:
        dst = set()
        src = set()
    elif family == "CS2R":
        # CS2R R10, SR_CLOCKLO
        dst = {regs[0]}
        src = set()
    elif family == "S2R":
        dst = {regs[0]}
        src = set()
    else:
        # Conservative fallback.
        dst = {regs[0]}
        src = set(regs[1:])

    return dst, src


def lds_to_consumer_distances(insns: list[tuple[int, str, str]]) -> list[dict]:
    """For each LDS.*, find first FFMA that uses any of its dst regs.

    Returns one dict per LDS instruction:
        {
            "addr": int,
            "mnem": str,         # "LDS.128" etc.
            "dst_regs": set[int],
            "ffma_distance": int | None,  # FFMAs between LDS and first consumer
            "instr_distance": int | None, # total instructions between
            "consumer_addr": int | None,
        }
    Distance None means we walked to the end of the basic block without
    finding a consumer (the loaded value is consumed in a later iteration
    after the loop closes).
    """
    out = []
    for i, (addr, mnem, ops) in enumerate(insns):
        if not mnem.startswith("LDS"):
            continue
        dst, _ = parse_dst_src(mnem, ops)
        if not dst:
            continue
        # Walk forward looking for an FFMA that consumes any of these regs.
        ffma_count = 0
        consumer = None
        for j in range(i + 1, len(insns)):
            jaddr, jmnem, jops = insns[j]
            jdst, jsrc = parse_dst_src(jmnem, jops)
            if jmnem == "FFMA":
                if jsrc & dst:
                    consumer = (jaddr, j - i)
                    break
                ffma_count += 1
            # If a later instruction overwrites our LDS dst before any FFMA
            # consumes it, the load is dead — count as "no consumer found".
            if jdst & dst and jmnem != "LDS":
                break
            # Bail at next BRA (loop boundary) — we don't follow control flow.
            if jmnem.startswith("BRA"):
                break
        out.append({
            "addr": addr,
            "mnem": mnem,
            "dst_regs": dst,
            "ffma_distance": ffma_count if consumer else None,
            "instr_distance": consumer[1] if consumer else None,
            "consumer_addr": consumer[0] if consumer else None,
        })
    return out


def lds_to_next_lds_spacing(insns: list[tuple[int, str, str]]) -> list[int]:
    """For each LDS.*, count FFMAs between it and the next LDS.* instruction."""
    out: list[int] = []
    last_lds_idx = None
    last_lds_ffma_count = 0
    for i, (_, mnem, _) in enumerate(insns):
        if mnem.startswith("LDS"):
            if last_lds_idx is not None:
                out.append(last_lds_ffma_count)
            last_lds_idx = i
            last_lds_ffma_count = 0
        elif mnem == "FFMA":
            if last_lds_idx is not None:
                last_lds_ffma_count += 1
    return out


def find_inner_loop(insns: list[tuple[int, str, str]]) -> tuple[int, int] | None:
    """Find the (start_idx, end_idx) of the densest FFMA-cluster region.

    Heuristic: locate the largest contiguous run of mostly-FFMA instructions
    (>=80% FFMA in a sliding window). That's the inner k-loop body.
    """
    n = len(insns)
    if n < 50:
        return None
    is_ffma = [1 if mnem == "FFMA" else 0 for _, mnem, _ in insns]
    # Sliding window of 100 instructions, find the window with the highest
    # FFMA density. Then expand outward while density stays >50%.
    win = 100
    best_start = 0
    best_density = 0.0
    for i in range(0, n - win):
        d = sum(is_ffma[i:i + win]) / win
        if d > best_density:
            best_density = d
            best_start = i
    if best_density < 0.5:
        return None
    # Expand the window outward.
    start = best_start
    while start > 0 and is_ffma[start - 1]:
        start -= 1
    end = best_start + win
    while end < n and is_ffma[end]:
        end += 1
    return (start, end)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("binary", type=Path, help="Compiled binary or .cubin file")
    p.add_argument("--kernel", default="fused_matmul",
                   help="Substring of kernel symbol name to analyze")
    p.add_argument("--inner-loop-lines", type=int, default=40,
                   help="Number of inner-loop SASS lines to dump")
    args = p.parse_args()

    print(f"# Scheduling analysis: {args.binary} kernel='{args.kernel}'")
    print()

    insns = extract_kernel_sass(args.binary, args.kernel)
    if not insns:
        print(f"ERROR: no SASS lines for kernel matching '{args.kernel}' found in {args.binary}", file=sys.stderr)
        sys.exit(1)
    print(f"## Total SASS instructions in matched kernel: {len(insns)}")
    print()

    # Mnemonic histogram (top 10) for orientation.
    mnem_count = collections.Counter(m for _, m, _ in insns)
    print("## Top mnemonics")
    print()
    print("| Mnemonic | Count |")
    print("|---|---:|")
    for mnem, cnt in mnem_count.most_common(10):
        print(f"| `{mnem}` | {cnt} |")
    print()

    # Tier 1 experiment 1: LDS-to-consumer distance.
    distances = lds_to_consumer_distances(insns)
    found = [d for d in distances if d["ffma_distance"] is not None]
    print(f"## LDS-to-first-FFMA-consumer distance ({len(distances)} LDS instructions, {len(found)} with consumer)")
    print()
    if found:
        ffma_d = [d["ffma_distance"] for d in found]
        instr_d = [d["instr_distance"] for d in found]
        print(f"- FFMAs between LDS and first consuming FFMA: "
              f"min={min(ffma_d)}, median={statistics.median(ffma_d):.0f}, "
              f"mean={statistics.mean(ffma_d):.1f}, max={max(ffma_d)}")
        print(f"- Total instructions: "
              f"min={min(instr_d)}, median={statistics.median(instr_d):.0f}, "
              f"mean={statistics.mean(instr_d):.1f}, max={max(instr_d)}")
        # Histogram bucketed
        buckets = [0, 5, 10, 20, 40, 80, 160, 320]
        hist = [0] * (len(buckets) + 1)
        for v in ffma_d:
            placed = False
            for i, b in enumerate(buckets):
                if v < b:
                    hist[i] += 1
                    placed = True
                    break
            if not placed:
                hist[-1] += 1
        print()
        print("| FFMAs between LDS and consumer | Count |")
        print("|---|---:|")
        prev = 0
        for i, b in enumerate(buckets):
            print(f"| [{prev}, {b}) | {hist[i]} |")
            prev = b
        print(f"| [{prev}, ∞) | {hist[-1]} |")
    no_consumer = [d for d in distances if d["ffma_distance"] is None]
    if no_consumer:
        print()
        print(f"  ({len(no_consumer)} LDS instructions had no FFMA consumer in the same basic block — "
              f"either dead loads, loop-carry across iterations, or boundary effects.)")
    print()

    # Tier 1 experiment 2: LDS-to-next-LDS spacing.
    spacing = lds_to_next_lds_spacing(insns)
    print(f"## LDS-to-next-LDS spacing (FFMAs between consecutive LDS, {len(spacing)} pairs)")
    print()
    if spacing:
        print(f"- min={min(spacing)}, median={statistics.median(spacing):.0f}, "
              f"mean={statistics.mean(spacing):.1f}, max={max(spacing)}")
        print()
        # Histogram
        buckets = [0, 1, 2, 4, 8, 16, 32]
        hist = [0] * (len(buckets) + 1)
        for v in spacing:
            placed = False
            for i, b in enumerate(buckets):
                if v < b:
                    hist[i] += 1
                    placed = True
                    break
            if not placed:
                hist[-1] += 1
        print("| FFMAs between consecutive LDS | Count |")
        print("|---|---:|")
        prev = 0
        for i, b in enumerate(buckets):
            print(f"| [{prev}, {b}) | {hist[i]} |")
            prev = b
        print(f"| [{prev}, ∞) | {hist[-1]} |")
    print()

    # Tier 1 experiment 3: inner loop excerpt.
    loop = find_inner_loop(insns)
    if loop:
        start, end = loop
        density = sum(1 for _, m, _ in insns[start:end] if m == "FFMA") / (end - start)
        print(f"## Inner loop excerpt (instructions {start}..{end}, FFMA density {density:.1%})")
        print()
        excerpt_start = max(0, start - 5)
        excerpt_end = min(len(insns), excerpt_start + args.inner_loop_lines)
        print("```")
        for addr, mnem, ops in insns[excerpt_start:excerpt_end]:
            print(f"  /*{addr:04x}*/  {mnem:12} {ops}")
        print("```")
        print()
        # Summary of just the loop body
        loop_mnems = collections.Counter(m for _, m, _ in insns[start:end])
        print(f"### Inner-loop mnemonic counts (full body, {end - start} instructions)")
        print()
        print("| Mnemonic | Count |")
        print("|---|---:|")
        for mnem, cnt in loop_mnems.most_common(8):
            print(f"| `{mnem}` | {cnt} |")
    print()


if __name__ == "__main__":
    main()
