#!/usr/bin/env python3
"""SASS instruction histogram and stall analysis for TMA kernel vs cuBLAS.

Reproduces the instruction counts and control-code analysis in the article's
"PTX is for Noobs: SASS Deep Dive" section. The flow:

1. Lower the `tma_db` strategy at a representative size and batch.
2. Compile the bench binary just like `run_benchmark` does.
3. `cuobjdump --dump-sass` the binary, find the `fused_matmul` symbol, and
   count opcodes by family (FFMA, LDS, STG, CS2R, UTMALDG, ISETP, ...).
4. Parse the hex encoding bytes (`nvdisasm --print-instruction-encoding`) to
   extract per-instruction stall counts and yield flags from the SASS control
   word — sm_120 uses the same 128-bit (64-bit insn + 64-bit ctrl) layout as
   sm_90.
5. Repeat steps 3 for cuBLAS by extracting PTX from libcublasLt's
   `cutlass_80_simt_sgemm_forwardCompat_*` kernel and counting PTX opcodes.

Output is plain text with two histogram tables and one stall summary, sized
to match the article's tables. Run as:

    python scripts/diagnostics/sass_analysis.py [SIZE]   # default 8192

Note: cuBLAS PTX extraction reads `libcublasLt.so` and is best-effort — if the
SGEMM kernel isn't shipped as PTX in your install (some cuBLAS builds ship only
cubins), the cuBLAS column will say "PTX not embedded".
"""

from __future__ import annotations

import collections
import dataclasses
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from deplodock.compiler.cuda.codegen import emit_kernel  # noqa: E402
from deplodock.compiler.cuda.lower import lower_graph  # noqa: E402
from deplodock.compiler.cuda.runner import _detect_arch, generate_benchmark_program  # noqa: E402
from deplodock.compiler.cuda.tuning import default_matmul_strategy_map  # noqa: E402
from deplodock.compiler.ir import Graph, Tensor  # noqa: E402
from deplodock.compiler.ops import FusedReduceElementwiseOp, InputOp  # noqa: E402
from deplodock.compiler.rewriter import Rewriter  # noqa: E402

# Opcode families we care about for the histogram. The article groups by these.
# Pattern → display name. Each pattern is matched as a prefix on the SASS
# mnemonic (the first whitespace-separated token after the address column).
# (label, list of mnemonic prefixes the family covers).
SASS_FAMILIES: list[tuple[str, list[str]]] = [
    ("FFMA*       (fused multiply-add — the actual compute)", ["FFMA"]),
    ("LDS.*       (shared-memory loads, incl. .128 vector form)", ["LDS"]),
    ("STS.*       (shared-memory stores)", ["STS"]),
    ("STG.*       (global stores, incl. .E.128 vector form)", ["STG"]),
    ("LDG.*       (global loads — should be 0 for TMA)", ["LDG"]),
    ("LDC.*       (constant loads — kernel params, TMA descriptors)", ["LDC"]),
    ("UTMALDG.*   (TMA load commands)", ["UTMALDG"]),
    ("CS2R/S2R    (special-register reads)", ["CS2R", "S2R"]),
    ("ISETP.*     (integer set-predicate, bounds + loop control)", ["ISETP"]),
    ("BAR/BSYNC/BSSY (block barriers + reconvergence)", ["BAR", "BSYNC", "BSSY", "BRX"]),
    ("MBARRIER    (mbarrier intrinsics)", ["MEMBAR", "ARRIVES", "ARRIVE"]),
    ("MOV/IMAD/IADD/LEA (address arithmetic, reg copies)", ["MOV", "UMOV", "IMAD", "IADD", "LEA", "SHF", "ULEA", "UIADD", "UIMAD", "ISUB"]),
    ("LOP3/PLOP3  (logic + predicate logic)", ["LOP3", "PLOP3", "ULOP3"]),
    ("HFMA2       (FP16x2 helpers — typically address calc)", ["HFMA2"]),
    ("BRA/EXIT/CALL (branches)", ["BRA", "EXIT", "CALL", "RET"]),
    ("FENCE/MEMBAR (memory ordering)", ["FENCE", "MEMBAR"]),
]


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def make_matmul_graph() -> Graph:
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", ("M", "K")))
    b = g.add_node(InputOp(), [], Tensor("B", ("K", "N")))
    c = g.add_node(FusedReduceElementwiseOp("sum", "mul", 1), [a, b], Tensor("C", ("M", "N")))
    g.inputs = [a, b]
    g.outputs = [c]
    return g


def compile_tma_bench(size: int, tmpdir: Path) -> Path:
    """Compile the TMA-DB bench binary at the given square size, return its path."""
    strategy_map, profile = default_matmul_strategy_map()
    selected = strategy_map[-1][1]
    for thr, cfg in strategy_map:
        if size <= thr:
            selected = cfg
            break
    print(
        f"# profile: {profile}, config for {size}: TM={selected.thread_m}, BK={selected.block_k}, ks={selected.k_splits}", file=sys.stderr
    )

    g = make_matmul_graph()
    kernel = lower_graph(Rewriter().apply(g.copy()), config=selected)
    src = emit_kernel(kernel)

    dim_args: dict[str, int] = {"M": size, "N": size, "K": size}
    if selected.k_splits > 1:
        dim_args["k_splits"] = selected.k_splits
    program = generate_benchmark_program(
        src,
        kernel,
        dim_args,
        num_iterations=2,
        compare_cublas=True,
        coarsen_cols=selected.coarsen_cols,
        coarsen_rows=selected.coarsen_rows,
        cublas_math_mode="default",
    )

    src_path = tmpdir / "bench.cu"
    bin_path = tmpdir / "bench"
    src_path.write_text(program)

    arch = _detect_arch() or "sm_120"
    cmd = ["nvcc", "-O3", "--fmad=true", "-arch", arch, "-lcuda", "-lcublas", "-lcurand", "-o", str(bin_path), str(src_path)]
    res = _run(cmd, timeout=300)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise RuntimeError("nvcc compilation failed — see stderr above")
    return bin_path


# A SASS line from cuobjdump looks like:
#   /*0080*/   FFMA.FTZ R7, R10, R11, R7;
#   /*ff20*/   @!P4 STG.E desc[UR4][R4.64+0x4], R178;
#   /*0420*/   @P0 BRA 0xcc0;
# We want the mnemonic (FFMA.FTZ / STG.E / BRA), skipping any leading predicate
# like `@P0`, `@!P4`, `@PT`. The predicate token is `@!?P[T0-7]`.
SASS_LINE_RE = re.compile(r"/\*[0-9a-fA-F]+\*/\s+(@!?P[T0-7]\s+)?([A-Z][A-Z0-9_.]*)")


def extract_kernel_sass(binary: Path, kernel_name: str) -> list[str]:
    """Return the SASS lines for one kernel symbol from the bench binary."""
    res = _run(["cuobjdump", "--dump-sass", str(binary)])
    if res.returncode != 0:
        raise RuntimeError(f"cuobjdump failed: {res.stderr}")
    lines = res.stdout.splitlines()
    out: list[str] = []
    in_kernel = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("Function :") or stripped.startswith(".text."):
            in_kernel = kernel_name in line
            continue
        if in_kernel and SASS_LINE_RE.search(line):
            out.append(line.rstrip())
    return out


def histogram(sass_lines: list[str]) -> dict[str, int]:
    counts: dict[str, int] = collections.Counter()
    for line in sass_lines:
        m = SASS_LINE_RE.search(line)
        if not m:
            continue
        mnem = m.group(2)
        counts[mnem] += 1
    return dict(counts)


def family_totals(counts: dict[str, int]) -> list[tuple[str, int]]:
    """Group the raw mnemonic counts by the families defined above."""
    matched: set[str] = set()
    rows = []
    for label, prefixes in SASS_FAMILIES:
        total = 0
        for mnem, c in counts.items():
            if any(mnem == p or mnem.startswith(p + ".") for p in prefixes):
                total += c
                matched.add(mnem)
        if total > 0:
            rows.append((label, total))
    other_mnems = [m for m in counts if m not in matched]
    other = sum(counts[m] for m in other_mnems)
    if other > 0:
        rows.append((f"OTHER ({len(other_mnems)} mnemonics — see raw histogram)", other))
    return rows


# --- Stall code parsing -----------------------------------------------------
#
# On sm_90+ each instruction is a 16-byte (128-bit) word: the lower 8 bytes are
# the instruction encoding and the upper 8 bytes hold the control word. The
# control word's lowest 4 bits are the stall count (0-15 cycles before issuing
# the next instruction), bit 4 is the yield flag. We parse the hex encoding
# emitted by `nvdisasm --print-instruction-encoding`.
#
# nvdisasm prints something like:
#     /*0080*/  /* 0x0000000a070b070a */  FFMA.FTZ R7, R10, R11, R7 ;
#                /* 0x000fc40000000a07 */
# The second hex word is the control. Bits [0:3] = stall, bit [4] = yield.


@dataclasses.dataclass
class StallStats:
    by_mnemonic: dict[str, list[int]]  # mnemonic → list of stall counts
    yield_flags: dict[str, int]  # mnemonic → count of yield=1
    raw_control_words: list[tuple[str, int]]  # (mnemonic, full 64-bit ctrl word)


def parse_stalls(binary: Path, kernel_name: str, work_dir: Path) -> StallStats | None:
    """Best-effort: dump nvdisasm with hex and parse stall + yield bits.

    Caveat: the SASS control-word bit layout for sm_120 is not publicly
    documented. The parser below assumes the same layout as sm_90 (Hopper):
    bits [0:3] stall count, bit [4] yield. The numbers may be off if NVIDIA
    changed the layout for Blackwell. The raw 64-bit control words are
    dumped alongside so a reader can re-parse if they have better info.
    """
    # cuobjdump --extract-elf writes to the CWD; run it from a fresh dir.
    extract = _run(
        ["cuobjdump", "--extract-elf", "all", str(binary)],
        cwd=str(work_dir),
    )
    cubins = sorted(work_dir.glob("*.sm_*.cubin"))
    if extract.returncode != 0 or not cubins:
        return None
    # Pick the largest cubin (the actual code, not the metadata stub).
    elf = max(cubins, key=lambda p: p.stat().st_size)

    res = _run(["nvdisasm", "--print-instruction-encoding", str(elf)])
    if res.returncode != 0:
        return None

    by_mnem: dict[str, list[int]] = collections.defaultdict(list)
    yields: dict[str, int] = collections.Counter()
    raw_ctrl: list[tuple[str, int]] = []
    in_kernel = False
    pending_mnem: str | None = None

    def _record(mnem: str, ctrl: int):
        by_mnem[mnem].append(ctrl & 0xF)
        if (ctrl >> 4) & 0x1:
            yields[mnem] += 1
        raw_ctrl.append((mnem, ctrl))

    for line in res.stdout.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("Function :") or stripped.startswith(".text."):
            in_kernel = kernel_name in line
            pending_mnem = None
            continue
        if not in_kernel:
            continue

        m_insn = SASS_LINE_RE.search(line)
        # The nvdisasm output for each insn is two lines:
        #   /*0e50*/   FFMA R20, R68, R92, R9 ;   /* 0xINSN_ENCODING */
        #                                          /* 0xCONTROL_WORD */
        # We capture the mnemonic from the first line, then look for the
        # _next_ hex literal that appears on its own (no mnemonic on the line).
        if m_insn:
            pending_mnem = m_insn.group(2)
            hex_matches = re.findall(r"/\*\s*0x([0-9a-fA-F]+)\s*\*/", line)
            if len(hex_matches) >= 2 and pending_mnem is not None:
                _record(pending_mnem, int(hex_matches[1], 16))
                pending_mnem = None
        else:
            hex_match = re.search(r"/\*\s*0x([0-9a-fA-F]+)\s*\*/", line)
            if hex_match and pending_mnem is not None:
                _record(pending_mnem, int(hex_match.group(1), 16))
                pending_mnem = None

    if not by_mnem:
        return None
    return StallStats(by_mnemonic=dict(by_mnem), yield_flags=dict(yields), raw_control_words=raw_ctrl)


# --- cuBLAS PTX side ---------------------------------------------------------


def find_libcublas() -> Path | None:
    for p in [
        Path("/usr/local/cuda/lib64/libcublasLt.so"),
        Path("/usr/local/cuda/lib64/libcublas.so"),
    ]:
        if p.exists():
            return p
    return None


CUBLAS_PTX_FAMILIES = [
    ("fma.", "fma.rn.f32        (FP32 multiply-add)"),
    ("cp.async", "cp.async          (LDGSTS cooperative load)"),
    ("st.shared", "st.shared         (smem store)"),
    ("ld.shared", "ld.shared         (smem load)"),
    ("ld.global", "ld.global         (global load)"),
    ("st.global", "st.global         (global store)"),
    ("bar.sync", "bar.sync          (__syncthreads)"),
    ("setp.", "setp.*            (predicate set)"),
    ("mov.", "mov.*             (register copies)"),
    ("add.", "add.*             (integer + FP add)"),
]


def dump_cublas_ptx_for_simt_sgemm(libpath: Path) -> str | None:
    """Return PTX text for the cutlass_80_simt_sgemm forwardCompat kernel, if shipped."""
    res = _run(["cuobjdump", "--dump-ptx", str(libpath)])
    if res.returncode != 0 or not res.stdout:
        return None
    text = res.stdout
    # Find a PTX module containing simt_sgemm. Modules are separated by
    # `// Compiling entry function ...` headers in cuobjdump output.
    needle = "cutlass_80_simt_sgemm"
    if needle not in text:
        return None
    # Slice from the line containing the kernel name backwards to the
    # previous `.entry` and forwards to the next `}` matching the entry block.
    idx = text.find(needle)
    start = text.rfind(".entry", 0, idx)
    if start < 0:
        start = max(0, idx - 4096)
    # Find the end of the .entry's body — naive brace matching.
    depth = 0
    end = len(text)
    started = False
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
            started = True
        elif ch == "}":
            depth -= 1
            if started and depth == 0:
                end = i + 1
                break
    return text[start:end]


def cublas_ptx_histogram(ptx: str) -> list[tuple[str, int]]:
    rows = []
    for prefix, label in CUBLAS_PTX_FAMILIES:
        # Count lines whose first non-whitespace token starts with the prefix.
        n = sum(1 for line in ptx.splitlines() if line.strip().startswith(prefix))
        if n > 0:
            rows.append((label, n))
    return rows


# --- Top-level driver --------------------------------------------------------


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 8192

    with tempfile.TemporaryDirectory(prefix="deplodock_sass_") as td:
        tmpdir = Path(td)
        print(f"# === SASS analysis for tma_db @ {size}x{size}x{size}, batch=1 ===\n")

        binary = compile_tma_bench(size, tmpdir)

        sass = extract_kernel_sass(binary, "fused_matmul")
        print(f"# fused_matmul: {len(sass)} SASS lines\n")

        counts = histogram(sass)
        print("## Our TMA kernel — opcode histogram (by family)")
        print()
        print("| Family | Count |")
        print("|---|---:|")
        for label, n in family_totals(counts):
            print(f"| `{label}` | {n} |")
        print()

        print("## Our TMA kernel — top mnemonics (raw, top 15)")
        print()
        print("| Mnemonic | Count |")
        print("|---|---:|")
        for mnem, n in sorted(counts.items(), key=lambda kv: -kv[1])[:15]:
            print(f"| `{mnem}` | {n} |")
        print()

        stalls = parse_stalls(binary, "fused_matmul", tmpdir)
        if stalls is not None:
            print("## Stall counts (best-effort, control-word low 4 bits)")
            print()
            print("> ⚠️ The SASS control-word bit layout for sm_120 (Blackwell) is not")
            print("> publicly documented. The numbers below assume the same layout as")
            print("> sm_90 (bits [0:3] = stall count, bit [4] = yield), which may not")
            print("> hold on Blackwell. Treat these as suggestive, not authoritative.")
            print("> Raw 64-bit control words for the first 20 FFMA instructions are")
            print("> dumped at the end so you can re-derive the layout if you have")
            print("> better information.")
            print()
            print("| Mnemonic | N | min | median | max | yield count |")
            print("|---|---:|---:|---:|---:|---:|")
            for mnem in sorted(stalls.by_mnemonic, key=lambda m: -len(stalls.by_mnemonic[m]))[:15]:
                vs = stalls.by_mnemonic[mnem]
                vs_sorted = sorted(vs)
                med = vs_sorted[len(vs) // 2]
                print(f"| `{mnem}` | {len(vs)} | {min(vs)} | {med} | {max(vs)} | {stalls.yield_flags.get(mnem, 0)} |")
            print()
            print("### Raw control words for the first 20 FFMA instructions")
            print()
            print("```")
            ffma_ctrls = [(m, c) for m, c in stalls.raw_control_words if m == "FFMA"][:20]
            for mnem, ctrl in ffma_ctrls:
                print(f"  {mnem}  ctrl=0x{ctrl:016x}  bin={ctrl:064b}")
            print("```")
            print()
        else:
            print("## Stall counts: nvdisasm hex parsing failed; raw SASS dumped above for manual inspection.")
            print()

        # cuBLAS side
        libpath = find_libcublas()
        if libpath is not None:
            ptx = dump_cublas_ptx_for_simt_sgemm(libpath)
            if ptx:
                print(f"## cuBLAS — `cutlass_80_simt_sgemm_*` PTX histogram (from {libpath.name})")
                print()
                print("| Family | Count |")
                print("|---|---:|")
                for label, n in cublas_ptx_histogram(ptx):
                    print(f"| `{label}` | {n} |")
                print()
            else:
                print("## cuBLAS PTX: not embedded in this libcublas build (only cubin shipped)\n")
        else:
            print("## cuBLAS PTX: libcublas not found\n")


if __name__ == "__main__":
    main()
