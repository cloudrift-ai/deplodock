#!/usr/bin/env python3
"""Discover golden matmul configs by autotuning, then regenerate the data module.

For each shape, this: traces ``torch.matmul`` via the shared ``matmul_snippet``,
autotunes it at ``-Xcicc -O1`` (ranking pass) into an isolated SQLite DB, reads
the winning knobs off the assembled ``CudaOp`` graph, re-benches the winner at
``-O3`` (deployable) against cuBLAS (``torch.matmul``, TF32 pinned off), and
records a :class:`MatmulGoldenConfig`. The collected list is spliced into the
``BEGIN/END GENERATED`` region of
``deplodock/compiler/pipeline/search/golden_configs.py``.

Usage:
    python scripts/find_golden_configs.py                       # all shapes
    python scripts/find_golden_configs.py --shapes square.512 --dry-run
    python scripts/find_golden_configs.py --shapes qwen3 --patience 80
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
# Keep the compiler's own INFO chatter out of the way; our progress is the signal.
logging.getLogger("deplodock").setLevel(logging.WARNING)
logger = logging.getLogger("find_golden_configs")

from deplodock import config  # noqa: E402
from deplodock.compiler.pipeline.search.golden_configs import (  # noqa: E402
    QWEN3_06B_HIDDEN,
    QWEN3_06B_INTER,
    QWEN3_06B_KV_DIM,
    QWEN3_06B_Q_DIM,
    MatmulGoldenConfig,
    matmul_snippet,
)

_DATA_MODULE = Path(__file__).resolve().parent.parent / "deplodock" / "compiler" / "pipeline" / "search" / "golden_configs.py"
_BEGIN = "# --- BEGIN GENERATED (scripts/find_golden_configs.py) ---"
_END = "# --- END GENERATED ---"


def _shape_table() -> list[tuple[str, int, int, int]]:
    """``(name, M, N, K)`` for every golden shape — all plain 2-D ``(M,K)@(K,N)``."""
    shapes: list[tuple[str, int, int, int]] = []
    # Standard squares.
    for s in (512, 1024, 2048, 4096):
        shapes.append((f"square.{s}", s, s, s))
    # Qwen3-Embedding-0.6B linears at seq 32/128/512 (M = seq; batch=1 squeezes away).
    H, INTER, Q, KV = QWEN3_06B_HIDDEN, QWEN3_06B_INTER, QWEN3_06B_Q_DIM, QWEN3_06B_KV_DIM
    qwen_projs = [
        ("q_proj", Q, H),
        ("kv_proj", KV, H),
        ("o_proj", H, Q),
        ("gate_up_proj", INTER, H),
        ("down_proj", H, INTER),
    ]
    for seq in (32, 128, 512):
        for proj, N, K in qwen_projs:
            shapes.append((f"qwen3_06b.{proj}.s{seq}", seq, N, K))
    return shapes


def _pin_true_fp32() -> None:
    """Pin the cuBLAS reference to true IEEE fp32 (SGEMM), not the TF32 tensor-core
    path. deplodock emits CUDA-core FMA accumulate, so without this the ratio would
    compare against a ~5-10x faster (and numerically different) TF32 GEMM."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"  # newer torch
    except AttributeError:
        pass


def _clear_mma_pins() -> None:
    """fp32 matmul must stay on the thread tier — drop any inherited MMA / warp pins
    (the matmul-MMA tests set ``DEPLODOCK_MMA=1`` etc.)."""
    for var in ("DEPLODOCK_MMA", "DEPLODOCK_WM", "DEPLODOCK_WN", "DEPLODOCK_ATOM_KIND"):
        os.environ.pop(var, None)


def _extract_knobs(assembled) -> dict:
    """Pull the matmul kernel's knobs off the assembled ``Graph[CudaOp]``. Knobs are
    merged forward onto the final CudaOp, so this is the full picked config. A split-K
    winner has a 2nd (combine) CudaOp — pick the one carrying ``BM`` (the main matmul);
    fall back to the richest knob dict."""
    from deplodock.compiler.ir.cuda.ir import CudaOp

    cuda_knobs = [dict(n.op.knobs) for n in assembled.nodes.values() if isinstance(n.op, CudaOp) and n.op.knobs]
    if not cuda_knobs:
        return {}
    for k in cuda_knobs:
        if "BM" in k:
            return k
    return max(cuda_knobs, key=len)


def _tune_and_bench(name, M, N, K, *, db_dir, patience, warmup, iters):
    """Run the full per-shape pipeline; return a ``MatmulGoldenConfig`` or ``None``."""
    from deplodock.commands.run import bench_full_model_real
    from deplodock.commands.trace import trace_inline_code
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.context import Context
    from deplodock.compiler.pipeline.search import SearchDB
    from deplodock.compiler.pipeline.search.two_level import run_two_level_tune
    from deplodock.compiler.target import live_compute_capability

    # 1. Trace via the shared snippet so the tuned graph == what a config reproduces.
    info = trace_inline_code(matmul_snippet(M, N, K))
    graph, module, args, kwargs = info["graph"], info["module"], info["args"], info["kwargs"]

    # 2. Tune at -O1 (ranking). The opt level folds into the perf-cache key, so it must
    #    be set BEFORE Context.probe(). Isolated DB per shape — never the shared cache.
    os.environ[config.NVCC_FLAGS] = "-Xcicc -O1"
    db_path = db_dir / f"iso_{name}.db"
    db = SearchDB(path=db_path)
    ctx = Context.probe()
    tune_backend = CudaBackend(bench_compile_timeout_s=2.0, bench_run_timeout_s=2.0, bench_wall_timeout_s=6.0)
    result = run_two_level_tune(graph, ctx=ctx, db=db, backend=tune_backend, patience=patience)
    if result.assembled is None:
        logger.warning("  %s: no kernel tuned — skipping", name)
        return None

    knobs = _extract_knobs(result.assembled)

    # 3. Re-bench the winner at -O3 (deployable) vs cuBLAS. With TF32 pinned off,
    #    results["Eager PyTorch"] IS the cuBLAS SGEMM latency.
    os.environ[config.NVCC_FLAGS] = ""
    bench_backend = CudaBackend(tune_db=str(db_path), bench_compile_timeout_s=60.0, bench_run_timeout_s=60.0)
    results, _ = bench_full_model_real(
        module, args, kwargs, result.assembled, bench_backend, warmup=warmup, iters=iters, bench_backends="eager,deplodock"
    )
    deplodock_us = results["Deplodock"]
    cublas_us = results["Eager PyTorch"]

    cap = live_compute_capability()
    cfg = MatmulGoldenConfig(
        name=name,
        M=M,
        N=N,
        K=K,
        gpu_name=_gpu_name(),
        compute_cap=cap,
        knobs=knobs,
        deplodock_us=round(deplodock_us, 1),
        cublas_us=round(cublas_us, 1),
    )
    flag = "GOLDEN" if cfg.golden else "      "
    logger.info(
        "  %s [%s] deplodock=%.1fus cublas=%.1fus ratio=%.2f  %s",
        name,
        flag,
        deplodock_us,
        cublas_us,
        cfg.ratio,
        knobs,
    )
    return cfg


_GPU_NAME_CACHE: list[str] = []


def _gpu_name() -> str:
    if not _GPU_NAME_CACHE:
        import cupy as cp

        _GPU_NAME_CACHE.append(cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    return _GPU_NAME_CACHE[0]


def _render(configs: list[MatmulGoldenConfig]) -> str:
    """Render the ``GOLDEN_CONFIGS = [...]`` assignment for the data module."""
    lines = ["GOLDEN_CONFIGS: list[GoldenConfig] = ["]
    for c in configs:
        dtype = "" if c.dtype == "fp32" else f'\n        dtype="{c.dtype}",'
        lines.append(
            "    MatmulGoldenConfig(\n"
            f'        name="{c.name}",\n'
            f"        M={c.M}, N={c.N}, K={c.K},{dtype}\n"
            f'        gpu_name="{c.gpu_name}",\n'
            f"        compute_cap={c.compute_cap!r},\n"
            f"        knobs={c.knobs!r},\n"
            f"        deplodock_us={c.deplodock_us!r}, cublas_us={c.cublas_us!r},\n"
            "    ),"
        )
    lines.append("]")
    return "\n".join(lines)


def _splice(rendered: str) -> None:
    """Rewrite the data module's generated region in place, then ruff-format it
    (the rendered knob dicts are one-liners that exceed the 140-char limit)."""
    import subprocess

    text = _DATA_MODULE.read_text()
    before, _, rest = text.partition(_BEGIN)
    _, _, after = rest.partition(_END)
    new = f"{before}{_BEGIN}\n{rendered}\n{_END}{after}"
    _DATA_MODULE.write_text(new)
    logger.info("Wrote %d configs to %s", rendered.count("MatmulGoldenConfig("), _DATA_MODULE)

    ruff = Path(sys.executable).parent / "ruff"
    ruff_cmd = str(ruff) if ruff.exists() else "ruff"
    try:
        subprocess.run([ruff_cmd, "format", str(_DATA_MODULE)], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:  # noqa: BLE001 — formatting is best-effort
        logger.warning("ruff format skipped (%s); run `make format` before committing", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover golden matmul configs by autotuning.")
    parser.add_argument("--shapes", default=None, help="Comma-separated name filters (substring match), e.g. 'square,qwen3_06b.q_proj'")
    parser.add_argument("--patience", type=int, default=50, help="Inner-search patience per kernel (default: 50)")
    parser.add_argument("--warmup", type=int, default=10, help="Bench warmup iterations (default: 10)")
    parser.add_argument("--iters", type=int, default=100, help="Bench measurement iterations (default: 100)")
    parser.add_argument("--db-dir", default=None, help="Directory for isolated per-shape tuning DBs (default: a temp dir)")
    parser.add_argument("--dry-run", action="store_true", help="Print the generated configs to stdout; do not rewrite the data module")
    args = parser.parse_args()

    _pin_true_fp32()
    _clear_mma_pins()

    shapes = _shape_table()
    if args.shapes:
        filters = [f.strip() for f in args.shapes.split(",") if f.strip()]
        shapes = [s for s in shapes if any(f in s[0] for f in filters)]
    if not shapes:
        logger.error("no shapes matched --shapes %r", args.shapes)
        sys.exit(2)

    db_dir = Path(args.db_dir) if args.db_dir else Path(tempfile.mkdtemp(prefix="deplodock-golden-"))
    db_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Tuning %d shape(s); isolated DBs under %s", len(shapes), db_dir)

    configs: list[MatmulGoldenConfig] = []
    for name, M, N, K in shapes:
        logger.info("[%d/%d] %s  (M=%d N=%d K=%d)", len(configs) + 1, len(shapes), name, M, N, K)
        try:
            cfg = _tune_and_bench(name, M, N, K, db_dir=db_dir, patience=args.patience, warmup=args.warmup, iters=args.iters)
        except RuntimeError as exc:
            # A bench watchdog abort dirties the CUDA context unrecoverably — stop, but
            # keep whatever landed so far so the run isn't a total loss.
            logger.error("  %s: aborted (%s); stopping early — re-run --shapes to fill gaps", name, exc)
            break
        except Exception as exc:  # noqa: BLE001 — a single bad shape shouldn't kill the sweep
            logger.error("  %s: failed (%s); skipping", name, exc)
            continue
        if cfg is not None:
            configs.append(cfg)

    if not configs:
        logger.error("no configs produced")
        sys.exit(1)

    golden_n = sum(1 for c in configs if c.golden)
    logger.info("Done: %d config(s), %d golden (>=95%% cuBLAS)", len(configs), golden_n)

    rendered = _render(configs)
    if args.dry_run:
        print()
        print(rendered)
    else:
        _splice(rendered)


if __name__ == "__main__":
    main()
