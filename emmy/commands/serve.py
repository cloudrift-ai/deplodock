"""``deplodock serve`` — vLLM embedding serving with deplodock kernels, one command.

Wraps ``vllm serve`` with the ``deplodock/serving`` plugin boilerplate
(``--runner pooling --enforce-eager --hf-overrides …DeplodockEmbedModel…``,
plus ``--max-model-len 4096`` — the dynamic-dim cap — and
``--gpu-memory-utilization=0.9`` unless overridden), and passes any
unrecognized flags through to vLLM verbatim. ``--stock`` drops the
plugin flags for the raw-vLLM baseline, so an A/B is two invocations of the
same command. ``--bench`` turns it into a one-shot benchmark: start the
server, wait for ``/health``, run ``vllm bench serve --backend
openai-embeddings`` against it, print the result table, and shut the server
down.

    deplodock serve Qwen/Qwen3-Embedding-0.6B --gpu-memory-utilization 0.8
    deplodock serve Qwen/Qwen3-Embedding-0.6B --bench --random-input-len 32
    deplodock serve Qwen/Qwen3-Embedding-0.6B --bench --stock     # baseline

Flag routing: deplodock's own flags (``--bench``, ``--stock``, the bench
params, ``--dry-run``) are extracted wherever they appear; everything else is
forwarded to ``vllm serve``. Tokens after a literal ``--`` are forwarded
verbatim without extraction (the escape hatch for a hypothetical name clash).
"""

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# The dynamic-dim cap (compiler/trace/dynamic.py DYNAMIC_DIM_MAX) — the plugin
# runner rejects anything larger at startup. Applied to --stock too, so the
# A/B compares both engines at the same max-model-len.
_DEFAULT_MAX_MODEL_LEN = "4096"
_DEFAULT_GPU_MEMORY_UTILIZATION = "0.9"

_PLUGIN_ARGS = ["--enforce-eager", "--hf-overrides", '{"architectures": ["DeplodockEmbedModel"]}']

_HEALTH_TIMEOUT_S = 1800  # first boot compiles the whole model; warm cubin cache is much faster


def _add_own_flags(parser, *, suppress_defaults: bool) -> None:
    """The deplodock-side flags, declared twice: on the subparser (for --help
    and flags placed before MODEL) and on the post-MODEL re-parse (with
    SUPPRESS defaults, so only flags actually present override)."""

    def d(value):
        return argparse.SUPPRESS if suppress_defaults else value

    parser.add_argument(
        "--stock", action="store_true", default=d(False), help="Serve stock vLLM kernels instead of the deplodock plugin (A/B baseline)."
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        default=d(False),
        help="Serve a generative (chat) model via DeplodockGenModel (--runner generate, fp16) instead of embeddings.",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        default=d(False),
        help="Start the server, run `vllm bench serve` against it, print results, shut down.",
    )
    parser.add_argument("--max-concurrency", type=int, default=d(32), help="Bench client concurrency (with --bench).")
    parser.add_argument("--num-prompts", type=int, default=d(256), help="Bench request count (with --bench).")
    parser.add_argument("--random-input-len", type=int, default=d(512), help="Bench tokens per request (with --bench).")
    parser.add_argument(
        "--bench-seed", type=int, default=d(0), help="Bench prompt-sampling seed (with --bench; `--seed` itself forwards to vllm serve)."
    )
    parser.add_argument("--dry-run", action="store_true", default=d(False), help="Print the vllm command(s) without running.")


def register_serve_command(subparsers):
    parser = subparsers.add_parser(
        "serve",
        help="Serve an embedding model via vLLM with deplodock-compiled kernels (optionally benchmark it)",
        description="Wraps `vllm serve` with the deplodock plugin flags; unrecognized flags pass through to vLLM "
        "(tokens after a literal `--` pass through verbatim).",
        allow_abbrev=False,
    )
    parser.add_argument("model", help="HuggingFace model ID (e.g., Qwen/Qwen3-Embedding-0.6B)")
    _add_own_flags(parser, suppress_defaults=False)
    parser.add_argument(
        "vllm_args",
        nargs=argparse.REMAINDER,
        metavar="VLLM_ARGS",
        help="Extra args forwarded to `vllm serve` (e.g. --gpu-memory-utilization 0.8 --port 8123).",
    )
    parser.set_defaults(func=handle_serve)


def _split_own_flags(args) -> list[str]:
    """argparse REMAINDER swallows *everything* after MODEL — including our own
    flags. Re-parse the remainder: extract deplodock flags into ``args``
    (SUPPRESS defaults → only flags actually present override), forward the
    rest; anything after a literal ``--`` forwards without extraction."""
    rem = list(args.vllm_args)
    verbatim: list[str] = []
    if "--" in rem:
        i = rem.index("--")
        rem, verbatim = rem[:i], rem[i + 1 :]
    own = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    _add_own_flags(own, suppress_defaults=True)
    _, passthrough = own.parse_known_args(rem, namespace=args)
    return passthrough + verbatim


def _has_flag(vllm_args: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(f"{flag}=") for a in vllm_args)


def _flag_value(vllm_args: list[str], flag: str, default: str) -> str:
    for i, a in enumerate(vllm_args):
        if a == flag and i + 1 < len(vllm_args):
            return vllm_args[i + 1]
        if a.startswith(f"{flag}="):
            return a.split("=", 1)[1]
    return default


def build_serve_cmd(model: str, *, stock: bool, vllm_args: list[str], generate: bool = False) -> list[str]:
    cmd = ["vllm", "serve", model, "--runner", "generate" if generate else "pooling"]
    if not stock and generate:
        cmd += ["--enforce-eager", "--hf-overrides", '{"architectures": ["DeplodockGenModel"]}']
        # Force fp16 across the deplodock↔vLLM seam: vLLM defaults --dtype auto → bf16 for a
        # bf16 checkpoint, but the deplodock trunk emits fp16. Reject an incompatible override.
        if _has_flag(vllm_args, "--dtype"):
            dt = _flag_value(vllm_args, "--dtype", "")
            if dt not in ("float16", "half", "fp16"):
                raise ValueError(f"generative serving requires fp16; --dtype {dt!r} is incompatible (use --dtype float16)")
        else:
            cmd += ["--dtype", "float16"]
        # The flattened width (sum of newly-scheduled tokens per step) must stay within
        # DYNAMIC_DIM_MAX (= _DEFAULT_MAX_MODEL_LEN). vLLM's default max_num_batched_tokens
        # (e.g. 8192 on a big GPU) would trip the model's startup bound, so cap it here.
        if _has_flag(vllm_args, "--max-num-batched-tokens"):
            mnbt = _flag_value(vllm_args, "--max-num-batched-tokens", "")
            if mnbt.isdigit() and int(mnbt) > int(_DEFAULT_MAX_MODEL_LEN):
                raise ValueError(f"--max-num-batched-tokens {mnbt} exceeds the dynamic-dim cap ({_DEFAULT_MAX_MODEL_LEN}); use it or lower")
        else:
            cmd += ["--max-num-batched-tokens", _DEFAULT_MAX_MODEL_LEN]
    elif not stock:
        cmd += _PLUGIN_ARGS
    if not _has_flag(vllm_args, "--max-model-len"):
        cmd += ["--max-model-len", _DEFAULT_MAX_MODEL_LEN]
    if not _has_flag(vllm_args, "--gpu-memory-utilization"):
        cmd += [f"--gpu-memory-utilization={_DEFAULT_GPU_MEMORY_UTILIZATION}"]
    return cmd + vllm_args


def build_bench_cmd(model: str, *, port: str, max_concurrency: int, num_prompts: int, random_input_len: int, seed: int) -> list[str]:
    return [
        "vllm",
        "bench",
        "serve",
        "--model",
        model,
        "--backend",
        "openai-embeddings",
        "--endpoint",
        "/v1/embeddings",
        "--base-url",
        f"http://localhost:{port}",
        "--max-concurrency",
        str(max_concurrency),
        "--num-prompts",
        str(num_prompts),
        "--random-input-len",
        str(random_input_len),
        "--seed",
        str(seed),
    ]


def _vllm_bin() -> str:
    """The vllm console script from this interpreter's environment (so the
    plugin entry point installed alongside deplodock is the one that loads),
    falling back to PATH."""
    candidate = Path(sys.executable).parent / "vllm"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("vllm")
    if found:
        return found
    logger.error("vllm not found — install the serving extra: pip install -e '.[compile,serving]'")
    sys.exit(1)


def handle_serve(args):
    vllm_args = _split_own_flags(args)  # re-parses own flags placed after MODEL into args
    if args.generate and args.bench:
        logger.error("--bench is not supported with --generate yet (the bench client targets /v1/embeddings).")
        sys.exit(1)
    serve_cmd = build_serve_cmd(args.model, stock=args.stock, vllm_args=vllm_args, generate=args.generate)
    port = _flag_value(vllm_args, "--port", "8000")
    bench_cmd = build_bench_cmd(
        args.model,
        port=port,
        max_concurrency=args.max_concurrency,
        num_prompts=args.num_prompts,
        random_input_len=args.random_input_len,
        seed=args.bench_seed,
    )

    if args.dry_run:
        # shlex.join so the printed line is shell-pastable (the JSON in
        # --hf-overrides needs quoting; the real execution uses argv lists).
        print(shlex.join(serve_cmd))
        if args.bench:
            print(shlex.join(bench_cmd))
        return

    vllm = _vllm_bin()
    serve_cmd[0] = vllm
    if not args.bench:
        # Plain serving: replace this process so signals/Ctrl-C reach vLLM directly.
        os.execv(vllm, serve_cmd)

    bench_cmd[0] = vllm
    _serve_and_bench(serve_cmd, bench_cmd, port)


def _serve_and_bench(serve_cmd: list[str], bench_cmd: list[str], port: str) -> None:
    log_file = tempfile.NamedTemporaryFile(mode="wb", prefix="deplodock-serve-", suffix=".log", delete=False)
    logger.info("Starting server (logs: %s)...", log_file.name)
    logger.info("  %s", shlex.join(serve_cmd))
    server = subprocess.Popen(serve_cmd, stdout=log_file, stderr=subprocess.STDOUT)
    try:
        _wait_for_health(server, port, log_file.name)
        logger.info("Server healthy — running benchmark...")
        logger.info("  %s", shlex.join(bench_cmd))
        rc = subprocess.run(bench_cmd).returncode
    finally:
        logger.info("Shutting down server...")
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
        log_file.close()
    if rc != 0:
        logger.error("Benchmark failed (rc=%d); server logs: %s", rc, log_file.name)
        sys.exit(rc)


def _wait_for_health(server: subprocess.Popen, port: str, log_path: str) -> None:
    """Poll /health until the server answers; fail fast if it exits first
    (first boot may compile the whole model — be patient)."""
    import httpx

    deadline = time.monotonic() + _HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if server.poll() is not None:
            tail = "".join(open(log_path, errors="replace").readlines()[-15:])
            logger.error("Server exited during startup (rc=%d). Log tail (%s):\n%s", server.returncode, log_path, tail)
            sys.exit(1)
        try:
            if httpx.get(f"http://localhost:{port}/health", timeout=3).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    logger.error("Server did not become healthy within %ds; logs: %s", _HEALTH_TIMEOUT_S, log_path)
    server.terminate()
    sys.exit(1)
