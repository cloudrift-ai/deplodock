#!/usr/bin/env python3
"""Bench a running generative (chat) server: single-stream decode tok/s (TTFT + steady-state)
and concurrent system throughput. Talks to any OpenAI-compatible `/v1/chat/completions`
endpoint — e.g. a server started by `emmy serve <model> --generate` (the emmy plugin)
or `... --generate --stock` (native vLLM baseline).

Usage (start a server in one shell, bench in another):
    emmy serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --generate --max-model-len 2048 \
        --gpu-memory-utilization=0.80 --max-num-batched-tokens 1024 --port 8001
    python scripts/bench_gen_serve.py --port 8001 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

try:
    import requests
except ImportError:
    sys.exit("requests required: pip install requests")

PROMPT = "Write a detailed explanation of how transformers work in machine learning."


def stream_one(base, model, max_tokens, timeout):
    """One streaming request. Returns (ttft_s, decode_tps, completion_tokens)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    t_first = t_last = None
    n_chunks = 0
    completion_tokens = None
    with requests.post(f"{base}/v1/chat/completions", json=body, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                completion_tokens = obj["usage"].get("completion_tokens")
            choices = obj.get("choices") or []
            if choices and choices[0].get("delta", {}).get("content"):
                now = time.perf_counter()
                t_first = t_first or now
                t_last = now
                n_chunks += 1
    n = completion_tokens or n_chunks
    ttft = (t_first - t0) if t_first else float("nan")
    decode_tps = (n - 1) / (t_last - t_first) if (t_first and t_last and t_last > t_first) else float("nan")
    return ttft, decode_tps, n


def blocking_one(base, model, max_tokens, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    r = requests.post(f"{base}/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["usage"]["completion_tokens"]


def main():
    ap = argparse.ArgumentParser(description="Generative (chat) served-throughput benchmark")
    ap.add_argument("--port", default="8001", help="Server port (default: 8001)")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Served model id")
    ap.add_argument("--max-tokens", type=int, default=128, help="Output tokens per request (default: 128)")
    ap.add_argument("--concurrency", type=int, default=16, help="Throughput-pass concurrency (default: 16)")
    ap.add_argument("--requests", type=int, default=64, help="Throughput-pass request count (default: 64)")
    ap.add_argument("--timeout", type=float, default=300, help="Per-request timeout seconds (default: 300)")
    args = ap.parse_args()
    base = f"http://localhost:{args.port}"

    print(f"[single-stream] model={args.model} max_tokens={args.max_tokens}")
    rows = [stream_one(base, args.model, args.max_tokens, args.timeout) for _ in range(3)]  # warmup + median of 3
    ttft = sorted(r[0] for r in rows)[1]
    dtps = sorted(r[1] for r in rows)[1]
    print(f"  TTFT={ttft * 1000:.0f} ms   decode={dtps:.1f} tok/s   (gen {rows[-1][2]} tokens, median of 3)")

    print(f"[throughput] concurrency={args.concurrency} requests={args.requests}")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        toks = list(ex.map(lambda _: blocking_one(base, args.model, args.max_tokens, args.timeout), range(args.requests)))
    wall = time.perf_counter() - t0
    total = sum(toks)
    print(f"  {total} output tokens in {wall:.1f} s   =>  {total / wall:.1f} tok/s system   ({args.requests / wall:.1f} req/s)")


if __name__ == "__main__":
    main()
