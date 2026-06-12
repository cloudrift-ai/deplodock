#!/usr/bin/env python3
"""Accuracy gate for the deplodock vLLM embedding plugin.

Embeds a fixed text set through two OpenAI-compatible ``/v1/embeddings``
endpoints (e.g. the deplodock-backed server and a stock vLLM server) and
asserts per-text cosine similarity, plus agreement of the pairwise-similarity
matrices (the quantity retrieval quality actually depends on).

    python scripts/compare_embeddings.py --a http://localhost:8311 --b http://localhost:8312 \
        --model Qwen/Qwen3-Embedding-0.6B [--min-cosine 0.99]
"""

from __future__ import annotations

import argparse
import sys

import httpx
import numpy as np

TEXTS = [
    "What is the capital of France?",
    "Paris is the capital and largest city of France.",
    "The mitochondria is the powerhouse of the cell.",
    "How do I reset my password?",
    "To reset your password, click 'Forgot password' on the login page.",
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: best gpu for llm inference",
    "The RTX 5090 has 32 GB of GDDR7 memory.",
    "El gato se sentó en la alfombra.",
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)",
    "A",
    "short",
    "This is a deliberately much longer paragraph intended to exercise a bigger sequence length. "
    "It rambles about embedding models, retrieval-augmented generation, vector databases, cosine "
    "similarity, and the importance of benchmarking serving stacks under realistic workloads.",
]


def embed(base_url: str, model: str, texts: list[str]) -> np.ndarray:
    resp = httpx.post(f"{base_url}/v1/embeddings", json={"model": model, "input": texts}, timeout=120)
    resp.raise_for_status()
    data = resp.json()["data"]
    return np.array([row["embedding"] for row in sorted(data, key=lambda r: r["index"])], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="first server base URL (e.g. the deplodock-backed one)")
    ap.add_argument("--b", required=True, help="second server base URL (e.g. stock vLLM)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--min-cosine", type=float, default=0.99)
    args = ap.parse_args()

    ea, eb = embed(args.a, args.model, TEXTS), embed(args.b, args.model, TEXTS)
    if ea.shape != eb.shape:
        print(f"FAIL: shape mismatch {ea.shape} vs {eb.shape}")
        return 1
    ea /= np.linalg.norm(ea, axis=1, keepdims=True)
    eb /= np.linalg.norm(eb, axis=1, keepdims=True)

    cos = (ea * eb).sum(axis=1)
    for t, c in zip(TEXTS, cos, strict=True):
        marker = "" if c >= args.min_cosine else "   <-- BELOW THRESHOLD"
        print(f"cosine={c:.6f}  {t[:60]!r}{marker}")
    sim_diff = np.abs(ea @ ea.T - eb @ eb.T).max()
    print(f"\nmin cosine: {cos.min():.6f}   pairwise-similarity matrix max diff: {sim_diff:.6f}")
    if cos.min() < args.min_cosine:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
