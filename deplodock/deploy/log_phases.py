"""Best-effort extraction of engine load/warmup sub-phases from container logs.

These split ``weights_load`` and ``cuda_graph_capture`` out of the single
``docker compose up --wait`` window. The function is pure, never raises, and
degrades silently to ``{}`` when patterns don't match (different engine, log-format
drift, eager mode with no graph capture). The orchestrator's wall-clock
``model_load_and_warmup`` stays the authoritative number — these are additive detail.
"""

import re

# vLLM V1 log lines, e.g.:
#   "Loading model weights took 14.2978 GB and 11.40 seconds"
#   "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|...| 67/67 [00:18<00:00, ...]"
_VLLM_WEIGHTS = re.compile(r"Loading model weights took \d+(?:\.\d+)? ?Gi?B and (\d+(?:\.\d+)?) seconds")
# CUDA-graph capture has no uniform "took N seconds" line across versions; read the
# tqdm progress bar's elapsed [MM:SS<...] (the bar reaches 100% on completion).
_VLLM_GRAPH_BAR = re.compile(r"Capturing CUDA graphs.*?\[(\d+):(\d+)<")

# SGLang log lines vary across versions; keep patterns lenient. Weights timing is
# unreliable, so we only attempt the graph-capture duration. Require an actual digit
# (not a lone ".") so prose like "Capture cuda graph begin." doesn't false-match.
_SGLANG_CAPTURE = re.compile(r"[Cc]apture(?: cuda)? graph.*?(\d+(?:\.\d+)?)\s*s")


def parse_engine_load_phases(logs: str, engine: str) -> dict:
    """Parse weights-load / cuda-graph-capture durations (seconds) from container logs.

    Returns a dict with any of ``weights_load`` / ``cuda_graph_capture`` that matched;
    missing keys are simply absent. Unknown engine or no match -> ``{}``. Never raises.
    """
    if not logs:
        return {}
    out: dict = {}
    if engine == "vllm":
        m = _VLLM_WEIGHTS.search(logs)
        if m:
            out["weights_load"] = float(m.group(1))
        mg = _VLLM_GRAPH_BAR.search(logs)
        if mg:
            out["cuda_graph_capture"] = float(int(mg.group(1)) * 60 + int(mg.group(2)))
    elif engine == "sglang":
        mc = _SGLANG_CAPTURE.search(logs)
        if mc:
            out["cuda_graph_capture"] = float(mc.group(1))
    return out
