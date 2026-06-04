"""Best-effort extraction of engine load/warmup sub-phases from container logs.

These split ``weights_load`` / ``torch_compile`` / ``cuda_graph_capture`` out of the
single ``docker compose up --wait`` window. The function is pure, never raises, and
degrades silently to ``{}`` when patterns don't match (different engine, log-format
drift, eager mode with no graph capture). The orchestrator's wall-clock
``model_load_and_warmup`` stays the authoritative number — these are additive detail.

Patterns are matched against real vLLM 0.20.1 output; older phrasings are kept too.
"""

import re

# Weight load. Two vLLM phrasings across versions:
#   v0.20+:  "Model loading took 33.38 GiB memory and 22.120985 seconds"
#   older:   "Loading model weights took 14.2978 GB and 11.40 seconds"
_VLLM_WEIGHTS = re.compile(r"(?:Model loading|Loading model weights) took .*?(\d+(?:\.\d+)?) seconds")
# torch.compile / inductor kernel compilation, reported on the init-engine line:
#   "init engine (profile, create kv cache, warmup model) took 200.37 s (compilation: 54.60 s)"
_VLLM_COMPILE = re.compile(r"\(compilation:\s*(\d+(?:\.\d+)?)\s*s\)")
# CUDA graph capture — prefer the clean summary line (reports total secs):
#   "Graph capturing finished in 4 secs, took 0.12 GiB"
_VLLM_GRAPH_DONE = re.compile(r"[Gg]raph capturing finished in (\d+(?:\.\d+)?) secs")
# Fallback: the tqdm bar's elapsed [MM:SS<...]. Take the LAST match (the bar's final
# state), not the first (which is [00:00<...] at 0%).
_VLLM_GRAPH_BAR = re.compile(r"Capturing CUDA graphs.*?\[(\d+):(\d+)<")

# SGLang log lines vary across versions; keep patterns lenient. Weights timing is
# unreliable, so we only attempt the graph-capture duration. Require an actual digit
# (not a lone ".") so prose like "Capture cuda graph begin." doesn't false-match.
_SGLANG_CAPTURE = re.compile(r"[Cc]apture(?: cuda)? graph.*?(\d+(?:\.\d+)?)\s*s")


def parse_engine_load_phases(logs: str, engine: str) -> dict:
    """Parse weights-load / torch-compile / cuda-graph durations (seconds) from logs.

    Returns a dict with any of ``weights_load`` / ``torch_compile`` /
    ``cuda_graph_capture`` that matched; missing keys are simply absent. Unknown engine
    or no match -> ``{}``. Never raises.
    """
    if not logs:
        return {}
    out: dict = {}
    if engine == "vllm":
        m = _VLLM_WEIGHTS.search(logs)
        if m:
            out["weights_load"] = float(m.group(1))
        mc = _VLLM_COMPILE.search(logs)
        if mc:
            out["torch_compile"] = float(mc.group(1))
        md = _VLLM_GRAPH_DONE.search(logs)
        if md:
            out["cuda_graph_capture"] = float(md.group(1))
        else:
            bars = _VLLM_GRAPH_BAR.findall(logs)
            if bars:
                mm, ss = bars[-1]
                out["cuda_graph_capture"] = float(int(mm) * 60 + int(ss))
    elif engine == "sglang":
        mc = _SGLANG_CAPTURE.search(logs)
        if mc:
            out["cuda_graph_capture"] = float(mc.group(1))
    return out
