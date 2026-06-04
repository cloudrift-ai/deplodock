"""Tests for best-effort engine-load-phase parsing from container logs."""

from deplodock.deploy.log_phases import parse_engine_load_phases

# Representative vLLM V1 startup log lines.
VLLM_LOGS = """\
vllm_0  | INFO 06-03 12:00:01 Loading model weights took 14.2978 GB and 11.40 seconds
vllm_0  | INFO 06-03 12:00:30 Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): \
100%|##########| 67/67 [00:18<00:00, 3.50it/s]
vllm_0  | INFO 06-03 12:00:31 init engine (profile, create kv cache, warmup model) took 18.95 seconds
"""

SGLANG_LOGS = """\
sglang_0 | [INFO] Capture cuda graph begin.
sglang_0 | [INFO] Capture cuda graph end. Time elapsed: 12.30 s
"""


def test_parse_vllm():
    out = parse_engine_load_phases(VLLM_LOGS, "vllm")
    assert out["weights_load"] == 11.40
    # CUDA-graph elapsed is read from the tqdm bar [00:18<...] → 18 seconds.
    assert out["cuda_graph_capture"] == 18.0


def test_parse_vllm_weights_only():
    logs = "Loading model weights took 7.5 GiB and 5.20 seconds\n"
    out = parse_engine_load_phases(logs, "vllm")
    assert out == {"weights_load": 5.20}


def test_parse_sglang_graph():
    out = parse_engine_load_phases(SGLANG_LOGS, "sglang")
    assert out["cuda_graph_capture"] == 12.30


def test_unmatched_returns_empty():
    assert parse_engine_load_phases("no relevant lines here", "vllm") == {}


def test_unknown_engine_returns_empty():
    assert parse_engine_load_phases(VLLM_LOGS, "tgi") == {}


def test_empty_logs_returns_empty():
    assert parse_engine_load_phases("", "vllm") == {}
