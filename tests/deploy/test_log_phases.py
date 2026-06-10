"""Tests for best-effort engine-load-phase parsing from container logs."""

from deplodock.deploy.log_phases import decompose_model_load, parse_engine_load_phases

# Real vLLM 0.20.1 startup log lines (captured from a live CloudRift pro6000 deploy of
# Qwen3.6-35B-A3B-FP8). The CUDA-graph tqdm bars are concatenated onto one line, as
# docker captures the \r-overwrites — the clean "Graph capturing finished in N secs"
# summary line is what we parse.
VLLM_LOGS = """\
vllm_0  | (EngineCore pid=122) INFO 06-04 00:35:18 [gpu_model_runner.py:4879] Model loading took 33.38 GiB memory and 22.120985 seconds
vllm_0  | (EngineCore pid=122) INFO 06-04 00:35:29 [backends.py:1128] Dynamo bytecode transform time: 10.31 s
vllm_0  | (EngineCore pid=122) INFO 06-04 00:36:07 [backends.py:391] Compiling a graph for compile range (1, 8192) takes 36.51 s
vllm_0  | (EngineCore pid=122) Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):   0%|          | 0/7 [00:00<?, ?it/s]Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|##########| 7/7 [00:02<00:00,  3.32it/s]
vllm_0  | (EngineCore pid=122) INFO 06-04 00:38:39 [gpu_model_runner.py:6133] Graph capturing finished in 4 secs, took 0.12 GiB
vllm_0  | (EngineCore pid=122) INFO 06-04 00:38:39 [core.py:299] init engine (profile, create kv cache, warmup model) took 200.37 s (compilation: 54.60 s)
"""  # noqa: E501

# Older vLLM phrasing for weight load.
VLLM_OLD_WEIGHTS = "INFO Loading model weights took 14.2978 GB and 11.40 seconds\n"

SGLANG_LOGS = """\
sglang_0 | [INFO] Capture cuda graph begin.
sglang_0 | [INFO] Capture cuda graph end. Time elapsed: 12.30 s
"""


def test_parse_vllm():
    out = parse_engine_load_phases(VLLM_LOGS, "vllm")
    assert out["weights_load"] == 22.120985
    assert out["torch_compile"] == 54.60
    assert out["init_engine"] == 200.37
    # cuda_graph_capture comes from the clean "finished in 4 secs" summary, NOT the
    # tqdm bar's first [00:00<...] update (which would be 0).
    assert out["cuda_graph_capture"] == 4.0


def test_decompose_full_breakdown_sums_to_total():
    # Raw values from the real deploy run; model_load_and_warmup wall-clock = 288.4s.
    raw = {"weights_load": 22.12, "torch_compile": 54.60, "cuda_graph_capture": 4.0, "init_engine": 200.37}
    leaves = decompose_model_load(raw, 288.4)
    assert leaves["weights_load"] == 22.12
    assert leaves["torch_compile"] == 54.60
    assert leaves["cuda_graph_capture"] == 4.0
    # engine_warmup = init_engine - compile - graph = 200.37 - 54.60 - 4.0
    assert leaves["engine_warmup"] == 141.77
    # startup = total - (weights + init_engine)
    assert leaves["startup"] == 65.91
    assert abs(sum(leaves.values()) - 288.4) < 0.05
    assert "other" not in leaves


def test_decompose_fallback_other_when_no_init_engine():
    # Older vLLM / SGLang: no init-engine line → unattributed time collapses to `other`.
    raw = {"weights_load": 10.0, "cuda_graph_capture": 3.0}
    leaves = decompose_model_load(raw, 100.0)
    assert leaves["weights_load"] == 10.0
    assert leaves["cuda_graph_capture"] == 3.0
    assert leaves["other"] == 87.0
    assert "engine_warmup" not in leaves
    assert "startup" not in leaves


def test_decompose_no_remainder_when_subsecond_skew():
    raw = {"weights_load": 22.12, "torch_compile": 54.60, "cuda_graph_capture": 4.0, "init_engine": 200.37}
    # total essentially equals weights + init_engine → startup remainder is ~0, omitted.
    leaves = decompose_model_load(raw, 222.49)
    assert "startup" not in leaves
    assert "engine_warmup" in leaves


def test_parse_vllm_old_weights_phrasing():
    out = parse_engine_load_phases(VLLM_OLD_WEIGHTS, "vllm")
    assert out == {"weights_load": 11.40}


def test_parse_vllm_graph_bar_fallback_uses_last_match():
    # No clean summary line → fall back to the tqdm bar, taking the LAST [MM:SS<].
    logs = "Capturing CUDA graphs:   0%| | 0/7 [00:00<?, ?it/s]Capturing CUDA graphs: 100%|#| 7/7 [00:18<00:00, 3.5it/s]\n"
    out = parse_engine_load_phases(logs, "vllm")
    assert out["cuda_graph_capture"] == 18.0


def test_parse_vllm_weights_only():
    logs = "Model loading took 7.5 GiB memory and 5.20 seconds\n"
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
