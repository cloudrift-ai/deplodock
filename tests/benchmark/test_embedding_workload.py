"""Embedding-recipe bench command + metrics parsing + smoke-response checks."""

from emmy.benchmark.results import parse_benchmark_metrics
from emmy.benchmark.workload import build_bench_command
from emmy.deploy.orchestrate import _check_chat_response, _check_embedding_response
from emmy.recipe.types import Recipe


def _recipe(task: str) -> Recipe:
    return Recipe.from_dict(
        {
            "model": {"huggingface": "Qwen/Qwen3-Embedding-0.6B", "task": task},
            "engine": {"llm": {"context_length": 4096, "vllm": {}}},
            "benchmark": {"max_concurrency": 8, "num_prompts": 32, "random_input_len": 128, "random_output_len": 1},
        }
    )


def test_embed_bench_command_targets_embeddings_endpoint():
    cmd = build_bench_command(_recipe("embed"))
    assert "--backend openai-embeddings" in cmd
    assert "--endpoint /v1/embeddings" in cmd
    assert "--random-output-len" not in cmd
    assert "--random-input-len 128" in cmd


def test_generate_bench_command_unchanged():
    cmd = build_bench_command(_recipe("generate"))
    assert "--backend" not in cmd
    assert "--random-output-len 1" in cmd


# Captured from a real `vllm bench serve --backend openai-embeddings` run (v0.22.1).
_EMBED_BENCH_OUTPUT = """\
============ Serving Benchmark Result ============
Successful requests:                     32
Failed requests:                         0
Maximum request concurrency:             8
Benchmark duration (s):                  0.12
Total input tokens:                      4096
Request throughput (req/s):              277.99
Total token throughput (tok/s):          35582.34
----------------End-to-end Latency----------------
Mean E2EL (ms):                          27.28
Median E2EL (ms):                        15.49
P99 E2EL (ms):                           70.07
==================================================
"""


def test_parse_embeddings_bench_output():
    m = parse_benchmark_metrics(_EMBED_BENCH_OUTPUT)
    assert m.successful_requests == 32
    assert m.failed_requests == 0
    assert m.request_throughput == 277.99
    assert m.total_token_throughput == 35582.34
    assert m.mean_e2el_ms == 27.28
    assert m.p99_e2el_ms == 70.07
    # Generation-only metrics stay unset rather than mis-parsing.
    assert m.mean_ttft_ms is None
    assert m.total_generated_tokens is None


def test_check_embedding_response():
    good = '{"data": [{"embedding": [0.6, 0.8], "index": 0}]}'
    assert _check_embedding_response(good)[0] == "pass"
    nan = '{"data": [{"embedding": [1.0, null], "index": 0}]}'
    assert _check_embedding_response(nan)[0] in ("fail", "retry")
    unnormalized = '{"data": [{"embedding": [3.0, 4.0], "index": 0}]}'
    verdict, detail = _check_embedding_response(unnormalized)
    assert verdict == "fail" and "norm" in detail
    not_ready = '{"error": "loading"}'
    assert _check_embedding_response(not_ready)[0] == "retry"
    assert _check_embedding_response("not json")[0] == "retry"


def test_check_chat_response():
    assert _check_chat_response('{"choices": [{"message": {"content": "The answer is 4."}}]}')[0] == "pass"
    assert _check_chat_response('{"choices": [{"message": {"content": "five"}}]}')[0] == "fail"
    assert _check_chat_response("oops")[0] == "retry"
    assert _check_chat_response('{"choices": [{"message": {}}]}')[0] == "retry"
