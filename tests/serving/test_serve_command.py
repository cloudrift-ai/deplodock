"""``emmy serve`` command construction + flag routing + dry-run (no vllm/GPU)."""

import argparse

import pytest

from emmy.commands.serve import build_bench_cmd, build_serve_cmd, handle_serve, register_serve_command

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _parse(argv):
    parser = argparse.ArgumentParser()
    register_serve_command(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def test_serve_cmd_plugin_defaults():
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=[])
    assert cmd[:3] == ["vllm", "serve", MODEL]
    assert "--runner" in cmd and "pooling" in cmd
    assert "--enforce-eager" in cmd
    assert '{"architectures": ["EmmyEmbedModel"]}' in cmd
    assert cmd[cmd.index("--max-model-len") + 1] == "4096"
    assert "--gpu-memory-utilization=0.9" in cmd


def test_serve_cmd_stock_drops_plugin_but_keeps_parity():
    cmd = build_serve_cmd(MODEL, stock=True, vllm_args=[])
    assert "--enforce-eager" not in cmd
    assert "--hf-overrides" not in cmd
    assert "pooling" in cmd
    assert cmd[cmd.index("--max-model-len") + 1] == "4096"  # same cap as the plugin → apples-to-apples
    assert "--gpu-memory-utilization=0.9" in cmd


def test_serve_cmd_passthrough_and_max_model_len_override():
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=["--max-model-len", "2048", "--gpu-memory-utilization", "0.8"])
    assert cmd.count("--max-model-len") == 1
    assert cmd[cmd.index("--max-model-len") + 1] == "2048"
    assert "--gpu-memory-utilization=0.9" not in cmd
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.8"


def test_serve_cmd_gpu_memory_utilization_equals_style_override():
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=["--gpu-memory-utilization=0.8"])
    assert "--gpu-memory-utilization=0.9" not in cmd
    assert "--gpu-memory-utilization=0.8" in cmd


def test_serve_cmd_max_model_len_equals_style():
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=["--max-model-len=2048"])
    assert "--max-model-len" not in cmd  # no separate default token added
    assert "--max-model-len=2048" in cmd
    assert "4096" not in cmd


def test_bench_cmd_targets_embeddings():
    cmd = build_bench_cmd(MODEL, port="8123", max_concurrency=8, num_prompts=32, random_input_len=64, seed=7)
    assert cmd[:3] == ["vllm", "bench", "serve"]
    assert cmd[cmd.index("--backend") + 1] == "openai-embeddings"
    assert cmd[cmd.index("--endpoint") + 1] == "/v1/embeddings"
    assert cmd[cmd.index("--base-url") + 1] == "http://localhost:8123"
    assert "--random-output-len" not in cmd


def test_own_flags_after_model_are_extracted(capsys):
    # The argparse-REMAINDER footgun: everything after MODEL lands in
    # vllm_args, INCLUDING emmy's own flags. They must still be honored
    # (this exact case once exec'd a real server out of a --dry-run test).
    args = _parse(["serve", MODEL, "--bench", "--dry-run", "--random-input-len", "32", "--gpu-memory-utilization", "0.8"])
    handle_serve(args)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2, out
    assert "--bench" not in out[0] and "--dry-run" not in out[0]
    assert "--gpu-memory-utilization 0.8" in out[0]
    assert "--random-input-len 32" in out[1]  # bench param, not forwarded to serve
    assert "--random-input-len" not in out[0]


def test_verbatim_passthrough_after_double_dash(capsys):
    args = _parse(["serve", MODEL, "--bench", "--dry-run", "--", "--port", "8222", "--seed", "3"])
    handle_serve(args)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2
    assert out[0].startswith(f"vllm serve {MODEL}")
    assert "--port 8222" in out[0] and "--seed 3" in out[0]
    assert " -- " not in out[0]  # the separator itself is not forwarded
    assert "http://localhost:8222" in out[1]  # bench targets the passthrough port


def test_dry_run_serve_only(capsys):
    args = _parse(["serve", MODEL, "--dry-run"])
    handle_serve(args)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert "bench" not in out[0]


def test_bench_seed_is_distinct_from_vllm_seed(capsys):
    args = _parse(["serve", MODEL, "--bench", "--dry-run", "--bench-seed", "9", "--seed", "1"])
    handle_serve(args)
    out = capsys.readouterr().out.strip().splitlines()
    assert "--seed 1" in out[0]  # vllm serve gets --seed
    assert "--seed 9" in out[1]  # the bench client gets --bench-seed


def test_serve_cmd_generate_branch():
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=[], generate=True)
    assert cmd[cmd.index("--runner") + 1] == "generate"
    assert '{"architectures": ["EmmyGenModel"]}' in cmd
    assert cmd[cmd.index("--dtype") + 1] == "float16"  # forced for seam coherence
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "4096"  # capped at the dynamic-dim limit
    assert "--enforce-eager" in cmd


def test_serve_cmd_generate_rejects_incompatible_dtype():
    with pytest.raises(ValueError, match="fp16"):
        build_serve_cmd(MODEL, stock=False, vllm_args=["--dtype", "bfloat16"], generate=True)


def test_serve_cmd_generate_rejects_oversized_batched_tokens():
    with pytest.raises(ValueError, match="dynamic-dim cap"):
        build_serve_cmd(MODEL, stock=False, vllm_args=["--max-num-batched-tokens", "8192"], generate=True)


def test_serve_cmd_generate_honors_explicit_batched_tokens():
    cmd = build_serve_cmd(MODEL, stock=False, vllm_args=["--max-num-batched-tokens", "2048"], generate=True)
    assert cmd.count("--max-num-batched-tokens") == 1  # the user's, no added default
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "2048"


def test_serve_generate_bench_is_rejected():
    args = _parse(["serve", MODEL, "--generate", "--bench", "--dry-run"])
    with pytest.raises(SystemExit):
        handle_serve(args)
