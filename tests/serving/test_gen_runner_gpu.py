"""Phase-2 multi-layer host-stitch test for ``DeplodockGenRunner`` (no vLLM).

``perf``-marked: needs CUDA + cupy. Builds a tiny multi-layer Qwen3, then runs a whole-model
Python stitch — ``embed`` → per layer (deplodock ``pre`` kernels → reconstruct RoPE →
reference causal GQA torch SDPA → deplodock ``post`` kernels) → ``final_norm`` → lm_head —
and checks the stitched logits against eager. This is the dress rehearsal for the vLLM
forward (Phase 3) without vLLM's runner, isolating the deplodock↔attention interleave.
fp32 (carve correctness is dtype-independent; the fp16 path is covered by the Phase-0 oracle).
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda")]


def _repeat_kv(x, n_rep):
    b, h, t, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def _reference_attention(runner, q_np, k_np, v_np, cos, sin, mask, apply_rotary):
    """The carve's attention seam, reconstructed for the host stitch: 2-D q/k/v → [1,H,T,D]
    → RoPE → causal GQA SDPA → 2-D attn_out. (Phase 3 replaces this with vLLM paged attention.)"""
    import torch
    import torch.nn.functional as F

    d, hq, hkv = runner.head_dim, runner.num_heads, runner.num_kv_heads
    t = q_np.shape[0]
    q = torch.from_numpy(np.ascontiguousarray(q_np)).view(t, hq, d).transpose(0, 1).unsqueeze(0)
    k = torch.from_numpy(np.ascontiguousarray(k_np)).view(t, hkv, d).transpose(0, 1).unsqueeze(0)
    v = torch.from_numpy(np.ascontiguousarray(v_np)).view(t, hkv, d).transpose(0, 1).unsqueeze(0)
    q, k = apply_rotary(q, k, cos, sin)
    k, v = _repeat_kv(k, hq // hkv), _repeat_kv(v, hq // hkv)
    attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=runner.scaling)  # [1, Hq, T, D]
    return attn.transpose(1, 2).reshape(t, hq * d).numpy()


def test_gen_runner_stitch_matches_eager():
    pytest.importorskip("cupy")
    import torch
    import transformers

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    from deplodock.compiler.trace.huggingface import build_causal_mask
    from deplodock.serving.gen_runner import DeplodockGenRunner

    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        use_sliding_window=False,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(config).eval()  # fp32; build_attention_split_wrapper does NOT mutate it

    runner = DeplodockGenRunner.from_model(model, dtype_str="float32")
    assert runner.num_layers == config.num_hidden_layers

    t = 7
    input_ids = list(range(1, t + 1))
    position_ids = torch.arange(t).unsqueeze(0)
    mask = build_causal_mask(t, torch.float32)  # [1, 1, T, T]
    cos, sin = model.model.rotary_emb(torch.zeros(1, t, config.hidden_size), position_ids)

    # --- deplodock host stitch ---
    hidden = runner.embed(input_ids)  # np [T, H]
    for layer in range(runner.num_layers):
        residual = hidden
        q, k, v = runner.forward_layer_pre(layer, hidden, position_ids)
        attn_out = _reference_attention(runner, q, k, v, cos, sin, mask, apply_rotary_pos_emb)
        hidden = runner.forward_layer_post(layer, attn_out, residual)
    hidden = runner.final_norm(hidden)
    with torch.no_grad():
        logits_dep = model.lm_head(torch.from_numpy(np.ascontiguousarray(hidden))).numpy()  # [T, vocab]

        # --- eager reference ---
        eager = model(torch.tensor([input_ids], dtype=torch.long)).logits[0].numpy()  # [T, vocab]

    assert logits_dep.shape == eager.shape
    np.testing.assert_allclose(logits_dep, eager, rtol=2e-3, atol=2e-3)
    # next-token greedy agrees too
    assert int(np.argmax(logits_dep[-1])) == int(np.argmax(eager[-1]))


def test_gen_runner_device_path_matches_host():
    """The device-resident decode path (``run_device`` / ``*_device``) must match the host numpy
    path for the real ``T`` rows (``T <= decode_bucket``) — stale prefix padding never leaks
    because pre/post are per-token-independent. Regression guard for Phase A
    (``plans/generative-device-resident-decode.md``)."""
    pytest.importorskip("cupy")
    import torch
    import transformers

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from deplodock.serving.gen_runner import DeplodockGenRunner

    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        use_sliding_window=False,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(config).eval()
    runner = DeplodockGenRunner.from_model(model, dtype_str="float32", decode_bucket=16)
    if not runner.has_device_decode:
        pytest.skip("decode-bucket programs unavailable for this shape")

    t = 5  # <= decode_bucket
    ids = list(range(1, t + 1))
    ids_t = torch.tensor(ids, dtype=torch.long, device="cuda")
    attn_width = runner.num_heads * runner.head_dim

    # embed / pre / post run the SAME GPU kernels on both paths → bit-identical for the real rows.
    h_np = runner.embed(ids)
    h_t = runner.embed_device(ids_t)
    np.testing.assert_array_equal(h_np, h_t.cpu().numpy())

    q_np, k_np, v_np = runner.forward_layer_pre(0, h_np)
    q, k, v = runner.forward_layer_pre_device(0, h_t)
    np.testing.assert_array_equal(q_np, q.cpu().numpy())
    np.testing.assert_array_equal(k_np, k.cpu().numpy())
    np.testing.assert_array_equal(v_np, v.cpu().numpy())

    attn = np.random.RandomState(0).randn(t, attn_width).astype(runner._np_dtype)
    out_np = runner.forward_layer_post(0, attn, h_np)
    out_t = runner.forward_layer_post_device(0, torch.from_numpy(attn).cuda(), h_t)
    np.testing.assert_array_equal(out_np, out_t.cpu().numpy())

    # final_norm runs a torch module CPU (host) vs the deep-copied CUDA module (device) — fp32 ULPs.
    fn_np = runner.final_norm(h_np)
    fn_t = runner.final_norm_device(h_t)
    np.testing.assert_allclose(fn_np, fn_t.cpu().numpy(), rtol=1e-4, atol=1e-4)
    # the host final_norm must still work AFTER the device path moved nothing in place (deepcopy):
    np.testing.assert_array_equal(fn_np, runner.final_norm(h_np))
