"""Hermetic equivalence test for the Phase-1 attention-split carve (no GPU/vLLM).

Proves the carve is correct: for one decoder layer, running ``pre`` → reconstruct RoPE →
external causal GQA torch SDPA → ``post`` reproduces the eager ``block(x)`` over the
flattened ``[num_tokens, H]`` layout. Exercises GQA (num_kv_heads < num_heads) and Qwen3's
per-head q/k norm. Pure eager, CPU, fp32 — no compile.
"""

import pytest


def _repeat_kv(x, n_rep):
    """[1, Hkv, T, D] -> [1, Hkv*n_rep, T, D] (GQA head expansion)."""
    b, h, t, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def _split_path_output(pre, post, attn, hidden2d, cos, sin, mask, apply_rotary):
    """Reference reconstruction: pre → rope → causal GQA SDPA → post. The [1,H,T,D] layout
    lives ONLY here (the carve's seam ABI is 2-D), per the plan."""
    import torch.nn.functional as F

    head_dim = attn.head_dim
    num_heads = attn.q_proj.out_features // head_dim
    num_kv = attn.k_proj.out_features // head_dim
    t = hidden2d.shape[0]

    q2d, k2d, v2d = pre(hidden2d)  # [T, Hq*D], [T, Hkv*D]
    # 2-D seam -> [1, n_heads, T, D] (no HF-style transpose hazard; explicit here).
    q = q2d.view(t, num_heads, head_dim).transpose(0, 1).unsqueeze(0)
    k = k2d.view(t, num_kv, head_dim).transpose(0, 1).unsqueeze(0)
    v = v2d.view(t, num_kv, head_dim).transpose(0, 1).unsqueeze(0)

    q, k = apply_rotary(q, k, cos, sin)  # reconstruct the RoPE the eager layer applies
    k = _repeat_kv(k, num_heads // num_kv)
    v = _repeat_kv(v, num_heads // num_kv)
    attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=attn.scaling)  # [1, Hq, T, D]
    attn_out = attn_out.transpose(1, 2).reshape(t, num_heads * head_dim)  # [T, Hq*D]
    return post(attn_out, hidden2d)


@pytest.mark.parametrize("arch", ["qwen3", "llama"])
def test_split_matches_eager_block(arch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from deplodock.compiler.trace.huggingface import build_attention_split_wrapper, build_causal_mask

    if arch == "qwen3":
        config = transformers.Qwen3Config(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            use_sliding_window=False,
        )
        model = transformers.Qwen3ForCausalLM(config)
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    else:
        config = transformers.LlamaConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        model = transformers.LlamaForCausalLM(config)
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    torch.manual_seed(0)
    model = model.eval()
    trunk = model.model
    block = trunk.layers[0]
    attn = block.self_attn

    t = 6
    hidden3d = torch.randn(1, t, config.hidden_size)
    position_ids = torch.arange(t).unsqueeze(0)
    cos, sin = trunk.rotary_emb(hidden3d, position_ids)  # [1, T, D]
    mask = build_causal_mask(t, torch.float32)  # [1, 1, T, T] additive

    with torch.no_grad():
        eager = block(hidden3d, position_embeddings=(cos, sin), attention_mask=mask)
        eager = eager[0] if isinstance(eager, tuple) else eager  # [1, T, H]

        pre, post = build_attention_split_wrapper(block)
        out = _split_path_output(pre, post, attn, hidden3d.squeeze(0), cos, sin, mask, apply_rotary_pos_emb)

    assert tuple(out.shape) == (t, config.hidden_size)
    torch.testing.assert_close(out, eager.squeeze(0), rtol=1e-4, atol=1e-4)


def test_pre_emits_2d_seam_shapes():
    """The pre wrapper's seam ABI: q[T, Hq*D], k/v[T, Hkv*D] (2-D, no transpose)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from deplodock.compiler.trace.huggingface import build_attention_split_wrapper

    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        use_sliding_window=False,
    )
    torch.manual_seed(0)
    block = transformers.Qwen3ForCausalLM(config).eval().model.layers[0]
    pre, _ = build_attention_split_wrapper(block)
    t = 5
    q, k, v = pre(torch.randn(t, config.hidden_size))
    assert tuple(q.shape) == (t, 4 * 16)  # [T, Hq*D]
    assert tuple(k.shape) == (t, 2 * 16)  # [T, Hkv*D]
    assert tuple(v.shape) == (t, 2 * 16)
