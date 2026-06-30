"""``DeplodockGenModel`` — the vLLM out-of-tree **generative** model class (Phase 3 of
``plans/generative-inference-support.md``).

Serve a decoder-only chat model (Qwen3 / Llama) through deplodock-compiled per-layer
kernels with vLLM owning the API / sampler / scheduler / paged KV-cache:

    vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --runner generate --enforce-eager \\
      --dtype float16 --hf-overrides '{"architectures":["DeplodockGenModel"]}'

NOT ``IsAttentionFree``: it constructs real vLLM ``Attention`` layers (one per decoder
layer, unique ``prefix``) so vLLM allocates a KV-cache spec and runs paged attention. All
weight-bearing **trunk** compute (embed + per-layer pre/post + final norm) lives in the
deplodock ``DeplodockGenRunner``; vLLM owns only ``lm_head`` (loaded via ``load_weights``)
and applies RoPE through a ``get_rope`` module the model builds (a bare ``Attention`` does
none). ``forward`` brackets each vLLM attention call with two deplodock replays (``pre`` /
``post``); RoPE is applied between ``pre`` and ``self.attn`` (A2).

Numpy host I/O at the runner boundary (the per-layer host-sync interleave — Top risk #1);
the device zero-copy path is the Phase-4 optimization.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from deplodock.serving.gen_runner import DeplodockGenRunner
from deplodock.serving.vllm_model import _trunk_dtype_str

logger = logging.getLogger(__name__)


def _build_rotary(config, head_dim, max_position):
    """Construct the model's RoPE the way stock vLLM does — applying each architecture's
    default-theta mutation first (Qwen3 → 1e6, else a missing rope_theta silently falls
    back to 10000 → wrong logits)."""
    try:
        from vllm.transformers_utils.config import set_default_rope_theta

        if getattr(config, "model_type", None) == "qwen3":
            set_default_rope_theta(config, default_theta=1000000)
    except Exception:  # noqa: BLE001 — older vLLM without the helper; config carries theta already
        pass
    return get_rope(
        head_dim,
        max_position=max_position,
        rope_parameters=getattr(config, "rope_parameters", None),
        is_neox_style=True,
    )


class DeplodockGenModel(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        mc = vllm_config.model_config
        config = mc.hf_config
        self.config = config
        self.dtype = mc.dtype

        # The carve (pre/post + a plain causal vLLM Attention) assumes FULL causal attention.
        # Sliding-window / per-layer-sliding / dual-chunk variants would silently miscompute
        # (neither side here is window/chunk aware) — reject rather than mislead.
        if getattr(config, "use_sliding_window", False) and getattr(config, "sliding_window", None):
            raise NotImplementedError("DeplodockGenModel: sliding-window attention is not supported")
        layer_types = getattr(config, "layer_types", None)
        if layer_types and any(lt == "sliding_attention" for lt in layer_types):
            raise NotImplementedError("DeplodockGenModel: per-layer sliding attention is not supported")
        if getattr(config, "dual_chunk_attention_config", None):
            raise NotImplementedError("DeplodockGenModel: dual-chunk attention is not supported")

        # The flattened width T = num_tokens is the SUM of newly-scheduled tokens across all
        # requests per step (continuous batching), bounded by max_num_batched_tokens — NOT
        # max_model_len. It must stay within the compiler's dynamic-dim / RoPE-buffer cap.
        from deplodock.compiler.trace.dynamic import DYNAMIC_DIM_MAX

        max_batched = vllm_config.scheduler_config.max_num_batched_tokens
        if max_batched and max_batched > DYNAMIC_DIM_MAX:
            raise ValueError(
                f"max_num_batched_tokens={max_batched} exceeds DYNAMIC_DIM_MAX ({DYNAMIC_DIM_MAX}); "
                f"serve with --max-num-batched-tokens {DYNAMIC_DIM_MAX} or lower"
            )

        self.runner = DeplodockGenRunner.create(model_id=mc.model, dtype_str=_trunk_dtype_str(mc.dtype))
        n_layers = self.runner.num_layers
        head_dim = self.runner.head_dim

        # One real vLLM Attention per layer — unique prefix (vLLM keys static_forward_context
        # / cache-spec discovery by it and rejects duplicates). No weights.
        self.attn = nn.ModuleList(
            [
                Attention(
                    self.runner.num_heads,
                    head_dim,
                    self.runner.scaling,
                    num_kv_heads=self.runner.num_kv_heads,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.layers.{i}.self_attn.attn",
                )
                for i in range(n_layers)
            ]
        )
        # RoPE is NOT in Attention — one shared module (position-only, layer-independent).
        self.rotary_emb = _build_rotary(config, head_dim, getattr(config, "max_position_embeddings", 8192))

        # vLLM owns ONLY lm_head; the runner owns embed + the trunk.
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, quant_config=vllm_config.quant_config, prefix=f"{prefix}.lm_head"
        )
        self.logits_processor = LogitsProcessor(config.vocab_size, scale=getattr(config, "logit_scale", 1.0))

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        device = positions.device
        # clamp guards vLLM's _dummy_run garbage-id profiling batches (out-of-vocab → IndexError).
        ids = input_ids.clamp(0, self.config.vocab_size - 1)
        t = int(ids.shape[0])
        # Decode hot path (T <= bucket): device-resident, no host numpy round-trip (Phase A,
        # plans/generative-device-resident-decode.md). Prefill / larger T keeps the host path.
        if self.runner.has_device_decode and 0 < t <= self.runner.decode_bucket:
            return self._forward_device(ids, positions)
        ids_np = ids.cpu().numpy()
        hidden_np = self.runner.embed(ids_np)  # [T, H] numpy
        for layer in range(self.runner.num_layers):
            residual_np = hidden_np
            q_np, k_np, v_np = self.runner.forward_layer_pre(layer, hidden_np, positions)
            q = torch.from_numpy(np.ascontiguousarray(q_np)).to(device)
            k = torch.from_numpy(np.ascontiguousarray(k_np)).to(device)
            v = torch.from_numpy(np.ascontiguousarray(v_np)).to(device)
            q, k = self.rotary_emb(positions, q, k)  # A2: vLLM applies RoPE
            attn_out = self.attn[layer](q, k, v)  # vLLM paged attention (pulls attn_metadata from forward context)
            hidden_np = self.runner.forward_layer_post(layer, attn_out.detach().cpu().numpy(), residual_np)
        hidden_np = self.runner.final_norm(hidden_np)
        return torch.from_numpy(np.ascontiguousarray(hidden_np)).to(device)

    def _forward_device(self, ids, positions):
        """Device-resident decode forward (T <= decode_bucket): q/k/v and attn_out stay CUDA
        tensors through RoPE + vLLM attention — no per-layer numpy↔torch host hop."""
        hidden = self.runner.embed_device(ids)  # [T, H] CUDA
        for layer in range(self.runner.num_layers):
            residual = hidden
            q, k, v = self.runner.forward_layer_pre_device(layer, hidden)
            q, k = self.rotary_emb(positions, q, k)  # A2: vLLM applies RoPE
            attn_out = self.attn[layer](q, k, v)  # vLLM paged attention
            hidden = self.runner.forward_layer_post_device(layer, attn_out, residual)
        return self.runner.final_norm_device(hidden)

    def compute_logits(self, hidden_states, *args):
        return self.logits_processor(self.lm_head, hidden_states)

    def embed_input_ids(self, input_ids):
        # vLLM embedding hook → the runner owns embedding; on-device gather (no host hop).
        return self.runner.embed_device(input_ids.clamp(0, self.config.vocab_size - 1))

    def load_weights(self, weights):
        """vLLM owns ONLY ``lm_head`` (the runner already loaded embed + trunk). Load
        ``lm_head.weight`` from the checkpoint; when ``tie_word_embeddings`` the checkpoint
        may carry only ``embed_tokens.weight``, so accept that alias for the head."""
        param = self.lm_head.weight
        loader = getattr(param, "weight_loader", default_weight_loader)
        tied = getattr(self.config, "tie_word_embeddings", False)
        loaded: set[str] = set()
        for name, w in weights:
            if name == "lm_head.weight":
                loader(param, w)
                loaded.add("lm_head.weight")
            elif tied and name in ("model.embed_tokens.weight", "embed_tokens.weight") and "lm_head.weight" not in loaded:
                loader(param, w)
                loaded.add("lm_head.weight")
        return loaded
