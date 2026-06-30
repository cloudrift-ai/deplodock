"""``DeplodockEmbedModel`` — the vLLM out-of-tree pooling model class.

Serve any causal-trunk embedding model (e.g. Qwen3-Embedding) with vLLM's
shell but deplodock-compiled kernels:

    vllm serve Qwen/Qwen3-Embedding-0.6B --runner pooling --enforce-eager \\
      --max-model-len 4096 --hf-overrides '{"architectures":["DeplodockEmbedModel"]}'

The class holds no ``nn.Parameter``s — weights live inside the compiled
program as graph constants, loaded by ``DeplodockForwardRunner.create`` at
engine start. vLLM's V1 engine treats a model with no ``Attention`` layers as
attention-free (empty KV-cache spec); ``attn_type = "encoder_only"`` turns
chunked prefill off, so every request reaches ``forward`` whole and
``positions == 0`` marks sequence starts. The runner computes everything
itself per sequence; vLLM's pooler then does last-token pooling + L2
normalization (+ matryoshka), identical to stock Qwen3-Embedding serving.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.models.interfaces import IsAttentionFree

from deplodock import config
from deplodock.serving.packed import split_spans
from deplodock.serving.runner import DeplodockForwardRunner

logger = logging.getLogger(__name__)

# vLLM's --dtype (mc.dtype, already resolved from `auto`) → the dtype the
# deplodock trunk computes at. Only fp16/fp32 are representable through the
# runner's numpy weight carrier, so anything else (e.g. bf16) downcasts to fp16.
_TRUNK_DTYPE = {torch.float16: "float16", torch.float32: "float32"}


def _trunk_dtype_str(torch_dtype) -> str:
    dtype_str = _TRUNK_DTYPE.get(torch_dtype)
    if dtype_str is None:
        logger.warning("[serving] --dtype %s unsupported by the deplodock trunk; computing in float16", torch_dtype)
        return "float16"
    return dtype_str


class DeplodockEmbedModel(nn.Module, IsAttentionFree):
    is_pooling_model = True
    default_seq_pooling_type = "LAST"
    attn_type = "encoder_only"  # whole-sequence prefills only (disables chunked prefill)

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        mc = vllm_config.model_config
        self.out_dtype = mc.dtype
        self.vocab_size = mc.get_vocab_size()
        self.pooler = DispatchPooler.for_embedding(mc.pooler_config)
        # Static mode (opt-in): static extents for both batch and seq_len. Batch cap
        # = vLLM's own max_num_seqs, so it's sized by --max-num-seqs (seq by
        # --max-model-len), not a deplodock-specific knob.
        batch = vllm_config.scheduler_config.max_num_seqs if config.serving_static() else 1
        self.runner = DeplodockForwardRunner.create(
            model_id=mc.model,
            max_seq_len=mc.max_model_len,
            dtype_str=_trunk_dtype_str(mc.dtype),
            batch=batch,
        )

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        # Packed (num_tokens_padded,) tensors: sequences flattened back-to-back,
        # positions 0-based per request. Everything stays on-device — each span's
        # ids slice straight into the runner (torch→cupy zero-copy) and hidden
        # states come back as torch CUDA tensors. Only the span boundaries need a
        # host read of positions (a tiny (num_tokens,) int vector); garbage
        # dummy-run batches survive via the vocab clamp + split_spans' chunking.
        # With batch_cap > 1 the whole step runs as one padded batched forward.
        n = int(input_ids.shape[0])
        ids = input_ids.clamp(0, self.vocab_size - 1).to(torch.int64)
        spans = [ids[a:b] for a, b in split_spans(positions.cpu().numpy().reshape(-1), self.runner.max_seq_len)]
        if self.runner.batch_cap > 1:
            chunks = self.runner.forward_hidden_states_batched(spans)
        else:
            chunks = [self.runner.forward_hidden_states(t) for t in spans]
        if chunks:
            out = torch.cat(chunks, dim=0).to(device=input_ids.device, dtype=self.out_dtype)
        else:
            out = torch.zeros((0, self.runner.hidden_size), device=input_ids.device, dtype=self.out_dtype)
        if out.shape[0] < n:  # pad back to num_tokens_padded
            out = torch.nn.functional.pad(out, (0, 0, 0, n - out.shape[0]))
        return out

    def embed_input_ids(self, input_ids):
        # Protocol stub (same as vLLM's Terratorch precedent): the compiled
        # program owns the real embedding table.
        return torch.empty((input_ids.shape[0], 0), device=input_ids.device, dtype=self.out_dtype)

    def load_weights(self, weights):
        # The wrapper has no nn.Parameters; the runner already loaded the
        # checkpoint itself. Not consuming the iterator skips reading the
        # safetensors entirely. An empty set satisfies the strict
        # all-params-initialized check trivially.
        return set()
