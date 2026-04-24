"""Thin wrappers that make HuggingFace CausalLM models trace-friendly.

The goal is a module whose ``forward(input_ids)`` runs the full model and
returns logits, without HF's dynamic causal-mask construction polluting
the FX graph. The mask is precomputed and stapled on as a buffer; HF's
``_update_causal_mask`` hooks are neutralised before export.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn


def build_full_model_wrapper(model, seq_len: int, dtype) -> nn.Module:
    """Return an ``nn.Module`` with ``forward(input_ids) -> logits``.

    The returned module carries a precomputed ``(1, 1, seq_len, seq_len)``
    causal mask (zeros on/below diagonal, ``-inf`` above) and short-circuits
    the HF mask machinery so the traced graph is free of mask-construction
    ops (arange/cumsum/diff/eq/le/__and__/new_ones/index/ne).
    """
    import torch
    import torch.nn as nn

    class FullModelWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            mask = torch.zeros((seq_len, seq_len), dtype=dtype)
            mask.masked_fill_(torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1), float("-inf"))
            self.register_buffer("causal_mask", mask[None, None, :, :])
            # Precompute position_ids so HF doesn't call torch.arange at forward time.
            self.register_buffer("position_ids", torch.arange(seq_len, dtype=torch.long)[None, :])
            # Short-circuit HF's dynamic mask builder(s). Different transformers
            # releases use different names; patch whichever exists.
            inner = getattr(model, "model", model)
            for attr in ("_update_causal_mask", "_prepare_4d_causal_attention_mask"):
                if hasattr(inner, attr):
                    setattr(inner, attr, _PassThroughMask(self))

        def forward(self, input_ids):
            out = self.model(
                input_ids=input_ids,
                attention_mask=self.causal_mask,
                position_ids=self.position_ids,
                use_cache=False,
                return_dict=False,
            )
            return out[0]

    return FullModelWrapper()


class _PassThroughMask:
    """Callable that returns the wrapper's precomputed causal mask verbatim."""

    def __init__(self, wrapper):
        self._wrapper = wrapper

    def __call__(self, *_, **__):
        return self._wrapper.causal_mask
