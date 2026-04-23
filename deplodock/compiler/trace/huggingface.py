"""Thin wrappers that make HuggingFace CausalLM models trace-friendly.

The goal is a module whose ``forward(input_ids)`` runs the full model and
returns logits, without HF's dynamic causal-mask construction polluting
the FX graph. The mask is precomputed and stapled on as a buffer; HF's
``_update_causal_mask`` hooks are neutralised before export.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch.nn as nn


def collect_const_feed(module: nn.Module, const_targets: dict[str, str]) -> dict[str, np.ndarray]:
    """Build a feed dict for every graph constant from the tracer.

    ``const_targets`` maps each placeholder name (e.g. ``p_attn_q_proj_weight``)
    to its dotted attribute path on ``module`` (e.g.
    ``self_attn.q_proj.weight``). That mapping is produced by
    ``trace_module_with_constants`` and is authoritative — ``torch.export``
    may drop prefixes like ``self_`` when naming placeholders, which
    string-only name mangling cannot reconstruct.

    Values are flattened float32 arrays; integer buffers round-trip
    through float without loss for values below 2**24.
    """
    import numpy as np

    state = dict(module.named_parameters())
    state.update(dict(module.named_buffers()))
    feed: dict[str, np.ndarray] = {}
    for placeholder_name, attr_path in const_targets.items():
        t = state.get(attr_path)
        if t is None:
            raise KeyError(f"constant {placeholder_name!r} -> {attr_path!r} not found on module")
        feed[placeholder_name] = t.detach().cpu().float().numpy().astype(np.float32).flatten()
    return feed


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
