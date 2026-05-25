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


def build_full_model_wrapper(model, seq_len: int, dtype, *, dynamic: bool = False) -> nn.Module:
    """Return an ``nn.Module`` with a trace-friendly forward.

    Static mode (``dynamic=False``, default): forward is
    ``forward(input_ids) -> logits`` and the module carries a precomputed
    ``(1, 1, seq_len, seq_len)`` causal mask + ``(1, seq_len)`` position_ids
    as buffers. HF's dynamic mask machinery is short-circuited so the
    traced graph is free of mask-construction ops (arange/cumsum/diff/
    eq/le/__and__/new_ones/index/ne).

    Dynamic mode (``dynamic=True``, plan M4): forward is
    ``forward(input_ids, attention_mask, position_ids) -> logits`` — the
    caller supplies the per-call mask + position_ids sized to the actual
    seq_len. The traced graph then has ``attention_mask`` and
    ``position_ids`` as inputs (shape ``(1, 1, seq_len, seq_len)`` and
    ``(1, seq_len)`` respectively); rewriting the seq_len dim to
    ``Dim('seq_len')`` post-trace yields a graph that compiles once and
    runs at any seq_len. ``seq_len`` is still used at construction time
    to seed the short-circuit hooks with a same-shape sentinel mask so
    HF's internal validation passes.
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

        if dynamic:

            def forward(self, input_ids, attention_mask, position_ids):
                # The caller controls mask + position_ids so the seq_len axis can
                # flow through to ``Dim('seq_len')`` after the rewrite step.
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=False,
                )
                return out[0]

        else:

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


def build_causal_mask(seq_len: int, dtype) -> torch.Tensor:  # noqa: F821
    """Return the ``(1, 1, seq_len, seq_len)`` causal mask the wrapper
    uses internally — exposed so callers in dynamic mode can construct a
    per-call mask sized to the actual seq_len."""
    import torch

    mask = torch.zeros((seq_len, seq_len), dtype=dtype)
    mask.masked_fill_(torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1), float("-inf"))
    return mask[None, None, :, :]


class _PassThroughMask:
    """Callable that returns the wrapper's precomputed causal mask verbatim."""

    def __init__(self, wrapper):
        self._wrapper = wrapper

    def __call__(self, *_, **__):
        return self._wrapper.causal_mask
