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


def build_full_model_wrapper(model, seq_len: int, dtype, *, dynamic: bool = False, slice_last_logits: bool = False) -> nn.Module:
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

    Dynamic mode replaces the rotary embedding too — with cos/sin
    precomputed for positions ``0..DYNAMIC_DIM_MAX-1`` and sliced to the
    runtime seq_len in-graph (``_SlicedRotary``).

    ``slice_last_logits`` (dynamic only, the Phase-0 generation oracle): instead of
    returning the whole-prefix logits ``[1, S, vocab]``, run the trunk to hidden states,
    slice the **final** position (``hidden[:, -1:, :]``), and apply ``lm_head`` to that
    one row → ``[1, 1, vocab]``. The generate loop only needs the next-token logits, so
    this avoids the O(S·vocab) lm_head over every prefix position and the full-buffer host
    copy each step. NOTE: the ``hidden[:, -1:, :]`` slice makes lm_head an **M=1 demoted
    matmul** that the *cold* lowering path (no learned prior) does not turn into a CUDA
    kernel — it lowers only once a prior/golden covers that shape. So the standalone
    generation oracle currently traces full logits and slices the last row on the host;
    this flag is the (correct, prior-dependent) optimized form. Keeping HF's in-graph
    rotary instead silently breaks: its ``inv_freq`` buffer is
    ``persistent=False`` and doesn't survive ``torch.export`` with its
    real value, so the traced cos/sin constant-fold to ``cos=1, sin=0``
    and RoPE degenerates to identity. (The bug was invisible to the
    zero-``input_ids`` accuracy check: identical value rows make the
    attention output independent of the attention weights.) The sliced
    buffer assumes positions are ``0..S-1`` — true for full-sequence
    prefill, which is the only dynamic-mode use.
    """
    import torch
    import torch.nn as nn

    if slice_last_logits and not dynamic:
        raise ValueError("slice_last_logits requires dynamic=True (the generation oracle is dynamic-seq_len)")

    class _PassThroughRotary(nn.Module):
        """Replaces the model's ``rotary_emb`` and returns precomputed real
        ``(cos, sin)`` — an ``nn.Module`` (not a bare callable) since the
        attribute it overrides is a registered submodule. Buffers live on this
        module (no wrapper backref) so it stays a clean leaf submodule."""

        def __init__(self, cos, sin) -> None:
            super().__init__()
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)

        def forward(self, *_, **__):
            return self.cos, self.sin

    class _SlicedRotary(nn.Module):
        """Dynamic-mode rotary replacement: cos/sin precomputed out to
        ``DYNAMIC_DIM_MAX`` positions, sliced to the runtime seq_len read off
        the ``position_ids`` argument — the slice end is a SymInt, so the
        traced graph keeps real rotary values at every seq_len."""

        def __init__(self, cos, sin) -> None:
            super().__init__()
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)

        def forward(self, x, position_ids, *_, **__):
            s = position_ids.shape[1]
            return self.cos[:, :s], self.sin[:, :s]

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
            # Short-circuit the rotary embedding in static mode. Its cos/sin are
            # ``inv_freq @ position_ids`` under ``@torch.no_grad`` / float32
            # autocast; ``torch.export`` folds that to ``cos=1, sin=0`` (the
            # ``inv_freq`` buffer, ``persistent=False``, doesn't survive tracing
            # with its real value), so RoPE degenerates to identity. Compute the
            # real per-position cos/sin eagerly and return them verbatim — the
            # trace then captures correct constant tensors. Dynamic mode hits
            # the same folding, so it precomputes out to DYNAMIC_DIM_MAX and
            # slices to the runtime seq_len in-graph (_SlicedRotary below).
            rotary = getattr(inner, "rotary_emb", None)
            if not dynamic and rotary is not None:
                with torch.no_grad():
                    sample = torch.zeros((1, seq_len, model.config.hidden_size), dtype=dtype)
                    cos, sin = rotary(sample, self.position_ids)
                inner.rotary_emb = _PassThroughRotary(cos, sin)
            elif rotary is not None:
                from emmy.compiler.trace.dynamic import DYNAMIC_DIM_MAX  # noqa: PLC0415

                # +1: torch.export guards a symbolic slice end STRICTLY below
                # the sliced extent (``cos[:, :S]`` with S == extent would
                # specialize), so the buffer carries one extra position.
                n_pos = DYNAMIC_DIM_MAX + 1
                with torch.no_grad():
                    sample = torch.zeros((1, n_pos, model.config.hidden_size), dtype=dtype)
                    full_pos = torch.arange(n_pos, dtype=torch.long)[None, :]
                    cos, sin = rotary(sample, full_pos)
                inner.rotary_emb = _SlicedRotary(cos, sin)

        if dynamic and slice_last_logits:

            def forward(self, input_ids, attention_mask, position_ids):
                # Generation oracle: run the trunk, then apply lm_head to ONLY the
                # final position so the output is [1, 1, vocab] (next-token logits),
                # not the whole-prefix [1, S, vocab].
                trunk = getattr(self.model, "model", self.model)
                out = trunk(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=False,
                )
                hidden = out[0]  # [1, S, H], final norm already applied by the trunk
                last = hidden[:, -1:, :]  # [1, 1, H] — the next-token position
                head = getattr(self.model, "lm_head", None)
                if head is None:
                    head = self.model.get_output_embeddings()
                return head(last)  # [1, 1, vocab]

        elif dynamic:

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


def build_layer_wrapper(block, rotary_emb, hidden_size: int, dtype, *, layer_type=None) -> nn.Module:
    """Trace-friendly single-decoder-layer wrapper for dynamic mode:
    ``forward(x)`` slices rotary cos/sin to the runtime seq_len and calls
    ``block(x, position_embeddings=(cos, sin))``.

    The static per-layer trace passes concrete ``(cos, sin)`` kwargs sized to
    the trace seq_len — exactly the specialisation dynamic mode must avoid. So
    this wrapper precomputes cos/sin out to ``DYNAMIC_DIM_MAX`` positions as
    buffers and slices them in-graph by ``x.shape[1]`` (a SymInt under
    ``torch.export``), the same trick as the whole-model ``_SlicedRotary``.
    The sliced buffer assumes positions ``0..S-1`` — full-sequence prefill,
    the only dynamic-mode use. The forward arg is named ``x``, so the CLI
    spec is ``--dynamic seq_len@x:1``.

    ``layer_type`` feeds rotary modules that key cos/sin on the layer's
    attention type (e.g. Gemma's sliding/global split); ``None`` for the
    common single-rope architectures."""
    import torch
    import torch.nn as nn

    from emmy.compiler.trace.dynamic import DYNAMIC_DIM_MAX

    # +1: torch.export guards a symbolic slice end STRICTLY below the sliced
    # extent (``cos[:, :s]`` with s == extent would specialize) — same as the
    # whole-model dynamic wrapper's rotary buffer.
    n_pos = DYNAMIC_DIM_MAX + 1
    with torch.no_grad():
        sample = torch.zeros((1, n_pos, hidden_size), dtype=dtype)
        full_pos = torch.arange(n_pos, dtype=torch.long)[None, :]
        try:
            cos, sin = rotary_emb(sample, full_pos, layer_type)
        except TypeError:
            cos, sin = rotary_emb(sample, full_pos)

    class LayerWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = block
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)

        def forward(self, x):
            s = x.shape[1]
            return self.block(x, position_embeddings=(self.cos[:, :s], self.sin[:, :s]))

    return LayerWrapper()


def build_attention_split_wrapper(block):
    """Carve SDPA out of one HF decoder layer (Phase 1). Returns ``(pre, post)`` ``nn.Module``s over
    the flattened **``[num_tokens, H]``** per-token layout:

    - ``pre(hidden[T, H]) -> (q, k, v)`` runs ``input_layernorm`` → separate
      ``q_proj`` / ``k_proj`` / ``v_proj`` → reshape-into-heads → per-head ``q_norm`` /
      ``k_norm`` (Qwen3 only), and returns **un-rotated** q,k,v in the 2-D seam ABI
      ``q[T, Hq·D]``, ``k/v[T, Hkv·D]`` — exactly what vLLM's ``Attention.forward`` consumes.
      RoPE is applied downstream (by vLLM, or by the test/oracle reference).
    - ``post(attn_out[T, Hq·D], residual[T, H]) -> layer_out[T, H]`` runs ``o_proj`` →
      ``residual +`` → ``post_attention_layernorm`` → ``mlp`` → second residual.

    Reads the block's OWN submodules (not a ``self_attn`` substitution — HF
    ``self_attn.forward`` returns ``(attn_output, weights)``, and the block adds the
    residual after it, so swapping ``self_attn`` can't yield a clean pre-graph). NOTE: HF's
    ``.view(B,S,-1,D).transpose(1,2)`` assumes a ``[batch, seq, hidden]`` input; on the
    flattened ``[T, H]`` layout the reshape is ``.view(T, n_heads, D)`` with **no transpose**.
    Qwen3 (q/k norm) and Llama (no q/k norm) are both handled; the architecture difference is
    just the presence of ``q_norm`` / ``k_norm``."""
    import torch.nn as nn

    attn = block.self_attn
    head_dim = attn.head_dim
    num_heads = attn.q_proj.out_features // head_dim
    num_kv_heads = attn.k_proj.out_features // head_dim
    q_norm = getattr(attn, "q_norm", None)
    k_norm = getattr(attn, "k_norm", None)

    class Pre(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = block.input_layernorm
            self.q_proj, self.k_proj, self.v_proj = attn.q_proj, attn.k_proj, attn.v_proj
            self.q_norm, self.k_norm = q_norm, k_norm

        def forward(self, hidden):
            h = self.input_layernorm(hidden)  # [T, H]
            t = h.shape[0]
            q = self.q_proj(h).view(t, num_heads, head_dim)  # [T, Hq, D] — NO transpose
            k = self.k_proj(h).view(t, num_kv_heads, head_dim)
            v = self.v_proj(h).view(t, num_kv_heads, head_dim)
            if self.q_norm is not None:
                q = self.q_norm(q)  # per-head RMSNorm over D
                k = self.k_norm(k)
            return q.reshape(t, num_heads * head_dim), k.reshape(t, num_kv_heads * head_dim), v.reshape(t, num_kv_heads * head_dim)

    class Post(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.o_proj = attn.o_proj
            self.post_attention_layernorm = block.post_attention_layernorm
            self.mlp = block.mlp

        def forward(self, attn_out, residual):
            h = residual + self.o_proj(attn_out)  # residual1 + o_proj(SDPA)
            return h + self.mlp(self.post_attention_layernorm(h))  # residual2 + MLP

    return Pre(), Post()


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
