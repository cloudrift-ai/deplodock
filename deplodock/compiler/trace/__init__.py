"""Frontend capture: PyTorch FX / HuggingFace → deplodock Graph IR.

Two modules:

- :mod:`.torch` — ``torch.export`` + FX graph walker that produces a
  ``Graph`` populated with frontend ops (``LinearOp``, ``MatmulOp``,
  ``SdpaOp``, elementwise, …). Entry points: ``trace_module``,
  ``trace_module_with_constants``, ``has_torch``.
- :mod:`.huggingface` — thin ``nn.Module`` adapter that makes HF
  ``CausalLM`` models trace-clean (precomputed causal mask,
  short-circuits HF's dynamic mask builder). Entry points:
  ``build_full_model_wrapper``, ``collect_const_feed``.
"""
