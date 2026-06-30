"""Frontend capture: PyTorch FX / HuggingFace → emmy Graph IR.

Two modules:

- :mod:`.torch` — ``torch.export`` + FX graph walker that produces a
  ``Graph`` populated with frontend ops (``LinearOp``, ``MatmulOp``,
  ``SdpaOp``, elementwise, …). Entry points: ``trace_module``,
  ``trace_module_with_constants``, ``has_torch``.
- :mod:`.huggingface` — thin ``nn.Module`` adapter that makes HF
  ``CausalLM`` models trace-clean (precomputed causal mask,
  short-circuits HF's dynamic mask builder). Entry point:
  ``build_full_model_wrapper``.
"""
