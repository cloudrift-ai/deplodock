"""``deplodock generate`` — standalone naive generation oracle (Phase 0 of
``plans/generative-inference-support.md``).

Re-runs the whole growing prefix each step (O(S^2)) on the deplodock CUDA backend with
**no vLLM** — deplodock controls the loop, so this is the token-for-token correctness
reference every later (vLLM) phase verifies against, and the seed of the eventual
standalone server. Runs the unquantized fp16 whole-model path.

The host loop (:func:`generate`) is pure and unit-testable with any ``logits_fn``;
:func:`handle_generate` wires it to a compiled fp16 program (:class:`_CompiledLM`).
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def register_generate_command(subparsers):
    parser = subparsers.add_parser("generate", help="Generate text from a model (standalone oracle; no vLLM)")
    parser.add_argument("model", help="HuggingFace model ID")
    parser.add_argument("--prompt", default="Hello, ", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k filter (0 = off)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p / nucleus filter (1.0 = off)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling RNG seed")
    parser.add_argument("--chat", action="store_true", help="Apply the tokenizer chat template to the prompt")
    parser.add_argument("--seq-len", type=int, default=32, help="Example seq_len for the dynamic trace")
    parser.set_defaults(func=handle_generate)


def generate(logits_fn, prompt_ids, *, max_new_tokens, eos_id, sampler):
    """Pure host generate loop. ``logits_fn(ids: list[int]) -> np.ndarray`` returns the
    next-token logits ``[vocab]`` for the given prefix; ``sampler(logits) -> int`` picks a
    token. Appends each sampled token and re-feeds the grown prefix, stopping at ``eos_id``
    or after ``max_new_tokens``. Returns the generated token ids (excluding the prompt)."""
    ids = list(prompt_ids)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        logits = logits_fn(ids)
        token = sampler(logits)
        generated.append(token)
        ids.append(token)
        if eos_id is not None and token == eos_id:
            break
    return generated


def handle_generate(args):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("torch and transformers are required: pip install torch transformers")
        sys.exit(1)

    from deplodock.serving.sampling import Sampler, apply_chat_template

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    lm = _CompiledLM.create(args.model, seq_len=args.seq_len)

    prompt_ids = apply_chat_template(tokenizer, args.prompt) if args.chat else tokenizer.encode(args.prompt)
    sampler = Sampler(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed)

    generated = generate(
        lm.logits,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eos_id=tokenizer.eos_token_id,
        sampler=sampler,
    )
    print(tokenizer.decode(generated, skip_special_tokens=True))


def _step_feed(ids_name, mask_name, pos_name, np_dtype, token_ids):
    """Build the per-step input feed (input_ids / additive causal mask / position_ids)
    sized to the current prefix length, keyed by the compiled graph's input names."""
    import numpy as np

    s = len(token_ids)
    ids = np.asarray(token_ids, dtype=np.int64).reshape(1, s)
    mask = np.triu(np.full((s, s), -np.inf, dtype=np.float32), k=1).astype(np_dtype)[None, None]
    pos = np.arange(s, dtype=np.int64).reshape(1, s)
    return {ids_name: ids, mask_name: mask, pos_name: pos}


class _CompiledLM:
    """Compiled fp16 whole-model program exposing ``logits(ids) -> np.ndarray[vocab]``.

    Traces ``AutoModelForCausalLM`` through the dynamic whole-model wrapper (full logits
    ``[1, S, vocab]``), compiles on the CUDA backend, binds fp16 constants, and re-runs the
    whole prefix each call via the serving ``rebind`` path (one compiled dynamic-seq_len
    program, request after request); ``logits()`` slices the final position on the host.
    Mirrors ``DeplodockForwardRunner.create``."""

    def __init__(self, program, input_names, output_name):
        self._program = program
        self._ids_name, self._mask_name, self._pos_name = input_names
        self._output_name = output_name

    @classmethod
    def create(cls, model_id, *, seq_len=32):
        import torch
        from transformers import AutoModelForCausalLM

        logger.info("[generate] loading %s (fp16, CPU trace)...", model_id)
        with torch.device("cpu"):
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
            model.eval()
            return cls.from_model(model, seq_len=seq_len)

    @classmethod
    def from_model(cls, model, *, seq_len=32):
        """Build from an already-loaded fp16 CausalLM module (the network-free path used
        by hermetic random-weight tests). ``model`` must be on CPU for the trace."""
        import numpy as np
        import torch

        from deplodock.compiler.backend.cuda.backend import CudaBackend
        from deplodock.compiler.backend.cuda.program import CompiledProgram
        from deplodock.compiler.backend.gpu_lock import gpu_lock
        from deplodock.compiler.loader.binder import bind_constants
        from deplodock.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs
        from deplodock.compiler.trace.huggingface import build_causal_mask, build_full_model_wrapper
        from deplodock.compiler.trace.torch import trace_module

        dtype = torch.float16
        np_dtype = np.dtype("float16")
        with torch.device("cpu"):
            # Full-logits wrapper: lm_head over all S positions, with the last row sliced
            # on the HOST in ``logits()``. The in-graph slice (``slice_last_logits=True``)
            # makes lm_head an M=1 *demoted* matmul that does NOT lower to CUDA on the cold
            # path (no learned prior) — leftover LoopOp 'linear_7'. The oracle is O(S^2)
            # regardless, so full logits + a host slice is the correct, cold-compilable cut.
            wrapper = build_full_model_wrapper(model, seq_len, dtype, dynamic=True)
            specs = parse_position_specs(
                ["seq_len@input_ids:1", "seq_len@attention_mask:2", "seq_len@attention_mask:3", "seq_len@position_ids:1"]
            )
            example = (
                torch.zeros((1, seq_len), dtype=torch.long),
                build_causal_mask(seq_len, dtype),
                torch.arange(seq_len).unsqueeze(0),
            )
            graph = trace_module(wrapper, example, dynamic_shapes=build_torch_dynamic_shapes(specs))

        logger.info("[generate] compiling...")
        compiled = CudaBackend(tune_db="auto").compile(graph)
        if len(compiled.inputs) != 3:
            raise RuntimeError(f"expected 3 graph inputs (input_ids, attention_mask, position_ids), got {compiled.inputs}")
        ids_name, mask_name, pos_name = compiled.inputs
        output_name = compiled.outputs[0]

        # fp16 constant carriers (via float32 numpy — safe for bf16 checkpoints); the
        # binder is dtype-agnostic, the graph's fp16 buffers cast at materialization.
        sources: dict[str, np.ndarray] = {}
        for path, t in wrapper.named_parameters(remove_duplicate=False):
            sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
        for path, t in wrapper.named_buffers(remove_duplicate=False):
            sources[path] = t.detach().cpu().to(torch.float32).numpy().astype(np_dtype, copy=False)
        const_feed = bind_constants(compiled, sources)

        feed = _step_feed(ids_name, mask_name, pos_name, np_dtype, [0] * seq_len)
        with gpu_lock():
            program = CompiledProgram.build(compiled, {**const_feed, **feed})
        del model, wrapper, sources, const_feed
        return cls(program, (ids_name, mask_name, pos_name), output_name)

    def logits(self, token_ids):
        """Next-token logits for the prefix ``token_ids`` (a list of ints). Re-binds the
        whole prefix at its current length and runs the program once (the O(S^2) oracle)."""
        import numpy as np

        from deplodock.compiler.backend.gpu_lock import gpu_lock

        s = len(token_ids)
        feed = _step_feed(self._ids_name, self._mask_name, self._pos_name, np.dtype("float16"), list(token_ids))
        with gpu_lock():
            self._program.rebind(feed)  # resolves seq_len from the input shapes
            self._program.run_once()
            out = self._program.outputs({"seq_len": s})[self._output_name]  # [1, S, vocab]
        return out[0, -1, :].astype(np.float32)  # next-token logits [vocab]
