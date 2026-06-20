"""Hermetic loop-wiring tests for the Phase-0 ``generate`` host loop (no GPU/model).

The loop is driven by a fake ``logits_fn`` so we can assert the wiring — last-token
slice (the loop only consumes the returned logits), append, prefix growth, and the
EOS / max-new-tokens stop conditions — without compiling anything.
"""

import argparse

import numpy as np

from deplodock.commands.generate import generate, register_generate_command
from deplodock.serving.sampling import Sampler


def _onehot(token, vocab=16):
    logits = np.full(vocab, -10.0)
    logits[token] = 10.0
    return logits


def test_generate_greedy_follows_logits_fn():
    # logits_fn always points at the next integer after the last token → 4,5,6,...
    def logits_fn(ids):
        return _onehot((ids[-1] + 1) % 16)

    out = generate(logits_fn, [3], max_new_tokens=4, eos_ids=None, sampler=Sampler())
    assert out == [4, 5, 6, 7]


def test_generate_stops_at_eos():
    def logits_fn(ids):
        return _onehot(2)  # always emit token 2

    out = generate(logits_fn, [0], max_new_tokens=10, eos_ids={2}, sampler=Sampler())
    assert out == [2]  # stops immediately after emitting EOS


def test_generate_stops_at_any_eos_in_set():
    def logits_fn(ids):
        return _onehot(5)  # 5 is one of several terminators

    out = generate(logits_fn, [0], max_new_tokens=10, eos_ids={2, 5, 7}, sampler=Sampler())
    assert out == [5]


def test_generate_respects_max_new_tokens():
    def logits_fn(ids):
        return _onehot(5)

    out = generate(logits_fn, [0], max_new_tokens=3, eos_ids={999}, sampler=Sampler())
    assert out == [5, 5, 5]


def test_generate_feeds_growing_prefix():
    seen_lengths = []

    def logits_fn(ids):
        seen_lengths.append(len(ids))
        return _onehot(1)

    generate(logits_fn, [0, 0], max_new_tokens=3, eos_ids=None, sampler=Sampler())
    # prompt len 2, then one appended token per step: 2, 3, 4
    assert seen_lengths == [2, 3, 4]


def test_slice_last_logits_wrapper_matches_full_logits():
    """The generation wrapper (``slice_last_logits=True``) returns ``[1, 1, vocab]`` equal
    to the full model's final-position logits — pure eager, CPU, no compile."""
    import pytest

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from deplodock.compiler.trace.huggingface import build_causal_mask, build_full_model_wrapper

    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval()  # fp32 on CPU
    s = 6
    wrapper = build_full_model_wrapper(model, s, torch.float32, dynamic=True, slice_last_logits=True)

    ids = torch.arange(s).unsqueeze(0) % config.vocab_size
    mask = build_causal_mask(s, torch.float32)
    pos = torch.arange(s).unsqueeze(0)
    with torch.no_grad():
        sliced = wrapper(ids, mask, pos)  # [1, 1, vocab]
        full = model(ids, attention_mask=mask, position_ids=pos, use_cache=False).logits  # [1, S, vocab]
    assert tuple(sliced.shape) == (1, 1, config.vocab_size)
    torch.testing.assert_close(sliced[:, 0, :], full[:, -1, :], rtol=1e-4, atol=1e-4)


def test_register_generate_command_parses():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    register_generate_command(sub)

    from deplodock.commands.generate import handle_generate

    args = parser.parse_args(["generate", "some/model", "--max-new-tokens", "5", "--temperature", "0.7", "--chat"])
    assert args.func is handle_generate
    assert args.model == "some/model"
    assert args.max_new_tokens == 5
    assert args.temperature == 0.7
    assert args.chat is True
