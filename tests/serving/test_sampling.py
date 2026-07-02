"""Unit tests for the Phase-0 pure sampler + chat-template helper (no GPU/vLLM)."""

import numpy as np

from emmy.serving.sampling import Sampler, apply_chat_template, greedy


def test_greedy_picks_argmax():
    logits = np.array([0.1, 3.0, 0.2, -1.0])
    assert greedy(logits) == 1


def test_sampler_temperature_zero_is_greedy():
    logits = np.array([0.1, 3.0, 0.2, -1.0])
    sampler = Sampler(temperature=0.0)
    assert all(sampler(logits) == 1 for _ in range(5))


def test_sampler_seed_is_reproducible():
    logits = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    a = [Sampler(temperature=1.0, seed=7)(logits) for _ in range(20)]
    b = [Sampler(temperature=1.0, seed=7)(logits) for _ in range(20)]
    assert a == b  # deterministic given the seed
    assert 0 <= min(a) and max(a) <= 4


def test_top_k_restricts_support():
    # token 0 dominates, token 3 is second; with top-1 only token 0 is ever drawn.
    logits = np.array([5.0, 0.0, 0.0, 2.0])
    sampler = Sampler(temperature=1.0, top_k=1, seed=0)
    assert {sampler(logits) for _ in range(50)} == {0}


def test_top_p_restricts_support():
    # one near-certain token: nucleus p=0.5 keeps only it.
    logits = np.array([10.0, 0.0, 0.0, 0.0])
    sampler = Sampler(temperature=1.0, top_p=0.5, seed=0)
    assert {sampler(logits) for _ in range(50)} == {0}


class _FakeTokenizer:
    def __init__(self, chat_template):
        self.chat_template = chat_template

    def encode(self, text):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert add_generation_prompt and tokenize
        return [1, 2, 3, len(messages)]


def test_apply_chat_template_uses_template_when_present():
    tok = _FakeTokenizer(chat_template="{{ messages }}")
    assert apply_chat_template(tok, "hi") == [1, 2, 3, 1]
    assert apply_chat_template(tok, "hi", system="sys") == [1, 2, 3, 2]


def test_apply_chat_template_falls_back_to_encode():
    tok = _FakeTokenizer(chat_template=None)
    assert apply_chat_template(tok, "hi") == [ord("h"), ord("i")]
