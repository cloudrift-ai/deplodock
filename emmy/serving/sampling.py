"""Pure-Python token sampling + chat-template helper for the standalone
``emmy generate`` oracle (Phase 0 of ``plans/generative-inference-support.md``).

No vLLM, no CUDA: every function operates on a 1-D ``logits`` vector (numpy), so the
whole module is unit-testable on CPU. ``Sampler`` is a callable ``logits -> token id``
covering greedy / temperature / top-k / top-p; ``apply_chat_template`` delegates to the
HF tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def greedy(logits: np.ndarray) -> int:
    """Argmax token id."""
    return int(np.argmax(logits))


def _top_k_mask(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep the ``k`` highest logits, set the rest to ``-inf``."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = np.partition(logits, -k)[-k]
    return np.where(logits < kth, -np.inf, logits)


def _top_p_mask(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus filter: keep the smallest set of highest-prob tokens whose
    cumulative probability reaches ``p`` (the top-1 token is always kept)."""
    if not 0.0 < p < 1.0:
        return logits
    order = np.argsort(logits)[::-1]
    probs = _softmax(logits[order])
    cum_before = np.cumsum(probs) - probs  # cumulative prob strictly before each token
    keep = cum_before < p
    keep[0] = True  # always keep the most-likely token
    masked = np.full_like(logits, -np.inf)
    masked[order[keep]] = logits[order[keep]]
    return masked


@dataclass
class Sampler:
    """Callable ``logits -> token id``. ``temperature <= 0`` is greedy (the default,
    used by the correctness oracle); otherwise temperature → top-k → top-p → multinomial,
    seeded for reproducibility."""

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, logits: np.ndarray) -> int:
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        if self.temperature <= 0.0:
            return greedy(logits)
        logits = logits / self.temperature
        logits = _top_k_mask(logits, self.top_k)
        logits = _top_p_mask(logits, self.top_p)
        probs = _softmax(logits)
        return int(self._rng.choice(probs.shape[0], p=probs))


def apply_chat_template(tokenizer, prompt: str, *, system: str | None = None) -> list[int]:
    """Render a single-turn user ``prompt`` (plus optional ``system``) through the
    tokenizer's chat template and return token ids. Falls back to a plain encode when
    the tokenizer carries no chat template."""
    if not getattr(tokenizer, "chat_template", None):
        return tokenizer.encode(prompt)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
