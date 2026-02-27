"""Variant: typed benchmark variant with raw matrix params."""

import re
from dataclasses import dataclass

from deplodock.hardware import gpu_short_name

_KNOWN_ABBREVIATIONS: dict[str, str] = {
    "cache": "cache",
    "dtype": "dtype",
    "expert": "exp",
    "kv": "kv",
    "latest": "latest",
    "lmsysorg": "lms",
    "moe": "moe",
    "openai": "oai",
    "parallel": "par",
    "quantization": "quant",
    "reasoning": "reas",
    "sglang": "sglang",
    "vllm": "vllm",
}


def _compact_segment(segment: str) -> str:
    """Abbreviate a single segment of a compound value.

    Lookup in known abbreviations first, then keep segments with digits intact,
    otherwise take the first character.
    """
    low = segment.lower()
    if low in _KNOWN_ABBREVIATIONS:
        return _KNOWN_ABBREVIATIONS[low]
    if any(c.isdigit() for c in segment):
        return segment
    return segment[0]


def _compact_value(value: object) -> str:
    """Sanitize and abbreviate a parameter value for use in filesystem paths.

    1. Replace filesystem-unsafe characters with dashes and collapse runs.
    2. If the result is compound (has dashes/underscores), abbreviate each
       segment via _compact_segment and join with dashes.
    """
    s = str(value)
    s = re.sub(r"[^\w.-]", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    # Only abbreviate compound values (plain numbers like "128" pass through).
    if not re.search(r"[-_]", s):
        return s
    parts = [p for p in re.split(r"[-_]+", s) if p]
    return "-".join(_compact_segment(p) for p in parts)


def _abbreviate(snake_case_name: str) -> str:
    """Abbreviate a snake_case name by taking the first letter of each word.

    Examples: max_concurrent_requests → mcr, num_prompts → np, max_concurrency → mc
    """
    return "".join(word[0] for word in snake_case_name.split("_"))


@dataclass(frozen=True)
class Variant:
    """A benchmark variant: one specific matrix combination.

    Preserves the raw matrix params dict and derives the GPU short name,
    GPU count, and string label via properties.
    """

    params: dict

    @property
    def gpu_short(self) -> str:
        """Short GPU name (e.g. 'rtx5090')."""
        return gpu_short_name(self.params["deploy.gpu"])

    @property
    def gpu_count(self) -> int:
        """Number of GPUs for this variant."""
        return self.params.get("deploy.gpu_count", 1)

    def __str__(self) -> str:
        gpu_part = f"{self.gpu_short}x{self.gpu_count}"
        non_deploy = {k: v for k, v in self.params.items() if not k.startswith("deploy.")}
        if not non_deploy:
            return gpu_part
        parts = []
        for key in sorted(non_deploy):
            last_segment = key.rsplit(".", 1)[-1]
            abbrev = _abbreviate(last_segment)
            parts.append(f"{abbrev}{_compact_value(non_deploy[key])}")
        return f"{gpu_part}_{'_'.join(parts)}"

    def __eq__(self, other):
        if isinstance(other, Variant):
            return self.params == other.params
        return NotImplemented

    def __hash__(self):
        return hash(tuple(sorted(self.params.items())))
