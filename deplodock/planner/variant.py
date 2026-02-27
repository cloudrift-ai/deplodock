"""Variant: typed benchmark variant with raw matrix params."""

from dataclasses import dataclass

from deplodock.hardware import gpu_short_name


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
            parts.append(f"{abbrev}{non_deploy[key]}")
        return f"{gpu_part}_{'_'.join(parts)}"

    def __eq__(self, other):
        if isinstance(other, Variant):
            return self.params == other.params
        return NotImplemented

    def __hash__(self):
        return hash(tuple(sorted(self.params.items())))
