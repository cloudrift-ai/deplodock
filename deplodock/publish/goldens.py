"""Golden configs: YAML-on-disk + in-memory dataclasses + DB materialization.

A *golden config* records, for one canonical shape on one GPU, the autotuned
knob set and the latencies of the deplodock kernel vs a reference (e.g.
cuBLAS). A config is ``golden`` when deplodock lands within 95% of the
reference, i.e. ``ratio = ref_us / deplodock_us >= 0.95``.

Goldens are the curated source of truth in the repo (``goldens/*.yaml``,
reviewable as text). The tuning DB is a derived store: this module materializes
goldens into ``SearchDB.perf`` with ``source='golden'`` so a single SQL query
returns "the best known config" regardless of origin (tuned or golden).

Regenerate the YAML via ``scripts/find_golden_configs.py``. Schema mirrors the
:class:`MatmulGoldenConfig` fields — same keys, just serialized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Qwen3-Embedding-0.6B linear dims (mirrors ``tests/perf/cases.py``). Kept here
# because ``scripts/find_golden_configs.py`` builds shape names from them.
QWEN3_06B_HIDDEN = 1024  # hidden_size
QWEN3_06B_INTER = 3072  # intermediate_size (gate / up / down)
QWEN3_06B_Q_DIM = 2048  # fused Q-projection output (16 heads * 128)
QWEN3_06B_KV_DIM = 1024  # fused K/V-projection output (8 heads * 128)


def matmul_snippet(M: int, N: int, K: int, dtype: str = "fp32") -> str:
    """The torch expression a matmul golden config tunes / benches / reproduces."""
    if dtype == "fp32":
        return f"torch.matmul(torch.randn({M},{K}), torch.randn({K},{N}))"
    tdt = {"fp16": "torch.float16", "bf16": "torch.bfloat16"}[dtype]
    return f"torch.matmul(torch.randn({M},{K},dtype={tdt}), torch.randn({K},{N},dtype={tdt}))"


def _knobs_env(knobs: dict) -> str:
    """Render a knobs dict as a ``DEPLODOCK_KNOBS`` value: ``BM=8,BN=32,...``."""
    return ",".join(f"{k}={v}" for k, v in knobs.items())


@dataclass(frozen=True, kw_only=True)
class GoldenConfig:
    """A kernel config measured within (or near) a reference on a specific GPU.

    Only the two raw latencies are stored; :attr:`ratio` and :attr:`golden`
    derive from them so the record cannot drift out of sync.
    """

    name: str
    gpu_name: str = "NVIDIA GeForce RTX 5090"
    compute_cap: tuple[int, int] = (12, 0)
    knobs: dict = field(default_factory=dict)
    deplodock_us: float = 0.0
    cublas_us: float = 0.0

    @property
    def ratio(self) -> float:
        """Reference latency / deplodock latency — 1.0 means parity, >1 means faster."""
        return self.cublas_us / self.deplodock_us if self.deplodock_us else 0.0

    @property
    def golden(self) -> bool:
        """Within 95% of cuBLAS or better."""
        return self.ratio >= 0.95


@dataclass(frozen=True, kw_only=True)
class MatmulGoldenConfig(GoldenConfig):
    """A golden config for a plain 2-D matmul ``(M,K) @ (K,N)``."""

    M: int
    N: int
    K: int
    dtype: str = "fp32"

    def snippet(self) -> str:
        return matmul_snippet(self.M, self.N, self.K, self.dtype)

    def repro_command(self, ir: str = "cuda") -> str:
        return f'DEPLODOCK_KNOBS="{_knobs_env(self.knobs)}" deplodock compile -c "{self.snippet()}" --ir {ir}'


# --- YAML I/O --------------------------------------------------------------

# Default location: ``<repo>/goldens/``. Overridable via the ``GOLDENS_DIR`` env
# var (testing). Resolved lazily so the path lookup doesn't fire at import.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def goldens_dir() -> Path:
    import os

    return Path(os.environ.get("DEPLODOCK_GOLDENS_DIR") or _REPO_ROOT / "goldens")


def _entry_to_config(entry: dict) -> GoldenConfig:
    kind = entry.get("kind", "matmul")
    if kind != "matmul":
        raise ValueError(f"unknown golden kind: {kind!r} (entry name={entry.get('name')!r})")
    return MatmulGoldenConfig(
        name=entry["name"],
        gpu_name=entry["gpu_name"],
        compute_cap=tuple(entry["compute_cap"]),
        knobs=dict(entry.get("knobs", {})),
        deplodock_us=float(entry.get("deplodock_us", 0.0)),
        cublas_us=float(entry.get("cublas_us", 0.0)),
        M=int(entry["M"]),
        N=int(entry["N"]),
        K=int(entry["K"]),
        dtype=entry.get("dtype", "fp32"),
    )


def load_goldens(directory: Path | None = None) -> list[GoldenConfig]:
    """Load every ``goldens/**/*.yaml`` file as a flat list of :class:`GoldenConfig`.

    Idempotent: pure function of the YAML on disk. Empty list when the
    directory does not exist (the publish package may be imported from an
    installed wheel where ``goldens/`` is not shipped)."""
    directory = directory or goldens_dir()
    if not directory.exists():
        return []
    configs: list[GoldenConfig] = []
    for path in sorted(directory.glob("**/*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f) or []
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a list of entries at top level, got {type(data).__name__}")
        for entry in data:
            configs.append(_entry_to_config(entry))
    return configs


def dump_goldens(configs: list[GoldenConfig], directory: Path | None = None) -> None:
    """Write ``configs`` partitioned by ``kind`` into ``goldens/<kind>.yaml``.

    Used by ``scripts/find_golden_configs.py`` after a sweep. Stable key order
    so re-dumping the same configs produces an identical file."""
    directory = directory or goldens_dir()
    directory.mkdir(parents=True, exist_ok=True)
    by_kind: dict[str, list[dict]] = {}
    for c in configs:
        if not isinstance(c, MatmulGoldenConfig):
            raise TypeError(f"unknown golden type: {type(c).__name__}")
        by_kind.setdefault("matmul", []).append(
            {
                "name": c.name,
                "kind": "matmul",
                "M": c.M,
                "N": c.N,
                "K": c.K,
                "dtype": c.dtype,
                "gpu_name": c.gpu_name,
                "compute_cap": list(c.compute_cap),
                "knobs": dict(c.knobs),
                "deplodock_us": c.deplodock_us,
                "cublas_us": c.cublas_us,
            }
        )
    header = (
        "# Golden configs — autotuned knob set within ~95% of a reference (cuBLAS for matmul).\n"
        "# Regenerated by scripts/find_golden_configs.py. Format: list of entries.\n\n"
    )
    for kind, entries in by_kind.items():
        out = directory / f"{kind}.yaml"
        with open(out, "w") as f:
            f.write(header)
            yaml.safe_dump(entries, f, sort_keys=False, default_flow_style=None, width=120)
