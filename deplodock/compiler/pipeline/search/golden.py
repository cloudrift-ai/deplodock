"""Golden matmul configs — known-good ``knobs`` per shape, measured vs cuBLAS.

A *golden config* records, for one matmul shape on one GPU, the autotuned knob
set and the latencies of the deplodock kernel vs cuBLAS (``torch.matmul``). A
config is ``golden`` when deplodock lands within 95% of cuBLAS (or better),
i.e. ``ratio = cublas_us / deplodock_us >= 0.95``.

The set is a ground truth for the tuning **prior**: it pins down what the
planner's first guess *should* land on for canonical shapes, and gives a
deployable-latency baseline to regression-test against.

A ``name`` is **not** unique — one shape may carry several golden configs (e.g. a
newly found faster knob set kept beside the old). Look configs up with
:func:`goldens_by_name` (returns a list) and never assume a single match.

This module is import-light (no torch / cupy at module top) so passes and tests
can read :data:`GOLDEN_CONFIGS` cheaply. The data lives as **one YAML file per GPU**
under ``goldens/`` (e.g. ``goldens/rtx5090_sm120.yaml``): a ``gpu_name`` /
``compute_cap`` header plus a ``configs`` list, each tagged with a ``kernel``
discriminator (``matmul`` / ``reduce`` / ``pointwise``). :func:`_load_goldens`
concatenates every file into :data:`GOLDEN_CONFIGS`. The set is hand-maintained via
the CLI golden workflow — ``deplodock tune --golden NAME --bench`` records the
winning knobs / latencies into the GPU's YAML, ``deplodock eval golden`` validates.
For the **fp32** configs the reference is
pinned to **true fp32** (``allow_tf32 = False``) so the ratio compares deplodock's
CUDA-core FMA kernel against a real SGEMM, not the ~5-10x faster TF32 tensor-core
path. The **fp16** squares (``*.fp16``) instead ride the warp-tier tensor-core path
and compare against cuBLAS HGEMM (torch's default fp16 matmul) — same tensor-core
hardware on both sides, so the ratio is apples-to-apples vs cuBLAS. On sm_90+ the
autotuner lands these on the swizzled s16816 ``mma_m16n8k16_f16`` (ldmatrix +
mma.sync) atom — the swizzled smem slab avoids shared-load bank conflicts (a
fragment load reading smem opaquely cannot), so mma.sync is the faster fp16
GEMM. On sm_120 the warp-tier prior + greedy now land on a square
**64×64 output tile on a 4-warp CTA with WARP_SPECIALIZE=1** (producer warp issues
TMA, consumer warps run the mma chain), measured at / above cuBLAS across the
squares (2048²: 106.7 µs / 1.06×; 4096²: 746 µs / 1.03×; 1024²: 0.94×). See the
warp-tier ranking in ``search/prior/AnalyticPrior`` (the ``D_*`` geometry features
over ``knob.knob_features``) and the WS=1-first emission order in
``085_warp_specialize``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from deplodock.compiler.pipeline.knob import STRUCT_PREFIX

# Qwen3-Embedding-0.6B linear dims (mirrors ``tests/perf/cases.py``).
QWEN3_06B_HIDDEN = 1024  # hidden_size
QWEN3_06B_INTER = 3072  # intermediate_size (gate / up / down)
QWEN3_06B_Q_DIM = 2048  # fused Q-projection output (16 heads * 128)
QWEN3_06B_KV_DIM = 1024  # fused K/V-projection output (8 heads * 128)


def matmul_snippet(M: int, N: int, K: int, dtype: str = "fp32") -> str:
    """The torch expression a matmul golden config tunes / benches / reproduces.

    Single source of truth: the autotune / repro paths feed this to
    ``trace_inline_code`` (so the tuned graph *is* this expression), and each
    config reproduces from the same call. fp32 is ``torch.randn``'s default, so
    no dtype kwarg is emitted for fp32 — matching the canonical example
    ``torch.matmul(torch.randn(2048,2048), torch.randn(2048,2048))``.
    """
    if dtype == "fp32":
        return f"torch.matmul(torch.randn({M},{K}), torch.randn({K},{N}))"
    tdt = {"fp16": "torch.float16", "bf16": "torch.bfloat16"}[dtype]
    return f"torch.matmul(torch.randn({M},{K},dtype={tdt}), torch.randn({K},{N},dtype={tdt}))"


def _knobs_env(knobs: dict) -> str:
    """Render a knobs dict as a ``DEPLODOCK_KNOBS`` value: ``BM=8,BN=32,...``.

    Structural-feature knobs (``STRUCT_PREFIX``) are dropped — a repro command
    pins tuning decisions, not the kernel's structural identity."""
    return ",".join(f"{k}={v}" for k, v in knobs.items() if not k.startswith(STRUCT_PREFIX))


@dataclass(frozen=True, kw_only=True)
class GoldenConfig:
    """A kernel config measured within (or near) cuBLAS on a specific GPU.

    Only the two raw latencies are stored; :attr:`ratio` and :attr:`golden`
    derive from them so the record cannot drift out of sync.
    """

    name: str  # e.g. "square.2048" / "qwen3_06b.q_proj.s128"
    gpu_name: str = "NVIDIA GeForce RTX 5090"
    compute_cap: tuple[int, int] = (12, 0)
    knobs: dict = field(default_factory=dict)  # dict(cuda_op.knobs), verbatim
    deplodock_us: float = 0.0
    cublas_us: float = 0.0

    @property
    def ratio(self) -> float:
        """cuBLAS latency / deplodock latency — 1.0 means parity, >1 means faster."""
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
        """The torch expression this config tunes / benches / reproduces."""
        return matmul_snippet(self.M, self.N, self.K, self.dtype)

    def repro_command(self, ir: str = "cuda") -> str:
        """A runnable ``deplodock`` command that rebuilds this config's kernel.

        e.g. ``DEPLODOCK_KNOBS="BM=8,..." deplodock compile -c "torch.matmul(...)" --ir cuda``
        """
        return f'DEPLODOCK_KNOBS="{_knobs_env(self.knobs)}" deplodock compile -c "{self.snippet()}" --ir {ir}'


@dataclass(frozen=True, kw_only=True)
class ReduceGoldenConfig(GoldenConfig):
    """A golden config for a row-reduce ``(M, K) → (M,)`` (``torch.sum(dim=-1)``).

    The good config is cooperative: ``BR > 1`` threads reduce each row in parallel
    (then a WarpShuffle / TreeHalve combine), so the prior must rank cooperative
    ``BR`` above the serial ``BR=1`` tile — the signal the matmul-only fit lacked.
    Enumerated by ``priority_mode="reduce"`` (``E_N=1``, free=M, reduce=K)."""

    M: int
    K: int

    def snippet(self) -> str:
        return f"torch.sum(torch.randn({self.M},{self.K}),dim=-1)"


@dataclass(frozen=True, kw_only=True)
class PointwiseGoldenConfig(GoldenConfig):
    """A golden config for an elementwise map ``(M, N) → (M, N)`` (``torch.relu``).

    Memory-bound: the good config is a wide coalesced tile (large ``BN`` / ``FM``,
    no reduce). Enumerated by ``priority_mode="pointwise"`` (``E_K=1``, free=M·N)."""

    M: int
    N: int

    def snippet(self) -> str:
        return f"torch.relu(torch.randn({self.M},{self.N}))"


_GOLDENS_DIR = Path(__file__).parent / "goldens"
_KERNEL_CLASSES = {"matmul": MatmulGoldenConfig, "reduce": ReduceGoldenConfig, "pointwise": PointwiseGoldenConfig}


def _load_goldens() -> list[GoldenConfig]:
    """Load every per-GPU golden YAML under :data:`_GOLDENS_DIR` into one flat list.

    One file per GPU: a ``gpu_name`` / ``compute_cap`` header (stamped onto every
    config so it isn't repeated per entry) plus a ``configs`` list, each tagged with
    a ``kernel`` discriminator (``matmul`` / ``reduce`` / ``pointwise``) selecting the
    dataclass. All files are concatenated — a ``name`` may recur across GPUs, told
    apart by ``compute_cap`` (see :func:`goldens_by_name`)."""
    out: list[GoldenConfig] = []
    for path in sorted(_GOLDENS_DIR.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        gpu_name, cap = doc["gpu_name"], tuple(doc["compute_cap"])
        for c in doc["configs"]:
            out.append(_KERNEL_CLASSES[c.pop("kernel")](gpu_name=gpu_name, compute_cap=cap, **c))
    return out


GOLDEN_CONFIGS: list[GoldenConfig] = _load_goldens()


def goldens_by_name(name: str) -> list[MatmulGoldenConfig]:
    """Every :class:`MatmulGoldenConfig` recorded under ``name`` — a **list**, not
    a single config: one shape may carry several golden knob sets (e.g. a newly
    found faster variant alongside the old one), so callers must not assume a name
    is unique. Empty when ``name`` is unknown. All entries share the shape (so any
    one's :meth:`~MatmulGoldenConfig.snippet` is interchangeable); they differ only
    in ``knobs`` / measured latency."""
    return [g for g in GOLDEN_CONFIGS if isinstance(g, MatmulGoldenConfig) and g.name == name]
