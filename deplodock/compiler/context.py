"""Per-compilation context passed through the rewrite pipeline.

A ``Context`` carries target-derived state (``compute_capability``) and
any future per-compilation knobs (tuning params, flags) that rules need
to read. Built once per ``run_pipeline`` invocation and threaded through
the engine; rules that take a ``ctx`` parameter receive it via the same
dispatcher binding that supplies ``graph`` / ``match`` / ``root``.

This replaces process-global access patterns like the ``_OVERRIDE``
module state in :mod:`deplodock.compiler.target`: explicit-by-design
keeps tests simple (no monkey-patching), keeps compilations independent
(no cross-thread state), and makes rule dependencies discoverable from
the signature alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock import config

# Per-block static-smem cap. Hard hardware limit since Maxwell (sm_50);
# anything declared as ``__shared__`` at compile time must fit.
# Universal across every arch we target.
STATIC_SMEM_CAP = 48 * 1024

# Per-block dynamic-smem opt-in cap by compute capability. NVIDIA assigns
# different sm_XX numbers to datacenter vs consumer SKUs within the same
# arch family (sm_80 A100 vs sm_86 RTX 30xx; sm_90 H100 vs sm_120 RTX
# 50xx), so this is keyed on cc, not "arch generation". Values from
# ``cudaDevAttrMaxSharedMemoryPerBlockOptin``.
_MAX_DYNAMIC_SMEM_BY_CC: dict[tuple[int, int], int] = {
    (7, 5): 64 * 1024,
    (8, 0): 163 * 1024,
    (8, 6): 99 * 1024,
    (8, 9): 99 * 1024,
    (9, 0): 227 * 1024,
    (10, 0): 227 * 1024,
    (12, 0): 99 * 1024,
}

# Tensor-core generation by compute capability — Volta(1)/Turing(2)/Ampere+Ada(3)
# /Hopper(4)/Blackwell(5). A coarse arch-capability axis for the learned prior's
# regime features (e.g. which mma shapes / dtypes the SM can issue); unknown ccs
# fall back to the major version.
_TENSOR_CORE_GEN: dict[tuple[int, int], int] = {
    (7, 0): 1,
    (7, 5): 2,
    (8, 0): 3,
    (8, 6): 3,
    (8, 9): 3,
    (9, 0): 4,
    (10, 0): 5,
    (12, 0): 5,
}


def _env_compile_flags() -> str:
    """Extra nvcc flags for this compile (``DEPLODOCK_NVCC_FLAGS``). Set by the
    CLI commands (via :func:`deplodock.config.set_nvcc_flags`); folded into
    :meth:`Context.structural_key` so the perf cache is partitioned by opt level."""
    return config.nvcc_flags()


def _max_dynamic_smem_for(cc: tuple[int, int]) -> int:
    """Per-block dynamic-smem opt-in cap for ``cc``. Falls back to
    ``STATIC_SMEM_CAP`` for unknown / no-hardware caps so callers that
    use it as a budget naturally degrade to the static-only ceiling."""
    return _MAX_DYNAMIC_SMEM_BY_CC.get(cc, STATIC_SMEM_CAP)


@dataclass(frozen=True)
class Context:
    """Per-compilation state visible to rules.

    Build with :meth:`from_target` (CLI / autotuner) or :meth:`probe`
    (live device). Frozen so a ``Context`` can't be mutated mid-pipeline
    — replace via :func:`dataclasses.replace` if a phase needs to alter
    a field.
    """

    compute_capability: tuple[int, int]
    static_smem_cap: int = STATIC_SMEM_CAP
    max_dynamic_smem: int = STATIC_SMEM_CAP  # overridden by from_target/probe
    # Hardware-universal per-CTA thread cap (CUDA compute capability ≥ 2.0).
    # Used by ``KernelOp.validate`` to filter autotune variants whose launch
    # geometry would be rejected by the driver before the kernel ever runs.
    max_threads_per_cta: int = 1024
    # Hardware warp width — 32 on every NVIDIA arch we target. Carried on
    # ``Context`` so cooperative-reduce gating (``010_partition_loops``)
    # and warp-shuffle dispatch (``100_materialize_tile``) read a single
    # source of truth instead of redefining the constant module-locally.
    warp_size: int = 32
    # Shared-memory vector-load width in bytes — ``LDS.128`` carries 128
    # bits (16 bytes) per lane on every NVIDIA arch since Volta, so this
    # is genuinely hardware-universal. Element counts are derived per
    # dtype at the use site (``lds128_bytes // BYTES_PER_ELEM``): 4 fp32
    # / 8 fp16 / 16 fp8 fuse into one transaction. ``007a`` uses it to
    # pick the per-thread N-chunk width that keeps each LDS.128 phase
    # bank-conflict-free.
    lds128_bytes: int = 16
    # Identifies which backend's perf rows this compile reads/writes — the
    # tune DB keys ``perf`` by ``(context_key, op_key, backend)``. Defaults to
    # ``"cuda"`` — the canonical autotune target. ``run_autotune`` replaces
    # this when a live :class:`Backend` is supplied.
    backend_name: str = "cuda"
    # Extra nvcc flags this compile uses (from ``DEPLODOCK_NVCC_FLAGS`` — e.g.
    # tune's ``-Xcicc -O1`` vs compile/run's -O3). Folded into
    # ``structural_key`` so the autotune ``perf`` cache is partitioned by opt
    # level: -O1-measured latencies (a fast-compile *ranking* signal) never
    # clobber -O3 ones, and a later -O3 ``run`` re-benches rather than reading a
    # stale -O1 number. Populated from the env by :meth:`probe` /
    # :meth:`from_target`.
    compile_flags: str = ""

    @classmethod
    def from_target(cls, cap: tuple[int, int]) -> Context:
        return cls(compute_capability=cap, max_dynamic_smem=_max_dynamic_smem_for(cap), compile_flags=_env_compile_flags())

    def structural_key(self) -> str:
        """Implements :class:`deplodock.compiler.structural.Structural`.

        Folds in only codegen-affecting fields. ``compute_capability``
        gates hardware-feature passes (TMA, cp.async, dynamic smem cap);
        anything derived from it (``max_dynamic_smem``) is implied.
        ``compile_flags`` is folded in because the nvcc opt level genuinely
        changes the emitted SASS / measured latency (it is NOT a debug flag).
        As other non-derived knobs land (forced TMA on/off, splitk overrides),
        extend this method explicitly — keep ambient I/O fields out so the
        autotuning cache survives debug-flag flips.
        """
        from deplodock.compiler.structural import digest  # noqa: PLC0415

        return digest("Context", self.compute_capability, self.compile_flags)

    def features(self) -> dict[str, float]:
        """Host/hardware regime as ``H_*`` features for the learned prior, so a
        SINGLE global prior spans every GPU and nvcc opt level (these are
        constant across a compile's sibling candidates → they never change the
        argmax; they only let the model fit per-regime offsets instead of
        averaging across regimes). Combines capability-derived facts with the
        live device's physical SKU properties:

        - ``H_cc`` — compute capability ``major*10 + minor``
        - ``H_tc_gen`` — tensor-core generation (``_TENSOR_CORE_GEN``)
        - ``H_smem_optin`` — per-block dynamic-smem opt-in cap (bytes)
        - ``H_opt`` — nvcc cicc opt level from ``compile_flags`` (tune's
          ``-Xcicc -O1`` → 1; compile/run's default → 3)
        - ``H_sm_count`` / ``H_smem_per_sm`` / ``H_smem_per_block`` /
          ``H_regs_per_block`` / ``H_warp_size`` — live device props
          (:func:`target.live_device_features`; absent on GPU-less hosts)
        """
        import re  # noqa: PLC0415

        from deplodock.compiler.target import live_device_features  # noqa: PLC0415

        major, minor = self.compute_capability
        m = re.search(r"-O(\d)", self.compile_flags)
        feats = {
            "H_cc": float(major * 10 + minor),
            "H_tc_gen": float(_TENSOR_CORE_GEN.get((major, minor), major)),
            "H_smem_optin": float(self.max_dynamic_smem),
            "H_opt": float(m.group(1)) if m else 3.0,
        }
        for k, v in live_device_features().items():
            feats[f"H_{k}"] = v
        return feats

    @classmethod
    def probe(cls) -> Context:
        """Build by probing the live CUDA device. Falls back to (0, 0) if
        cupy is unavailable — callers treat that as "no hardware feature
        support" (rules gating on capability self-skip via ``RuleSkipped``).

        ``max_dynamic_smem`` is the *live device's* opt-in cap, not the
        target's: passes that gate on compute capability honor the
        ``set_target`` override (so they take the target's codegen path),
        but the dynamic-smem budget must still fit the actual hardware
        the kernel will run on — otherwise ``cudaFuncSetAttribute`` rejects
        the launch. Without this distinction, ``--target sm_90`` on an
        sm_86 box would request 227 KB on a 99 KB device.
        """
        from deplodock.compiler.target import compute_capability, live_compute_capability  # noqa: PLC0415

        cap = compute_capability()
        live = live_compute_capability()
        # No live CUDA device → compile-only flow, trust the target's cap.
        # Live device present → clamp so the actual launch fits.
        if live == (0, 0):
            smem = _max_dynamic_smem_for(cap)
        else:
            smem = min(_max_dynamic_smem_for(cap), _max_dynamic_smem_for(live))
        return cls(compute_capability=cap, max_dynamic_smem=smem, compile_flags=_env_compile_flags())
