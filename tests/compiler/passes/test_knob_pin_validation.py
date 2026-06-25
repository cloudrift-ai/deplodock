"""Unit tests for the strict per-op knob-pin validator (``enumeration/_validate``).

Each kernel lowers on ONE tier (MAP / scalar SEMIRING / warp MMA / MONOID), and each
tier owns a disjoint slice of the knob schema. A force-pinned ``DEPLODOCK_<KNOB>`` foreign
to the tier an op resolves to used to be silently dropped (or overwritten by an OFF
sentinel); the validator turns that into a hard :class:`KnobPinError`.

A twisted monoid is the MONOID algebra (transport of structure), and the streaming flash
shares the cooperative reduce's ``monoid_build`` move + knob slice — so it is the SAME
``MONOID`` tier, not a separate one. ``BK``/``FK`` are therefore legal on a flash nest
(split-KV / serial re-bracketing is associativity-licensed), whether or not a given shape
realizes them (the pipeline's job, like ``SPLITK`` on a non-linear matmul).

These tests call :func:`validate_pins` directly (no graph, no CUDA) — the contradiction
is purely a function of ``(algebra, env pins)``. End-to-end refusal through the real
pipeline is covered by ``test_knob_pinning.py`` (CUDA).
"""

from __future__ import annotations

import pytest

from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._validate import KnobPinError, validate_pins

_ATOM = "mma_m16n8k16_f16"


def _pin(monkeypatch, knobs: dict) -> None:
    for k, v in knobs.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", str(v))


# --- Senseless combinations: the validator must REFUSE -----------------------

# Each row: (algebra, pins) where no tier the algebra resolves to can satisfy the pins.
_REFUSED = [
    # Warp MMA forced (MMA=<kind>) + a scalar/cooperative-only knob.
    (AlgebraKind.SEMIRING, {"MMA": _ATOM, "BN": 32}, "scalar THREAD width on a warp kernel"),
    (AlgebraKind.SEMIRING, {"MMA": _ATOM, "BM": 8}, "scalar THREAD width on a warp kernel"),
    (AlgebraKind.SEMIRING, {"MMA": _ATOM, "BR": 4}, "cooperative lanes on a warp kernel"),
    (AlgebraKind.SEMIRING, {"MMA": _ATOM, "FK": 4}, "strip-mine accumulators on a warp kernel"),
    # Scalar forced (MMA=0) + a warp-only knob.
    (AlgebraKind.SEMIRING, {"MMA": 0, "WM": 2}, "warp count on a scalar kernel"),
    (AlgebraKind.SEMIRING, {"MMA": 0, "WN": 2}, "warp count on a scalar kernel"),
    # BR>1 is cooperative/streaming only — foreign to BOTH SEMIRING tiers regardless of MMA.
    (AlgebraKind.SEMIRING, {"BR": 4}, "cooperative lanes on a matmul"),
    # Warp geometry pin (WM) vs a scalar THREAD pin (BN) — the two forced tiers exclude.
    (AlgebraKind.SEMIRING, {"WM": 2, "BN": 32}, "warp geometry vs scalar THREAD tile"),
    # A cooperative MONOID reduce: warp / split-K / tensor-core are all foreign.
    (AlgebraKind.MONOID, {"WM": 2}, "warp count on a cooperative reduce"),
    (AlgebraKind.MONOID, {"MMA": _ATOM}, "tensor-core atom on a cooperative reduce"),
    (AlgebraKind.MONOID, {"SPLITK": 4}, "cross-CTA split-K on a cooperative reduce"),
    # A pointwise MAP nest has no reduce / warp tier.
    (AlgebraKind.MAP, {"BK": 32}, "K-chunk on a pointwise nest"),
    (AlgebraKind.MAP, {"FK": 4}, "strip-mine on a pointwise nest"),
    (AlgebraKind.MAP, {"SPLITK": 4}, "split-K on a pointwise nest"),
    (AlgebraKind.MAP, {"BR": 4}, "cooperative lanes on a pointwise nest"),
    (AlgebraKind.MAP, {"MMA": _ATOM}, "tensor-core atom on a pointwise nest"),
    (AlgebraKind.MAP, {"WM": 2}, "warp count on a pointwise nest"),
    # Staging / transport on a tier that never stages (smem-free reduce / no K-tower).
    (AlgebraKind.MONOID, {"STAGE": 11}, "STAGE on an smem-free cooperative reduce"),
    (AlgebraKind.MONOID, {"TMA": 1}, "TMA on an smem-free cooperative reduce"),
    (AlgebraKind.MAP, {"STAGE": 11}, "STAGE on a pointwise nest with no K-tower"),
    (AlgebraKind.MAP, {"TMA": 1}, "TMA on a pointwise nest with no K-tower"),
]


@pytest.mark.parametrize(("algebra", "pins", "why"), _REFUSED, ids=lambda v: v if isinstance(v, str) else "")
def test_refuses_tier_foreign_pin(algebra, pins, why, monkeypatch):
    _pin(monkeypatch, pins)
    with pytest.raises(KnobPinError):
        validate_pins(algebra)


# --- Legitimate combinations: the validator must NOT refuse ------------------

# Each row's pins are all reachable on (at least one of) the algebra's tiers.
_ALLOWED = [
    # No pins at all.
    (AlgebraKind.SEMIRING, {}),
    # A full scalar matmul pin (MMA unset / scalar tier).
    (AlgebraKind.SEMIRING, {"BN": 32, "BM": 8, "FM": 26, "FN": 4, "BK": 32, "SPLITK": 1, "BR": 1}),
    # A scalar matmul forced via MMA=0 + a real cross-CTA split-K.
    (AlgebraKind.SEMIRING, {"MMA": 0, "BN": 16, "BM": 16, "SPLITK": 4}),
    # A full warp matmul pin (atom + warp counts + register cells + atom-K chunk).
    (AlgebraKind.SEMIRING, {"MMA": _ATOM, "WM": 2, "WN": 2, "FM": 2, "FN": 2, "BK": 2}),
    # Warp-tier split-K (the atomic-free combine, 140_atomic_free_splitk) is supported.
    (AlgebraKind.SEMIRING, {"MMA": _ATOM, "WM": 2, "WN": 2, "FM": 2, "FN": 2, "BK": 2, "SPLITK": 2}),
    # Universal / OFF values are inert next to a warp pin (the value-aware table).
    (AlgebraKind.SEMIRING, {"MMA": _ATOM, "SPLITK": 1, "FK": 1, "BR": 1, "FM": 1, "FN": 1}),
    # A cooperative reduce: free THREAD tile (strided rows) + BR lanes + K-chunk + strip-mine.
    (AlgebraKind.MONOID, {"BN": 128, "BM": 1, "BR": 4, "BK": 32, "FK": 2}),
    # A cooperative reduce keeps the register tile at 1 — pinning FM=FN=1 is inert.
    (AlgebraKind.MONOID, {"MMA": 0, "FM": 1, "FN": 1, "SPLITK": 1}),
    # A pointwise nest: only the free-axis thread + register tile.
    (AlgebraKind.MAP, {"BN": 64, "BM": 4, "FN": 4, "FM": 2}),
    # STAGE / TMA on the staged tiers (scalar reduce / warp matmul) is fine. An all-zero
    # STAGE mask and TMA=0 are inert everywhere (universal / OFF values).
    (AlgebraKind.SEMIRING, {"STAGE": 11, "TMA": 1}),
    (AlgebraKind.SEMIRING, {"MMA": _ATOM, "STAGE": 11, "TMA": 1}),
    (AlgebraKind.MONOID, {"STAGE": "00", "TMA": 0}),
    (AlgebraKind.MAP, {"STAGE": "0", "TMA": 0}),
]


@pytest.mark.parametrize(("algebra", "pins"), _ALLOWED, ids=lambda v: v if isinstance(v, str) else "")
def test_allows_tier_native_pins(algebra, pins, monkeypatch):
    _pin(monkeypatch, pins)
    validate_pins(algebra)  # must not raise


# --- The streaming flash IS the MONOID tier (no separate streaming tier) -------
# Flash shares the cooperative reduce's ``monoid_build`` move + knob slice, so it lowers
# on the same ``MONOID`` tier. Warp / split-K / tensor-core / staging knobs are foreign to
# it (identical to the cooperative-reduce ``_REFUSED`` rows above), but the K-chunk knobs
# ``BK``/``FK`` are now LEGAL on it (split-KV / serial re-bracketing is associativity-
# licensed on the nested monoid too) — the tier-split collapse of Phase 0.
def test_streaming_flash_uses_the_one_monoid_tier(monkeypatch):
    # BK/FK no longer hard-error on a flash (MONOID) nest — they are legal on the tier
    # (a particular shape may downgrade them, the pipeline's job, not a pin contradiction).
    for pins in ({"BK": 32}, {"FK": 4}, {"BN": 32, "BM": 1, "BR": 4}):
        _pin(monkeypatch, pins)
        validate_pins(AlgebraKind.MONOID)  # must not raise


def test_greedy_pipeline_refuses_end_to_end(monkeypatch):
    """The refusal propagates through a real greedy compile (``Pipeline.run`` →
    ``010_build`` → ``validate_pins``): a warp atom pinned beside a scalar THREAD
    width fails loudly instead of silently dropping ``BN``. No CUDA — the validator
    fires at the enumeration seed, before any kernel is emitted."""
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp
    from deplodock.compiler.pipeline import TILE_PASSES, Pipeline

    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (128, 128)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (128, 128)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (128, 128)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    monkeypatch.setenv("DEPLODOCK_MMA", _ATOM)
    monkeypatch.setenv("DEPLODOCK_BN", "32")
    with pytest.raises(KnobPinError):
        Pipeline.build(TILE_PASSES).run(g)


def test_search_path_exempt_from_validation(monkeypatch):
    """The tune SEARCH (``Run.drive``) is exempt: a union pin vector that no single
    op can satisfy (warp ``WM``/``WN`` + scalar ``BN``) still drives to terminals,
    because each op takes its tier's subset and a per-op contradiction is a pruned
    branch, not a hard error (``ctx.validate_pins=False`` under drive)."""
    from deplodock.compiler.context import Context
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp
    from deplodock.compiler.pipeline import TILE_PASSES, Pipeline
    from deplodock.compiler.pipeline.pipeline import Run
    from deplodock.compiler.pipeline.search.policy.mcts import TuningSearch

    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (128, 128)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (128, 128)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (128, 128)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    # A union pin: WM/WN (warp) + BN (scalar) — contradictory for a single kernel,
    # so the greedy path would refuse, but the search must explore it.
    for k, v in {"WM": 2, "WN": 2, "BN": 32, "FM": 2, "FN": 2, "BK": 2}.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", str(v))
    run = Run(pipeline=Pipeline.build(TILE_PASSES), ctx=Context.from_target((12, 0)), search=TuningSearch(patience=2))
    terminals = list(run.drive(g))
    assert terminals, "search must reach at least one terminal under a union pin (no KnobPinError)"


def test_error_names_the_offending_pins(monkeypatch):
    """The message names the algebra, the tiers it can lower on, and each pin's legal
    tiers — enough to see which knob to drop."""
    _pin(monkeypatch, {"MMA": _ATOM, "BR": 4})
    with pytest.raises(KnobPinError) as exc:
        validate_pins(AlgebraKind.SEMIRING)
    msg = str(exc.value)
    assert "semiring" in msg
    assert "MMA=" in msg and "BR=4" in msg
