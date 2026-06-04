"""Tests for the MMA fragment-factorization planner wiring.

Covers ``_atom.is_atom_eligible`` per Design decision 4 of
``plans/mma-fragment-factorization.md``. M2 ships the WMMA F16 entry +
the eligibility dispatcher; M3 will extend coverage to fork enumeration.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.context import Context
from deplodock.compiler.dtype import F16, F32
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Assign
from deplodock.compiler.pipeline.passes.lowering.tile._atom import (
    ATOM_REGISTRY,
    Atom,
    is_atom_eligible,
)
from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import (
    WarpTileParams,
    _enumerate_warp_matmul_impl,
)


def _input(g: Graph, name: str, shape: tuple, *, dtype) -> str:
    """Add a typed input tensor — dtype lookup at planner time goes
    through ``graph.nodes[buf].output.dtype``, so the test fixture has
    to stamp the dtype on the input Tensor."""
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape, dtype=dtype), node_id=name)


def _ctx(*, cc: tuple[int, int] = (8, 0)) -> Context:
    """Test ``Context`` with arbitrary compute capability. ``warp_size`` and
    ``max_dynamic_smem`` defaults are fine — the WMMA eligibility only
    reads ``compute_capability``."""
    return Context(compute_capability=cc)


def _matmul_loop_op(*, M: int = 64, N: int = 64, K: int = 64) -> LoopOp:
    """Build a plain matmul LoopOp ``C[M,N] = sum_k A[M,K] * B[K,N]``."""
    i = Axis("i", M)
    j = Axis("j", N)
    k = Axis("k", K)
    return LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Loop(
                                axis=k,
                                body=(
                                    Load(name="a", input="a", index=(Var("i"), Var("k"))),
                                    Load(name="b", input="b", index=(Var("k"), Var("j"))),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a", "b")),
                                    Accum(name="acc", value="p"),
                                ),
                            ),
                            Write(output="c", index=(Var("i"), Var("j")), value="acc"),
                        ),
                    ),
                ),
            ),
        ),
    )


def _pointwise_loop_op() -> LoopOp:
    """Build a pointwise (relu) LoopOp — no reduce, ineligible for any
    matmul atom kind."""
    i = Axis("i", 64)
    return LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Load(name="x_v", input="x", index=(Var("i"),)),
                    Assign(name="r", op=ElementwiseImpl("maximum"), args=("x_v", "x_v")),
                    Write(output="o", index=(Var("i"),), value="r"),
                ),
            ),
        ),
    )


def _matmul_graph(*, M: int = 64, N: int = 64, K: int = 64, dtype=F16) -> Graph:
    """Build a graph wrapping the matmul LoopOp with typed inputs."""
    g = Graph()
    _input(g, "a", (M, K), dtype=dtype)
    _input(g, "b", (K, N), dtype=dtype)
    op = _matmul_loop_op(M=M, N=N, K=K)
    g.add_node(op=op, inputs=["a", "b"], output=Tensor("c", (M, N), dtype=F32), node_id="c")
    g.inputs = ["a", "b"]
    g.outputs = ["c"]
    return g


def test_mma_eligibility_fires_on_f16_matmul():
    """A TinyLlama-shape matmul (M=N=K=64; M%16==0, N%8==0, K%16==0) with
    F16 inputs and a sm_80 target satisfies every mma.sync F16 gate."""
    g = _matmul_graph(M=64, N=64, K=64, dtype=F16)
    loop_op = g.nodes["c"].op
    assert is_atom_eligible(ATOM_REGISTRY["mma_m16n8k16_f16"], loop_op, _ctx(cc=(8, 0)), graph=g)


def test_mma_eligibility_rejects_f32_loads():
    """F32 operands fall through the per-Load dtype check."""
    g = _matmul_graph(M=64, N=64, K=64, dtype=F32)
    loop_op = g.nodes["c"].op
    assert not is_atom_eligible(ATOM_REGISTRY["mma_m16n8k16_f16"], loop_op, _ctx(cc=(8, 0)), graph=g)


def test_mma_eligibility_rejects_pre_ampere():
    """mma.sync.m16n8k16 needs sm_80+ (Ampere). Volta (sm_70) fails the cc
    gate (min_cc=(8, 0))."""
    g = _matmul_graph(M=64, N=64, K=64, dtype=F16)
    loop_op = g.nodes["c"].op
    assert not is_atom_eligible(ATOM_REGISTRY["mma_m16n8k16_f16"], loop_op, _ctx(cc=(7, 0)), graph=g)


def test_mma_eligibility_rejects_non_matmul():
    """A pointwise LoopOp has no matmul-reduce — ineligible for any MMA
    kind."""
    g = Graph()
    _input(g, "x", (64,), dtype=F16)
    op = _pointwise_loop_op()
    g.add_node(op=op, inputs=["x"], output=Tensor("o", (64,), dtype=F16), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]
    assert not is_atom_eligible(ATOM_REGISTRY["mma_m16n8k16_f16"], op, _ctx(cc=(8, 0)), graph=g)


def test_mma_eligibility_rejects_indivisible_extents():
    """An output extent that isn't a multiple of the atom dim (e.g. M=63,
    not a multiple of 16) can't cover with m16n8k16 cells."""
    g = _matmul_graph(M=63, N=64, K=64, dtype=F16)
    loop_op = g.nodes["c"].op
    assert not is_atom_eligible(ATOM_REGISTRY["mma_m16n8k16_f16"], loop_op, _ctx(cc=(8, 0)), graph=g)


def _matmul_with_epilogue_loop_op(*, M: int = 64, N: int = 64, K: int = 64) -> LoopOp:
    """Matmul whose accumulator feeds a post-reduce ``add(acc, r)`` (fused
    residual) before the Write — the mma fragment-store can't apply it."""
    i, j, k = Axis("i", M), Axis("j", N), Axis("k", K)
    return LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Loop(
                                axis=k,
                                body=(
                                    Load(name="a", input="a", index=(Var("i"), Var("k"))),
                                    Load(name="b", input="b", index=(Var("k"), Var("j"))),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a", "b")),
                                    Accum(name="acc", value="p"),
                                ),
                            ),
                            Load(name="r", input="r", index=(Var("i"), Var("j"))),
                            Assign(name="e", op=ElementwiseImpl("add"), args=("acc", "r")),
                            Write(output="c", index=(Var("i"), Var("j")), value="e"),
                        ),
                    ),
                ),
            ),
        ),
    )


def _matmul_collapsed_k_loop_op(*, M: int = 64, N: int = 64, K: int = 64) -> LoopOp:
    """Matmul whose A operand spreads the K axis across two index dims (a
    collapsed-reshape access ``020_stage_inputs`` can't stage)."""
    i, j, k = Axis("i", M), Axis("j", N), Axis("k", K)
    return LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Loop(
                                axis=k,
                                body=(
                                    Load(name="a", input="a", index=(Var("k"), Var("k"))),
                                    Load(name="b", input="b", index=(Var("k"), Var("j"))),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a", "b")),
                                    Accum(name="acc", value="p"),
                                ),
                            ),
                            Write(output="c", index=(Var("i"), Var("j")), value="acc"),
                        ),
                    ),
                ),
            ),
        ),
    )


def test_mma_eligibility_rejects_fused_epilogue():
    """A matmul whose accumulator feeds a post-reduce ``add`` (fused residual)
    is gated off — the mma fragment-store path drops the epilogue and leaves
    the scalar accumulator undefined at codegen."""
    g = _matmul_graph(M=64, N=64, K=64, dtype=F16)
    op = _matmul_with_epilogue_loop_op()
    g.nodes["c"].op = op
    assert not is_atom_eligible(ATOM_REGISTRY["mma_m16n8k16_f16"], op, _ctx(cc=(9, 0)), graph=g)


def test_mma_eligibility_rejects_collapsed_k_operand():
    """An operand whose K axis spans two index dims (collapsed reshape, e.g.
    an attention output reaching the o_proj) can't be staged for ldmatrix."""
    g = _matmul_graph(M=64, N=64, K=64, dtype=F16)
    op = _matmul_collapsed_k_loop_op()
    g.nodes["c"].op = op
    assert not is_atom_eligible(ATOM_REGISTRY["mma_m16n8k16_f16"], op, _ctx(cc=(9, 0)), graph=g)


def test_mma_eligibility_rejects_in_kernel_intermediate():
    """An operand buffer also produced (Written) inside the same kernel is a
    register-resident intermediate — not a stage-able gmem input."""
    g = _matmul_graph(M=64, N=64, K=64, dtype=F16)
    i, j, k = Axis("i", 64), Axis("j", 64), Axis("k", 64)
    op = LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            # Produce buffer "a" in-kernel, then consume it as a matmul operand.
                            Load(name="bv", input="b", index=(Var("i"), Var("j"))),
                            Write(output="a", index=(Var("i"), Var("j")), value="bv"),
                            Loop(
                                axis=k,
                                body=(
                                    Load(name="a", input="a", index=(Var("i"), Var("k"))),
                                    Load(name="b2", input="b", index=(Var("k"), Var("j"))),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a", "b2")),
                                    Accum(name="acc", value="p"),
                                ),
                            ),
                            Write(output="c", index=(Var("i"), Var("j")), value="acc"),
                        ),
                    ),
                ),
            ),
        ),
    )
    g.nodes["c"].op = op
    assert not is_atom_eligible(ATOM_REGISTRY["mma_m16n8k16_f16"], op, _ctx(cc=(9, 0)), graph=g)


def test_unregistered_kind_raises():
    """The dispatcher has no fallback — an unknown kind surfaces a
    ``KeyError`` rather than silently returning False. "scalar" is the
    canonical example: it's the *absence* of an atom (modelled by
    ``ScalarTileParams``), not a registered kind."""
    g = _matmul_graph(dtype=F16)
    loop_op = g.nodes["c"].op
    with pytest.raises(KeyError):
        is_atom_eligible(Atom(name="scalar", shape=(1, 1, 1), operand_dtypes=(), group_size=1), loop_op, _ctx(), graph=g)


def test_atom_registry_priority_order():
    """The registry's insertion order is the source of truth for the planner's
    enumeration priority. The s16816 ``mma.sync`` path is now the sole
    tensor-core family (legacy WMMA removed): f16 first, then bf16."""
    assert tuple(ATOM_REGISTRY) == (
        "mma_m16n8k16_f16",
        "mma_m16n8k16_bf16",
    )


def test_registry_spec_shape_and_group_size():
    """Sanity-check the seeded spec — its shape drives M/N/K divisibility
    and its group_size drives the warp-tier launch geometry math."""
    spec = ATOM_REGISTRY["mma_m16n8k16_f16"]
    assert spec.shape == (16, 8, 16)
    assert spec.group_size == 32
    assert spec.operand_dtype("a") == F16
    assert spec.operand_dtype("b") == F16
    assert spec.operand_dtype("c") == F32


# --- Warp-tier enumerator (M3) ---------------------------------------


def _enum_warp(*, M: int, N: int, K: int, kinds: tuple[str, ...] = ("mma_m16n8k16_f16",)) -> list[WarpTileParams]:
    return _enumerate_warp_matmul_impl(
        E_M=M,
        E_N=N,
        E_K=K,
        ctx=_ctx(cc=(8, 0)),
        force_splitk_one=False,
        atoms=tuple(ATOM_REGISTRY[k] for k in kinds),
        m_axis_name="m",
        n_axis_name="n",
        m_forced_mask=False,
        n_forced_mask=False,
    )


def test_warp_enumerator_empty_when_no_atoms():
    """No atoms (the default at M1) → no rows."""
    assert _enum_warp(M=64, N=64, K=64, kinds=()) == []


def test_warp_enumerator_emits_rows_for_aligned_matmul():
    """A 64×64×64 matmul aligned to the 16×8×16 atom shape yields ≥1
    warp-tier variant (M%16==0, N%8==0, K%16==0)."""
    rows = _enum_warp(M=64, N=64, K=64)
    assert rows, "expected ≥1 row for a 64-square mma.sync-aligned matmul"
    assert all(r.atom.name == "mma_m16n8k16_f16" for r in rows)
    # Every row's WM·WN·32 must fit in the per-CTA thread budget (1024).
    assert all(r.wm * r.wn * 32 <= 1024 for r in rows)


def test_warp_enumerator_rejects_indivisible_extents():
    """Non-divisor M / N / K (vs the 16×8×16 atom shape) — no rows.
    M must divide 16, N must divide 8, K must divide 16."""
    assert _enum_warp(M=63, N=64, K=64) == []
    assert _enum_warp(M=64, N=60, K=64) == []
    assert _enum_warp(M=64, N=64, K=15) == []


def test_warp_enumerator_priority_orders_by_cells_sweet_spot():
    """Cells-near-16 ranks first; cells-far-from-16 ranks last. The
    register-budget sweet spot for ``mma_m16n8k16_*`` on sm_8x/9x/12x
    is FM·FN ≈ 16 (~120 regs/lane → 2 blocks/SM occupancy). The
    pre-2026 prior rewarded ``min(fm·fn, 64)`` monotonically and
    pushed the picker to FM=1 FN=32 cells=32 — 3.0× slower than cuBLAS
    on 2048² fp16. See ``plans/mma-warp-scoring.md``."""
    rows = _enum_warp(M=128, N=128, K=128)
    assert len(rows) >= 2
    first_dist = abs(rows[0].fm * rows[0].fn - 16)
    last_dist = abs(rows[-1].fm * rows[-1].fn - 16)
    assert first_dist <= last_dist


# --- Warp-tier scoring on TMA-capable hardware ----------------------


def test_warp_priority_prefers_small_bk_on_sm_90(monkeypatch):
    """On TMA-capable arches (sm_90+) ``_priority_matmul_warp`` lifts
    ``BK ≈ 2`` (and tied 128-thread CTAs) above the sm_80-era larger-BK
    + 256-thread preference. Bench-validated at 2048² fp16 on RTX 5090:
    ``WM=1 WN=4 FM=FN=4 BK=2`` runs 84 µs vs ``WM=1 WN=8 FM=FN=4 BK=32``
    at 108 µs — TMA-pipelined beats gmem-direct by ~22 %.
    """
    from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import _priority_matmul_warp

    gold = WarpTileParams(wn=4, wm=1, fm=4, fn=4, bk=2, splitk=1, atom=ATOM_REGISTRY["mma_m16n8k16_f16"])
    baseline = WarpTileParams(wn=8, wm=1, fm=4, fn=4, bk=32, splitk=1, atom=ATOM_REGISTRY["mma_m16n8k16_f16"])
    # On sm_90+ the gold tile must outscore the gmem-direct sibling.
    gold_score = _priority_matmul_warp(gold, ctx=_ctx(cc=(9, 0)))
    baseline_score = _priority_matmul_warp(baseline, ctx=_ctx(cc=(9, 0)))
    assert gold_score > baseline_score, f"gold {gold_score!r} should beat baseline {baseline_score!r} on sm_90"
    # The same call on sm_80 (no TMA) should KEEP preferring large BK and
    # 256-thread CTAs — the pre-2026 behavior is unchanged off-Hopper.
    gold_score_80 = _priority_matmul_warp(gold, ctx=_ctx(cc=(8, 0)))
    baseline_score_80 = _priority_matmul_warp(baseline, ctx=_ctx(cc=(8, 0)))
    assert baseline_score_80 > gold_score_80, "non-TMA arches must retain the legacy 256-thread + large-BK prior"


# --- Warp-tier knob narrow (M1, plans/mma-perf-closures.md) ----------


def test_warp_enumerator_wm_narrow(monkeypatch):
    """``DEPLODOCK_WM=2`` pin narrows the warp tier to ``wm=2`` only.
    Without this plumbing the impl iterates the full
    ``_TUNE_WARP_AXIS_CHOICES`` regardless and the CLI bench gate of
    every Phase A milestone fails to land on the targeted variant."""
    monkeypatch.setenv("DEPLODOCK_WM", "2")
    rows = _enum_warp(M=128, N=128, K=128)
    assert rows, "WM=2 should leave at least one variant at 128-square"
    assert all(r.wm == 2 for r in rows)


def test_warp_enumerator_wn_narrow(monkeypatch):
    """``DEPLODOCK_WN=2`` pin narrows the warp tier to ``wn=2`` only."""
    monkeypatch.setenv("DEPLODOCK_WN", "2")
    rows = _enum_warp(M=128, N=128, K=128)
    assert rows
    assert all(r.wn == 2 for r in rows)


def test_warp_enumerator_bk_narrow(monkeypatch):
    """``DEPLODOCK_BK=2`` pin narrows the warp tier to ``bk=2`` only.
    M2's staged-MMA bench gate pins ``BK=2`` so the picker lands on the
    pipelined-async variant."""
    monkeypatch.setenv("DEPLODOCK_BK", "2")
    rows = _enum_warp(M=128, N=128, K=128)
    assert rows
    assert all(r.bk == 2 for r in rows)


def test_warp_enumerator_atom_kind_alias_narrow(monkeypatch):
    """``DEPLODOCK_ATOM_KIND=mma_m16n8k16_f16`` — the ``MMA`` knob's alias
    spelling — pins the kind even when the caller passes a multi-kind tuple,
    scoping the picker to a single tensor-core family."""
    monkeypatch.delenv("DEPLODOCK_MMA", raising=False)
    monkeypatch.setenv("DEPLODOCK_ATOM_KIND", "mma_m16n8k16_f16")
    rows = _enum_warp(M=128, N=128, K=128, kinds=("mma_m16n8k16_f16", "mma_m16n8k16_bf16"))
    assert rows
    assert all(r.atom.name == "mma_m16n8k16_f16" for r in rows)


def test_mma_mode_decodes_three_way(monkeypatch):
    """``MMA`` knob string decode: unset / truthy → auto-enumerate, falsy →
    disabled, anything else → enabled with that atom kind pinned."""
    from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import mma_mode

    monkeypatch.delenv("DEPLODOCK_MMA", raising=False)
    assert mma_mode() == (True, None)
    for truthy in ("1", "true", "True", "yes", "on", ""):
        monkeypatch.setenv("DEPLODOCK_MMA", truthy)
        assert mma_mode() == (True, None), f"MMA={truthy!r} should auto-enumerate"
    for falsy in ("0", "false", "False", "no", "off"):
        monkeypatch.setenv("DEPLODOCK_MMA", falsy)
        assert mma_mode() == (False, None), f"MMA={falsy!r} should disable"
    monkeypatch.setenv("DEPLODOCK_MMA", "mma_m16n8k16_f16")
    assert mma_mode() == (True, "mma_m16n8k16_f16")


def test_warp_enumerator_mma_name_pins_kind(monkeypatch):
    """``DEPLODOCK_MMA=<kind name>`` pins the atom kind exactly like
    ``DEPLODOCK_ATOM_KIND`` — one knob enables + pins in one go."""
    monkeypatch.delenv("DEPLODOCK_ATOM_KIND", raising=False)
    monkeypatch.setenv("DEPLODOCK_MMA", "mma_m16n8k16_f16")
    rows = _enum_warp(M=128, N=128, K=128, kinds=("mma_m16n8k16_f16", "mma_m16n8k16_bf16"))
    assert rows
    assert all(r.atom.name == "mma_m16n8k16_f16" for r in rows)


def test_mma_primary_wins_over_atom_kind_alias(monkeypatch):
    """When both spellings are set, the primary ``DEPLODOCK_MMA`` wins over
    its ``DEPLODOCK_ATOM_KIND`` alias (``Knob.raw`` checks primary first)."""
    monkeypatch.setenv("DEPLODOCK_MMA", "mma_m16n8k16_bf16")
    monkeypatch.setenv("DEPLODOCK_ATOM_KIND", "mma_m16n8k16_f16")
    rows = _enum_warp(M=128, N=128, K=128, kinds=("mma_m16n8k16_f16", "mma_m16n8k16_bf16"))
    assert rows
    assert all(r.atom.name == "mma_m16n8k16_bf16" for r in rows)


def test_warp_enumerator_force_splitk_one():
    """``force_splitk_one`` clamps SPLITK choices to (1,) — every emitted
    row has splitk=1."""
    rows = _enumerate_warp_matmul_impl(
        E_M=64,
        E_N=64,
        E_K=64,
        ctx=_ctx(cc=(8, 0)),
        force_splitk_one=True,
        atoms=(ATOM_REGISTRY["mma_m16n8k16_f16"],),
        m_axis_name="m",
        n_axis_name="n",
        m_forced_mask=False,
        n_forced_mask=False,
    )
    assert rows
    assert all(r.splitk == 1 for r in rows)


# --- Planner end-to-end (M3) -----------------------------------------


_planner = __import__("importlib").import_module("deplodock.compiler.pipeline.passes.lowering.tile.010_partition_loops")


def test_planner_emits_warp_tower_when_mma_enabled(monkeypatch):
    """With ``DEPLODOCK_MMA=1`` and an F16 64×64×64 matmul, the planner
    emits at least one TileOp variant whose body contains a ``WarpTile``
    wrapping an ``AtomTile``. mma.sync auto-enumerates on sm_90+, so target
    sm_90 here; the single-warp variant is pruned (ldmatrix needs staged
    smem) so a multi-warp pin (WM=WN=2) keeps the tower."""
    from deplodock.compiler.ir.tile.ir import AtomTile, GridTile, TileOp, WarpTile
    from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import WarpTileParams

    monkeypatch.setenv("DEPLODOCK_MMA", "1")
    monkeypatch.setenv("DEPLODOCK_WM", "2")
    monkeypatch.setenv("DEPLODOCK_WN", "2")

    g = _matmul_graph(M=64, N=64, K=64, dtype=F16)
    loop_op = g.nodes["c"].op
    plan = _planner._plan_kernel(loop_op, _ctx(cc=(9, 0)), kernel_name="c", graph=g)
    assert plan is not None
    assert plan.params, "expected ≥1 enumerated variant"
    warp_rows = [p for p in plan.params if isinstance(p, WarpTileParams)]
    assert warp_rows, "expected ≥1 WarpTileParams row when DEPLODOCK_MMA=1"

    # Materialize the first warp row and verify the tower shape.
    tile_op = _planner._materialize(plan, warp_rows[0])
    assert isinstance(tile_op, TileOp)
    found_warp = False
    found_atom = False
    for s in tile_op.body.iter():
        if isinstance(s, WarpTile):
            found_warp = True
        if isinstance(s, AtomTile):
            found_atom = True
    assert found_warp, "warp variant body must contain a WarpTile"
    assert found_atom, "warp variant body must contain an AtomTile"
    # GridTile should still be the outermost tier.
    assert any(isinstance(s, GridTile) for s in tile_op.body), "warp variant must keep the GridTile outer wrapper"
    # Knobs reflect the warp tier.
    assert tile_op.knobs["ATOM_KIND"] == "mma_m16n8k16_f16"
    assert "BR" not in tile_op.knobs  # warp tier doesn't carry BR


def test_planner_scalar_only_when_mma_disabled(monkeypatch):
    """With ``DEPLODOCK_MMA=0`` set explicitly, the planner emits only
    scalar variants. (Default is now ON; setting ``0`` is the opt-out.)"""
    from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import ScalarTileParams, WarpTileParams

    monkeypatch.setenv("DEPLODOCK_MMA", "0")

    g = _matmul_graph(M=64, N=64, K=64, dtype=F16)
    loop_op = g.nodes["c"].op
    plan = _planner._plan_kernel(loop_op, _ctx(cc=(8, 0)), kernel_name="c", graph=g)
    assert plan is not None
    assert all(isinstance(p, ScalarTileParams) for p in plan.params)
    assert not any(isinstance(p, WarpTileParams) for p in plan.params)


def test_planner_mma_name_pin_enables_pre_sm90(monkeypatch):
    """``DEPLODOCK_MMA=<kind name>`` behaves like an atom-kind pin: it forces
    the kind on a pin-only arch (sm_80-89, where mma.sync doesn't
    auto-enumerate) — previously this took a separate ``DEPLODOCK_ATOM_KIND``
    pin on top of the gate."""
    from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import WarpTileParams

    monkeypatch.delenv("DEPLODOCK_ATOM_KIND", raising=False)
    monkeypatch.setenv("DEPLODOCK_MMA", "mma_m16n8k16_f16")
    monkeypatch.setenv("DEPLODOCK_WM", "2")
    monkeypatch.setenv("DEPLODOCK_WN", "2")

    g = _matmul_graph(M=64, N=64, K=64, dtype=F16)
    loop_op = g.nodes["c"].op
    plan = _planner._plan_kernel(loop_op, _ctx(cc=(8, 0)), kernel_name="c", graph=g)
    assert plan is not None
    warp_rows = [p for p in plan.params if isinstance(p, WarpTileParams)]
    assert warp_rows, "MMA=<kind> must enable the warp tier on a pin-only arch"
    assert all(p.atom.name == "mma_m16n8k16_f16" for p in warp_rows)
