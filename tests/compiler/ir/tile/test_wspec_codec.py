"""The ``WSPEC`` warp-specialization codec — round-trip, the role registry, and legality.

``WarpSpec`` is the worker-mapping pin: a role→warp-count allocation (with per-role params) spelled
``<token><np>[:<param>,...]`` per role. Roles are registered descriptors (``role.py``); COMPUTE is
implicit. Pin-only this cut — these cover the codec + the legality gate, not materialization.
"""

from __future__ import annotations

import types

import pytest

from emmy.compiler.ir.schedule import ROLE_REGISTRY, WarpSpec, role_for


@pytest.mark.parametrize("spec", ["", "p2", "p2:q8", "p2:q8/s1", "s1"])
def test_wspec_round_trips(spec: str) -> None:
    assert WarpSpec.parse(spec).spell() == spec


def test_empty_is_uniform() -> None:
    assert WarpSpec.parse("").roles == ()
    assert WarpSpec.parse("").aux_warps == 0


def test_aux_warps_sums_non_compute_bands() -> None:
    assert WarpSpec.parse("p2:q8/s1").aux_warps == 3  # COMPUTE (TilePlan.units) is not counted here


def test_default_params_dropped() -> None:
    # q defaults to 1 — a bare p2 stores no q param, so it spells back without one.
    ws = WarpSpec.parse("p2")
    assert ws.roles[0].params == ()


def test_role_for_and_unknown() -> None:
    assert role_for("p").token == "p"
    assert set(ROLE_REGISTRY) >= {"p", "s"}
    with pytest.raises(ValueError, match="unknown warp-spec role"):
        role_for("z")


@pytest.mark.parametrize("spec", ["pq", "p2:zz9", "x1"])
def test_malformed_raises_value_error(spec: str) -> None:
    with pytest.raises(ValueError, match="WSPEC"):
        WarpSpec.parse(spec)


def test_producer_legality_needs_stage() -> None:
    # The producer role is meaningful only when the schedule stages operands (a stage to drive).
    ws = WarpSpec.parse("p2")
    assert ws.is_legal(types.SimpleNamespace(stage=object()))
    assert not ws.is_legal(types.SimpleNamespace(stage=None))
