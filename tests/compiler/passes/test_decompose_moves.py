"""The carrier-parameterized decomposition move (``partition/decompose.py``).

``legal_decomps`` is the ONE move behind split-K, cooperative reduce, and
strip-mine: factor a reduce axis, with legality read off the carrier's algebra.
These tests pin the trait gating and prove the matmul / coop offer functions
(which now delegate to it) stay byte-identical to a reference enumeration. See
``plans/algebra-licensed-decomposition-moves.md`` (phase 3).
"""

from __future__ import annotations

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.stmt import Accum
from deplodock.compiler.pipeline.passes.lowering.tile.partition._tower import Role
from deplodock.compiler.pipeline.passes.lowering.tile.partition.decompose import AxisDecomp, legal_decomps

_ADD = Accum(name="acc", value="v", op=ElementwiseImpl("add"))
_SUB = Accum(name="acc", value="v", op=ElementwiseImpl("subtract"))  # non-associative


def _ax() -> Axis:
    return Axis("k", 64)


def test_associative_carrier_licenses_split():
    decomps = legal_decomps(
        _ADD, _ax(), 64, factor_menus=[[1, 2, 4], [1], [1]], placement=[Role.BLOCK, Role.STAGE_INNER, Role.REGISTER], masked=False
    )
    splits = sorted(d.factors[0] for d in decomps)
    assert splits == [1, 2, 4]  # 2 and 4 divide 64


def test_non_associative_carrier_only_trivial():
    # A subtract carrier can't be split — only the all-1 factorization survives.
    decomps = legal_decomps(
        _SUB, _ax(), 64, factor_menus=[[1, 2, 4], [1, 2], [1]], placement=[Role.BLOCK, Role.STAGE_INNER, Role.REGISTER], masked=False
    )
    assert all(f == 1 for d in decomps for f in d.factors)
    assert decomps == [AxisDecomp(axis=_ax(), factors=(1, 1, 1), placement=(Role.BLOCK, Role.STAGE_INNER, Role.REGISTER))]


def test_allow_split_false_forbids_partition():
    # The cost/soundness gate (epilogue / multi-accum) forbids the partition factor
    # even for an associative+commutative carrier.
    decomps = legal_decomps(
        _ADD,
        _ax(),
        64,
        factor_menus=[[1, 2, 4], [1], [1]],
        placement=[Role.BLOCK, Role.STAGE_INNER, Role.REGISTER],
        masked=False,
        allow_split=False,
    )
    assert sorted(d.factors[0] for d in decomps) == [1]


def test_masked_requires_identity():
    # has_identity licenses the ceil-div mask; a fill-less carrier yields nothing.
    ident = legal_decomps(
        _ADD, _ax(), 64, factor_menus=[[1], [8], [1]], placement=[Role.THREAD, Role.STAGE_INNER, Role.REGISTER], masked=True
    )
    assert ident  # add has identity 0 → masked fill legal
    none = legal_decomps(
        _SUB, _ax(), 64, factor_menus=[[1], [8], [1]], placement=[Role.THREAD, Role.STAGE_INNER, Role.REGISTER], masked=True
    )
    assert none == []


def test_exact_divide_unless_masked():
    # Unmasked: only products dividing the extent. bk=5 never divides 64.
    decomps = legal_decomps(
        _ADD, _ax(), 64, factor_menus=[[1], [5, 8], [1]], placement=[Role.THREAD, Role.STAGE_INNER, Role.REGISTER], masked=False
    )
    assert sorted(d.factors[1] for d in decomps) == [8]
    # Masked: a non-dividing chunk is allowed (ceil-div + identity fill).
    masked = legal_decomps(
        _ADD, _ax(), 64, factor_menus=[[1], [5, 8], [1]], placement=[Role.THREAD, Role.STAGE_INNER, Role.REGISTER], masked=True
    )
    assert sorted(d.factors[1] for d in masked) == [5, 8]


# --------------------------------------------------------------------------- #
# Byte-identical: the offer functions match a reference enumeration.
# --------------------------------------------------------------------------- #


def _ref_matmul(k_extent, *, allow_split, sk_choices, bk_choices, fk_choices):
    sks = (1,) if not allow_split else sk_choices
    out = [
        (bk, fk, sk) for sk in sks for bk in bk_choices for fk in fk_choices if sk * bk * fk <= k_extent and k_extent % (sk * bk * fk) == 0
    ]
    out.sort(key=lambda t: (t[2] != 1, -t[0], t[1], t[2]))
    return out


def test_one_move_expresses_splitk_coop_and_splitkv():
    """The forcing-function proof (plan phase 5): ONE carrier-trait-gated
    ``legal_decomps`` call expresses split-K (additive ``Accum``), cooperative
    reduce (``max`` ``Accum``), AND flash split-KV (the online-softmax LSE
    ``Monoid``) — same partition factorizations, differing only in the carrier
    whose ``combine_partials`` recombines the pieces. This validates the
    ``AxisDecomp`` abstraction across all three instances before any deletion."""
    from deplodock.compiler.pipeline.passes.loop.recognize._flash import flash_combine

    ax = Axis("kv", 256)
    menus = [[1, 2, 4, 8], [1], [1]]
    place = [Role.BLOCK, Role.SERIAL_OUTER, Role.REGISTER]

    _max = Accum(name="m", value="v", op=ElementwiseImpl("maximum"))
    splitk = legal_decomps(_ADD, ax, 256, factor_menus=menus, placement=place, masked=False)
    coop = legal_decomps(_max, ax, 256, factor_menus=menus, placement=place, masked=False)
    flash = flash_combine("m", "l", "O", "s", "V")  # the FlashCombine LSE Monoid
    assert flash.commutative and flash.associative and flash.has_identity  # licenses the split
    splitkv = legal_decomps(flash, ax, 256, factor_menus=menus, placement=place, masked=False)

    parts = lambda decs: sorted(d.factors[0] for d in decs)  # noqa: E731
    assert parts(splitk) == parts(coop) == parts(splitkv) == [1, 2, 4, 8]
    # The recombine is derived from the carrier: an additive op-fold for the
    # Accums, the 12-step state-merge for the flash LSE — one operator surface.
    assert len(_ADD.combine_partials()) == 1
    assert len(flash.combine_partials()) == 12  # the authored combine_states (m,l,O merge)


def test_matmul_offers_byte_identical():
    from deplodock.compiler.ir.expr import Var
    from deplodock.compiler.ir.loop import Axis as LAxis
    from deplodock.compiler.ir.loop import Load, Loop, Write
    from deplodock.compiler.pipeline.passes.lowering.tile.partition.knobs import BK_CHOICES, FK_CHOICES, SPLITK_CHOICES
    from deplodock.compiler.pipeline.passes.lowering.tile.partition.moves import matmul_reduce_offers
    from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import MatmulSkeleton, _map_axis

    for k in (64, 128, 96):
        kl = Loop(axis=LAxis("k", k), body=(Load(name="a", input="a", index=(Var("k"),)), Accum(name="acc", value="a")))
        nl = Loop(axis=LAxis("n", 128), body=(kl, Write(output="o", index=(Var("n"),), value="acc")))
        skel = MatmulSkeleton(
            inner_n=_map_axis(nl),
            outer_m=_map_axis(Loop(axis=LAxis("m", 128), body=())),
            extra_outer=(),
            k_loop=kl,
            k_name="k",
            k_extent=k,
            inner_body=(kl, Write(output="o", index=(Var("n"),), value="acc")),
            leading=(),
            target_names=frozenset({"k"}),
        )
        ref = _ref_matmul(k, allow_split=True, sk_choices=SPLITK_CHOICES, bk_choices=BK_CHOICES, fk_choices=FK_CHOICES)
        assert matmul_reduce_offers(skel) == ref
