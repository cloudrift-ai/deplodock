"""The hardware **atom** — a fixed-shape primitive realizing one algebra kind, subtyped by
that kind (the same ``Map`` / ``Monoid`` / ``Semiring`` trichotomy as the ``*Kernel`` types).

An atom is "an algebra kind realized at a fixed shape by a hardware primitive." The two kinds
are deliberately asymmetric, because the hardware is:

- :class:`SemiringAtom` — one tensor-core mma cell: a fixed ``(m, n, k)`` shape + per-operand
  dtypes (``a``/``b`` the f16/bf16 multiplicands, ``c`` the f32 accumulator). The
  **constrained** kind: tensor cores impose a fixed fragment geometry, so an atom is selected
  **by name** from :data:`ATOM_REGISTRY` (the ``TILE`` codec's ``a:<name>``). It rides on the
  ``atom`` field of :class:`~deplodock.compiler.ir.tile.schedule.WarpTile`; the kernel-IR
  ``MmaSyncPtx`` / ``RegFragment`` / ``RegStore`` render off its ``shape`` + ``operand_dtype``.
- :class:`MonoidAtom` — the cooperative-combine atom for a ``Monoid`` reduce. The **generic**
  kind: a warp-shuffle / smem-tree combine works on arbitrary carried state, so it carries NO
  fixed shape and NO stored realization — the shuffle-vs-tree mechanism is **derived** from the
  level + width by ``ReduceStage.combine`` (storing it would be the recovered tag the rebuild
  forbids). All it carries is the per-component accumulator dtype, the one fact the renderer
  (``WarpShuffle`` / ``TreeHalve``) needs that isn't on the carrier or the ``ReducePlan``.

This asymmetry is fundamental, not an oversight: mma needs a named fixed geometry; the
cooperative combine does not. ``SemiringAtom`` is the special case; ``MonoidAtom`` is already
the general thing.

Kept dependency-free (dtypes only) so ``ir/tile`` and the kernel materializer import it;
``pipeline/knob.py`` deliberately does NOT import it (the atom's geometry reaches the
featurizer through the knob layer, not this module).
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.dtype import BF16, F16, F32, DataType


@dataclass(frozen=True)
class SemiringAtom:
    """One tensor-core mma cell: a fixed ``(m, n, k)`` shape + per-operand dtypes.

    ``operand_dtypes`` maps each role (``"a"`` / ``"b"`` / ``"c"``) to its element dtype;
    ``a``/``b`` are the multiplicands (f16 or bf16), ``c`` the f32 accumulator. Frozen +
    hashable so it rides on a frozen ``Mma`` / ``WarpTile``."""

    name: str
    shape: tuple[int, int, int]
    operand_dtypes: tuple[tuple[str, DataType], ...]

    def operand_dtype(self, role: str) -> DataType:
        """The element dtype of operand ``role`` (``"a"`` / ``"b"`` / ``"c"``)."""
        for r, dt in self.operand_dtypes:
            if r == role:
                return dt
        raise KeyError(f"atom {self.name!r} has no operand role {role!r}")

    @property
    def atom_m(self) -> int:
        return self.shape[0]

    @property
    def atom_n(self) -> int:
        return self.shape[1]

    @property
    def atom_k(self) -> int:
        return self.shape[2]

    @property
    def ab_dtype(self) -> str:
        """The shared multiplicand dtype token (``"f16"`` / ``"bf16"``) the ``MmaSyncPtx``
        wrapper selects on (``dpl_mma_m16n8k16_{f16,bf16}``)."""
        return self.operand_dtype("a").name


@dataclass(frozen=True)
class MonoidAtom:
    """The cooperative-combine atom for a ``Monoid`` reduce — deliberately thin (see the
    module docstring). Carries only the per-component accumulator dtype; the shuffle-vs-tree
    realization is derived (``ReduceStage.combine``), never stored. One ``dtype`` spans every
    carried-state component (the current ``WarpShuffle`` / ``TreeHalve`` fold a whole tuple at
    one accumulator dtype)."""

    dtype: DataType = F32


#: Every atom — a ``SemiringAtom`` (mma cell) or a ``MonoidAtom`` (cooperative combine).
Atom = SemiringAtom | MonoidAtom


#: The registered mma atoms, keyed by the name the ``TILE`` codec (``a:<name>``) spells.
#: The s16816 mma.sync family — one cell shape, f16 / bf16 multiplicands, f32 accumulator.
ATOM_REGISTRY: dict[str, SemiringAtom] = {
    "mma_m16n8k16_f16": SemiringAtom("mma_m16n8k16_f16", (16, 8, 16), (("a", F16), ("b", F16), ("c", F32))),
    "mma_m16n8k16_bf16": SemiringAtom("mma_m16n8k16_bf16", (16, 8, 16), (("a", BF16), ("b", BF16), ("c", F32))),
}


def atom_for(name: str) -> SemiringAtom:
    """The registered :class:`SemiringAtom` for ``name`` (the ``TILE`` codec's ``a:<name>``)."""
    try:
        return ATOM_REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown atom kind {name!r} (have {sorted(ATOM_REGISTRY)})") from None


__all__ = ["ATOM_REGISTRY", "Atom", "MonoidAtom", "SemiringAtom", "atom_for"]
