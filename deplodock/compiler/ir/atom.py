"""The contraction **atom** — one leaf multiply-accumulate the contraction tier tiles over.

An atom is the smallest cell a :class:`~deplodock.compiler.ir.tile.structural.Contraction` tiles four
ways (GRID / UNIT / REGISTER / ATOM). Two kinds, one interface (``shape`` + :attr:`lanes`):

- :class:`AtomKind` — a tensor-core ``mma.sync`` cell: a fixed ``(m, n, k)`` shape, per-operand
  dtypes (``a``/``b`` the f16/bf16 multiplicands, ``c`` the f32 accumulator), and ``lanes == 32``
  (the warp that executes one mma cooperatively — its 32 lanes hold the fixed PTX fragment layout).
- :class:`ScalarAtom` — a plain scalar fma cell: ``(1, 1, 1)`` and ``lanes == 1`` (one thread). No
  operand-dtype spec — the scalar cell folds the carrier directly, dtypes resolved by the body.

The **unit** is the atom's parallel thread footprint (:attr:`lanes`): a warp (32) for mma, a single
thread (1) for scalar. So the warp tile and the scalar parallel thread-tile are the *same* tiling
level — a 2-D grid of units — differing only in ``lanes``. ``block_threads = units · lanes``.

``AtomKind`` rides on the ``atom`` field of :class:`~deplodock.compiler.ir.schedule.TilePlan`;
the kernel-IR ``MmaSyncPtx`` / ``RegFragment`` / ``RegStore`` render off its ``shape`` +
``operand_dtype`` per role; the ``TILE`` codec names an mma atom by its registry key
(:data:`ATOM_REGISTRY`). The cooperative-reduce combine (``WarpShuffle`` / ``TreeHalve``) needs no
atom spec — it works on arbitrary carried state with a derived mechanism — so it has no atom.

Kept dependency-free (dtypes only) so ``ir/tile`` and the kernel materializer import it;
``pipeline/knob.py`` deliberately does NOT import it (the atom's geometry reaches the
featurizer through the knob layer, not this module).
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.dtype import BF16, F16, F32, DataType


@dataclass(frozen=True)
class AtomKind:
    """One tensor-core mma cell: a fixed ``(m, n, k)`` shape + per-operand dtypes.

    ``operand_dtypes`` maps each role (``"a"`` / ``"b"`` / ``"c"``) to its element dtype;
    ``a``/``b`` are the multiplicands (f16 or bf16), ``c`` the f32 accumulator. Frozen + hashable so
    it rides on a frozen ``TilePlan`` / ``Contraction``. :attr:`lanes` is 32 — an mma is
    warp-cooperative (one warp executes a cell)."""

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
    def lanes(self) -> int:
        """The atom's parallel thread footprint — 32 (one warp executes an mma.sync cell)."""
        return 32

    @property
    def ab_dtype(self) -> str:
        """The shared multiplicand dtype token (``"f16"`` / ``"bf16"``) the ``MmaSyncPtx``
        wrapper selects on (``dpl_mma_m16n8k16_{f16,bf16}``)."""
        return self.operand_dtype("a").name


@dataclass(frozen=True)
class ScalarAtom:
    """The scalar fma leaf — one thread folds one output cell. The degenerate atom: a ``(1, 1,
    1)`` cell with ``lanes == 1``. No operand-dtype spec (the scalar body folds the carrier
    directly, dtypes resolved during lowering), so it carries only its identity. Selected when a
    contraction has no tensor-core ``TilePlan`` (the register-tile / per-cell tier)."""

    name: str = "scalar"
    shape: tuple[int, int, int] = (1, 1, 1)

    @property
    def atom_m(self) -> int:
        return 1

    @property
    def atom_n(self) -> int:
        return 1

    @property
    def atom_k(self) -> int:
        return 1

    @property
    def lanes(self) -> int:
        """The atom's parallel thread footprint — 1 (one thread folds a scalar cell)."""
        return 1


#: The singleton scalar atom — the register-tile / per-cell contraction leaf.
SCALAR_ATOM = ScalarAtom()

#: A contraction leaf atom — the tensor-core mma cell or the scalar fma cell. One interface
#: (``shape`` / ``atom_m`` / ``atom_n`` / ``atom_k`` / ``lanes``); the expansion's ``Unit`` is
#: selected by which kind.
Atom = AtomKind | ScalarAtom


#: The registered mma atoms, keyed by the name the ``TILE`` codec (``a:<name>``) spells.
#: The s16816 mma.sync family — one cell shape, f16 / bf16 multiplicands, f32 accumulator.
ATOM_REGISTRY: dict[str, AtomKind] = {
    "mma_m16n8k16_f16": AtomKind("mma_m16n8k16_f16", (16, 8, 16), (("a", F16), ("b", F16), ("c", F32))),
    "mma_m16n8k16_bf16": AtomKind("mma_m16n8k16_bf16", (16, 8, 16), (("a", BF16), ("b", BF16), ("c", F32))),
}


def atom_for(name: str) -> AtomKind:
    """The registered :class:`AtomKind` for ``name`` (the ``TILE`` codec's ``a:<name>``)."""
    try:
        return ATOM_REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown atom kind {name!r} (have {sorted(ATOM_REGISTRY)})") from None


__all__ = ["ATOM_REGISTRY", "AtomKind", "atom_for"]
