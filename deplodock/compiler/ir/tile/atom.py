"""The tensor-core **atom** — one hardware mma instruction's cell spec.

An :class:`AtomKind` is the smallest tensor-core multiply-accumulate the warp tier tiles
over: a fixed ``(m, n, k)`` cell + its per-operand dtypes (``a``/``b`` the f16/bf16
multiplicands, ``c`` the f32 accumulator). It rides on the ``atom`` field of
:class:`~deplodock.compiler.ir.tile.schedule.WarpTile`; the kernel-IR ``MmaSyncPtx`` /
``RegFragment`` / ``RegStore`` render off its ``shape`` + ``operand_dtype`` per role; the
``TILE`` codec names an atom by its registry key (:data:`ATOM_REGISTRY`).

The s16816 family (``mma.sync.aligned.m16n8k16``) is the sole atom today — one cell shape,
two ab-dtypes (f16 / bf16). The cooperative-reduce combine (``WarpShuffle`` / ``TreeHalve``)
needs no atom spec — it works on arbitrary carried state, and its accumulator dtype + the
shuffle-vs-tree mechanism are derived (off the carrier + ``ReduceStage.combine``), so there's
no ``MonoidAtom`` sibling.

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
