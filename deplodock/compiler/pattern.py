"""Pattern AST and text parser for graph matching rules."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Pattern AST
# ---------------------------------------------------------------------------


@dataclass
class PatternVar:
    """Captures a subgraph; same name = same node (fan-out)."""

    name: str


@dataclass
class PatternWildcard:
    """Match any single node (_) or any subgraph (__)."""

    greedy: bool = False


@dataclass
class PatternNode:
    """Match a specific op type with optional constraints on fields."""

    op_class: str  # e.g. "Reduce", "Elementwise" — mapped to ReduceOp, ElementwiseOp
    constraints: dict[str, str] = field(default_factory=dict)
    inputs: list[PatternNode | PatternVar | PatternWildcard] = field(default_factory=list)


Pattern = PatternNode | PatternVar | PatternWildcard


# ---------------------------------------------------------------------------
# Op class name mapping — pattern short names to actual class names
# ---------------------------------------------------------------------------

_OP_CLASS_MAP: dict[str, str] = {
    "Elementwise": "ElementwiseOp",
    "Reduce": "ReduceOp",
    "Scan": "ScanOp",
    "Gather": "GatherOp",
    "Scatter": "ScatterOp",
    "FusedReduceElementwise": "FusedReduceElementwiseOp",
    "Constant": "ConstantOp",
    "Transpose": "TransposeOp",
    "Reshape": "ReshapeOp",
    "Matmul": "MatmulOp",
    "FusedRMSNorm": "FusedRMSNormOp",
    "FusedSoftmax": "FusedSoftmaxOp",
    "FusedSiLUMul": "FusedSiLUMulOp",
    "FusedAttention": "FusedAttentionOp",
}

# Field ordering per op class — maps positional constraints to field names.
_OP_FIELDS: dict[str, list[str]] = {
    "Elementwise": ["fn"],
    "Reduce": ["fn", "axis"],
    "Scan": ["fn", "axis"],
    "Gather": ["axis"],
    "Scatter": ["axis", "reduce_fn"],
    "FusedReduceElementwise": ["reduce_fn", "elementwise_fn", "axis"],
    "Constant": ["name"],
    "Transpose": ["axes"],
    "Reshape": ["shape"],
    "Matmul": [],
    "FusedRMSNorm": ["eps"],
    "FusedSoftmax": ["axis"],
    "FusedSiLUMul": [],
    "FusedAttention": ["num_heads", "head_dim", "scale"],
}


def resolve_op_class(short_name: str) -> str:
    """Map pattern short name to actual op class name."""
    if short_name not in _OP_CLASS_MAP:
        raise ValueError(f"Unknown op class {short_name!r}. Known: {', '.join(_OP_CLASS_MAP)}")
    return _OP_CLASS_MAP[short_name]


# ---------------------------------------------------------------------------
# Text parser
# ---------------------------------------------------------------------------
#
# Grammar (informal):
#   alternatives := pattern ('|' pattern)*
#   pattern      := op_node | var | wildcard
#   op_node      := NAME '{' constraints '}' '(' args ')'
#                 | NAME '(' args ')'       (no constraints)
#   constraints  := value (',' value)*
#   args         := pattern (',' pattern)*  | empty
#   var          := '$' NAME
#   wildcard     := '__' | '_'
#
# Examples:
#   Reduce{sum, $k}(Elementwise{mul}($A, $B))
#   Reduce{sum,$k}(Elementwise{mul}($A,$B)) | Reduce{sum,$k}(Elementwise{mul}($B,$A))


class _Parser:
    """Recursive descent parser for pattern text syntax."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def _skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t\n\r":
            self.pos += 1

    def _peek(self) -> str:
        self._skip_ws()
        if self.pos >= len(self.text):
            return ""
        return self.text[self.pos]

    def _consume(self, char: str) -> None:
        self._skip_ws()
        if self.pos >= len(self.text) or self.text[self.pos] != char:
            raise ValueError(f"Expected {char!r} at position {self.pos}, got {self.text[self.pos : self.pos + 1]!r}")
        self.pos += 1

    def _read_name(self) -> str:
        self._skip_ws()
        match = re.match(r"[A-Za-z_]\w*", self.text[self.pos :])
        if not match:
            raise ValueError(f"Expected name at position {self.pos}")
        name = match.group()
        self.pos += len(name)
        return name

    def _read_constraint_value(self) -> str:
        self._skip_ws()
        match = re.match(r"[\$A-Za-z_][\w]*|[0-9]+", self.text[self.pos :])
        if not match:
            raise ValueError(f"Expected constraint value at position {self.pos}")
        val = match.group()
        self.pos += len(val)
        return val

    def parse_alternatives(self) -> list[Pattern]:
        """Parse top-level: pattern ('|' pattern)*."""
        patterns = [self.parse_pattern()]
        while True:
            self._skip_ws()
            if self.pos < len(self.text) and self.text[self.pos] == "|":
                self.pos += 1
                patterns.append(self.parse_pattern())
            else:
                break
        return patterns

    def parse_pattern(self) -> Pattern:
        """Parse a single pattern (op_node, var, or wildcard)."""
        self._skip_ws()
        if self.pos >= len(self.text):
            raise ValueError("Unexpected end of pattern")

        # Variable: $NAME
        if self.text[self.pos] == "$":
            self.pos += 1
            name = self._read_name()
            return PatternVar(name=name)

        # Wildcard: __ or _
        if self.text[self.pos] == "_":
            if self.pos + 1 < len(self.text) and self.text[self.pos + 1] == "_":
                self.pos += 2
                return PatternWildcard(greedy=True)
            self.pos += 1
            return PatternWildcard(greedy=False)

        # Op node: NAME{constraints}(inputs) or NAME(inputs)
        name = self._read_name()
        constraints: dict[str, str] = {}

        self._skip_ws()
        if self.pos < len(self.text) and self.text[self.pos] == "{":
            self._consume("{")
            field_names = _OP_FIELDS.get(name, [])
            idx = 0
            while self._peek() != "}":
                val = self._read_constraint_value()
                if idx < len(field_names):
                    constraints[field_names[idx]] = val
                else:
                    constraints[f"_arg{idx}"] = val
                idx += 1
                self._skip_ws()
                if self.pos < len(self.text) and self.text[self.pos] == ",":
                    self.pos += 1
            self._consume("}")

        inputs: list[Pattern] = []
        self._skip_ws()
        if self.pos < len(self.text) and self.text[self.pos] == "(":
            self._consume("(")
            if self._peek() != ")":
                inputs.append(self.parse_pattern())
                while self._peek() == ",":
                    self._consume(",")
                    inputs.append(self.parse_pattern())
            self._consume(")")

        return PatternNode(
            op_class=resolve_op_class(name),
            constraints=constraints,
            inputs=inputs,
        )


def parse_pattern(text: str) -> list[Pattern]:
    """Parse pattern text into a list of alternatives.

    Returns a list with one element for simple patterns, or multiple
    elements for '|'-separated alternatives.
    """
    parser = _Parser(text.strip())
    alternatives = parser.parse_alternatives()
    return alternatives
