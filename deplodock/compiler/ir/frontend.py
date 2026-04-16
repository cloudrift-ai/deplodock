"""Frontend (Torch) IR — ops captured directly from PyTorch tracing.

These ops exist in the graph between tracing and decomposition. Every one
of them has a decomposition rule in ``compiler/rules/decomposition/`` that
rewrites it into ``ir.tensor`` primitives (elementwise + reduce + indexmap
+ constants). After the decomposition pass completes, none of these ops
should remain in the graph.

Two groups:

1. **Layout-only ops** — ``TransposeOp``, ``ReshapeOp``, ``SliceOp``,
   ``CatOp``, ``UnsqueezeOp``. Rewritten to a single ``IndexMapOp`` each.
2. **Compound math ops** — ``LinearOp``, ``MatmulOp``, ``SdpaOp``,
   ``MeanOp``. Rewritten to elementwise/reduce chains (sometimes with
   inserted ``IndexMapOp`` unsqueezes so the broadcast contraction works).
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.base import Op, _drop_axis

# ---------------------------------------------------------------------------
# Layout-only ops (decomposed to IndexMapOp)
# ---------------------------------------------------------------------------


@dataclass
class TransposeOp(Op):
    """Permute dimensions.

    ``axes`` either lists a full permutation (``len(axes) == ndim``) or
    names two axes to swap (``len(axes) == 2``), matching torch's
    ``permute``/``transpose`` overloads.
    """

    axes: tuple[int, ...]

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        in_shape = input_shapes[0]
        ndim = len(in_shape)
        if len(self.axes) == 2:
            # Tracer convention: 2-tuple is always a swap (aten.transpose).
            a, b = self.axes[0] % ndim, self.axes[1] % ndim
            out = list(in_shape)
            out[a], out[b] = out[b], out[a]
            return tuple(out)
        return tuple(in_shape[a] for a in self.axes)

    def forward(self, *inputs):
        import numpy as np

        a = inputs[0]
        ndim = a.ndim
        if len(self.axes) == 2:
            ax0, ax1 = self.axes[0] % ndim, self.axes[1] % ndim
            return np.swapaxes(a, ax0, ax1)
        return np.transpose(a, self.axes)


@dataclass
class ReshapeOp(Op):
    """Reshape tensor without changing data."""

    shape: tuple[int | str, ...]

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        if -1 not in self.shape:
            return tuple(self.shape)
        in_numel = 1
        for d in input_shapes[0]:
            in_numel *= int(d)
        known = 1
        for d in self.shape:
            if d != -1:
                known *= int(d)
        resolved = list(self.shape)
        resolved[resolved.index(-1)] = in_numel // known if known else 1
        return tuple(resolved)

    def forward(self, *inputs):
        import numpy as np

        return np.reshape(inputs[0], self.shape)


@dataclass
class SliceOp(Op):
    """Extract a sub-tensor along a dimension.

    Inputs: [tensor, dim_const, start_const, end_const] where the
    constants are scalar ConstantOps from the tracer.
    """

    shape: tuple[int | str, ...]

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return tuple(self.shape)

    def forward(self, *inputs):
        tensor = inputs[0]
        dim = int(inputs[1].flat[0]) if len(inputs) > 1 else 0
        start = int(inputs[2].flat[0]) if len(inputs) > 2 else 0
        end = int(inputs[3].flat[0]) if len(inputs) > 3 else tensor.shape[dim]
        slices = [slice(None)] * tensor.ndim
        slices[dim] = slice(start, end)
        return tensor[tuple(slices)]


@dataclass
class CatOp(Op):
    """Concatenate tensors along a dimension.

    Inputs: [dim_const, tensor_1, tensor_2, ...] where dim_const
    is a scalar ConstantOp indicating the concat axis.
    """

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        # Tensor inputs are all but the trailing scalar dim-constant.
        # Find them by skipping shape-(1,) inputs at the tail.
        tensor_shapes = [s for s in input_shapes if len(s) > 1 or (len(s) == 1 and isinstance(s[0], int) and s[0] != 1)]
        if not tensor_shapes:
            return tuple(input_shapes[0])
        # Cat along the last dim by default (matches CatOp tracer convention).
        ndim = len(tensor_shapes[0])
        out = list(tensor_shapes[0])
        last = ndim - 1
        total = 0
        for s in tensor_shapes:
            d = s[last]
            if not isinstance(d, int):
                return tuple(out)  # symbolic; bail out
            total += d
        out[last] = total
        return tuple(out)

    def forward(self, *inputs):
        import numpy as np

        arrays = []
        dim = -1
        for inp in inputs:
            if inp.ndim == 0 or (inp.ndim == 1 and inp.size == 1):
                dim = int(inp.flat[0])
            else:
                arrays.append(inp)
        return np.concatenate(arrays, axis=dim)


@dataclass
class UnsqueezeOp(Op):
    """PyTorch aten.unsqueeze: add a size-1 dimension."""

    dim: int = 0

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        in_shape = list(input_shapes[0])
        d = self.dim if self.dim >= 0 else len(in_shape) + 1 + self.dim
        in_shape.insert(d, 1)
        return tuple(in_shape)

    def forward(self, *inputs):
        import numpy as np

        return np.expand_dims(inputs[0], axis=self.dim)


# ---------------------------------------------------------------------------
# Compound math ops (decomposed to elementwise + reduce chains)
# ---------------------------------------------------------------------------


@dataclass
class LinearOp(Op):
    """PyTorch aten.linear: output = x @ weight.T [+ bias]."""

    has_bias: bool = False

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        x_shape = input_shapes[0]
        w_shape = input_shapes[1]  # (out_features, in_features)
        return tuple(x_shape[:-1]) + (w_shape[-2],)

    def forward(self, *inputs):
        x, w = inputs[0], inputs[1]
        result = x @ w.T
        if self.has_bias:
            result = result + inputs[2]
        return result


@dataclass
class MatmulOp(Op):
    """PyTorch aten.mm/matmul/addmm: output = A @ B [+ bias]."""

    has_bias: bool = False

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        a_shape = input_shapes[0]
        b_shape = input_shapes[1]
        # Standard matmul: A(..., M, K) @ B(..., K, N) → (..., M, N)
        return tuple(a_shape[:-1]) + (b_shape[-1],)

    def forward(self, *inputs):
        a, b = inputs[0], inputs[1]
        result = a @ b
        if self.has_bias:
            result = result + inputs[2]
        return result


@dataclass
class SdpaOp(Op):
    """PyTorch scaled_dot_product_attention(Q, K, V, ...)."""

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        # SDPA output mirrors Q's batch+heads+seq dims, with V's last (head_dim).
        q_shape = input_shapes[0]
        v_shape = input_shapes[2]
        return tuple(q_shape[:-1]) + (v_shape[-1],)

    def forward(self, *inputs):
        import numpy as np

        q, k, v = inputs[0], inputs[1], inputs[2]
        # Align ndims: pad K/V with leading 1s to match Q's rank.
        while k.ndim < q.ndim:
            k = np.expand_dims(k, 0)
        while v.ndim < q.ndim:
            v = np.expand_dims(v, 0)
        # GQA: if Q has more heads than K/V, expand K/V by repeating heads.
        if q.ndim >= 3 and k.shape[-3] != q.shape[-3]:
            group = q.shape[-3] // k.shape[-3]
            k = np.repeat(k, group, axis=-3)
            v = np.repeat(v, group, axis=-3)
        d_k = q.shape[-1]
        scores = q @ np.swapaxes(k, -2, -1) / np.sqrt(d_k)
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        return attn @ v


@dataclass
class MeanOp(Op):
    """PyTorch aten.mean.dim: reduction that averages along an axis.

    Kept as its own op so the tracer does a faithful 1:1 capture; a
    decomposition rule rewrites it into sum + div.
    """

    axis: int | str = -1

    def infer_output_shape(self, input_shapes: list[tuple]) -> tuple:
        return _drop_axis(input_shapes[0], self.axis)

    def forward(self, *inputs):
        import numpy as np

        return np.mean(inputs[0], axis=self.axis)
