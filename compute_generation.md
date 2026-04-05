# Tensor

Multidimensional array like numpy.

- **Reshape/View**: zero-cost reinterpretation of the shape without moving data (reshape, squeeze, unsqueeze, flatten).
- **Broadcasting**: elementwise ops on operands with different shapes follow numpy broadcasting rules — dimensions of size 1 are expanded to match.

# Operation Set

1. **Elementwise**: `fn([x1, x2, ...], [y1, y2, ...]) = [fn(x1, y1), fn(x2, y2), ...]`
   Apply a scalar function independently to each element. Covers add, mul, exp, gelu, etc.

2. **Reduce**: `reduce(fn, x, axes) -> y` where `rank(y) = rank(x) - len(axes)`
   Collapse one or more dimensions via an associative binary op (sum, max, prod).
   Gives dot products, softmax denominators, pooling.

3. **Scan**: `scan(fn, x, axis) = [x1, fn(x1, x2), fn(fn(x1, x2), x3), ...]`
   Cumulative application of an associative binary op along an axis (prefix sum, cummax).
   Needed for causal masking, parallel scan attention variants.

4. **Gather**: `gather(x, indices, axis) -> y`
   Read elements from arbitrary positions. Generalizes permutation to non-bijective index maps.
   Covers embedding lookups, sparse attention, advanced indexing.

5. **Scatter**: `scatter(x, indices, updates, axis, reduce_fn) -> y`
   Write (or reduce) values into arbitrary positions. Inverse of gather.
   Covers scatter-add in GNNs, histogram-like operations.
