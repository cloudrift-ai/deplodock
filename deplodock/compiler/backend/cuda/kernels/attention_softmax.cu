__global__ void __KERNEL_NAME__(
    float* __restrict__ scores,
    int batch_heads, int seq_len
) {
    int row_idx = blockIdx.x;
    if (row_idx >= batch_heads * seq_len) return;

    float* row = scores + row_idx * seq_len;

    // Phase 1: find row max.
    float local_max = -1e30f;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x)
        local_max = fmaxf(local_max, row[j]);

    // Warp reduce max.
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));

    __shared__ float warp_vals[8];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_vals[warp_id] = local_max;
    __syncthreads();

    float row_max = -1e30f;
    if (threadIdx.x < blockDim.x / warpSize)
        row_max = warp_vals[threadIdx.x];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = row_max;
    __syncthreads();
    row_max = s_max;

    // Phase 2: exp(x - max) and sum.
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        float v = expf(row[j] - row_max);
        row[j] = v;
        local_sum += v;
    }

    // Warp reduce sum.
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    if (lane == 0) warp_vals[warp_id] = local_sum;
    __syncthreads();

    float row_sum = 0.0f;
    if (threadIdx.x < blockDim.x / warpSize)
        row_sum = warp_vals[threadIdx.x];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = row_sum;
    __syncthreads();
    float inv_sum = 1.0f / s_sum;

    // Phase 3: normalize.
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x)
        row[j] *= inv_sum;
}
