__global__ void __KERNEL_NAME__(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int rows, int dim, float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_x = x + row * dim;
    float* row_out = out + row * dim;

    // Phase 1: compute sum of squares (parallel reduction).
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = row_x[i];
        local_ss += v * v;
    }

    // Warp-level reduction.
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_ss += __shfl_down_sync(0xffffffff, local_ss, offset);

    // Cross-warp reduction via shared memory.
    __shared__ float warp_sums[8];  // up to 256 threads = 8 warps
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = local_ss;
    __syncthreads();

    float ss = 0.0f;
    if (threadIdx.x < blockDim.x / warpSize)
        ss = warp_sums[threadIdx.x];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        ss += __shfl_down_sync(0xffffffff, ss, offset);

    // Broadcast rsqrt to all threads.
    __shared__ float s_rsqrt;
    if (threadIdx.x == 0)
        s_rsqrt = rsqrtf(ss / (float)dim + eps);
    __syncthreads();

    float scale = s_rsqrt;

    // Phase 2: write normalized + scaled output.
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        row_out[i] = row_x[i] * scale * weight[i];
}
