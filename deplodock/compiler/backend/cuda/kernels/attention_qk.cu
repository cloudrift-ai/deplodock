__global__ void __KERNEL_NAME__(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ scores,
    int batch_heads, int seq_len, int head_dim, float scale
) {
    int bh = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (bh < batch_heads && i < seq_len && j < seq_len) {
        const float* q_row = Q + (bh * seq_len + i) * head_dim;
        const float* k_row = K + (bh * seq_len + j) * head_dim;

        float acc = 0.0f;
        for (int d = 0; d < head_dim; d++)
            acc += q_row[d] * k_row[d];

        scores[bh * seq_len * seq_len + i * seq_len + j] = acc * scale;
    }
}
