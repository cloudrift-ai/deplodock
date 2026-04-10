__global__ void __KERNEL_NAME__(
    const float* __restrict__ scores,
    const float* __restrict__ V,
    float* __restrict__ out,
    int batch_heads, int seq_len, int head_dim
) {
    int bh = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (bh < batch_heads && i < seq_len && d < head_dim) {
        const float* score_row = scores + (bh * seq_len + i) * seq_len;
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++)
            acc += score_row[j] * V[(bh * seq_len + j) * head_dim + d];

        out[(bh * seq_len + i) * head_dim + d] = acc;
    }
}
