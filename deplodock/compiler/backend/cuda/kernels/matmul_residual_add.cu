__global__ void __KERNEL_NAME__(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ residual,
    float* __restrict__ out,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += A[row * K + k] * B[k * N + col];
        out[row * N + col] = acc + residual[row * N + col];
    }
}
