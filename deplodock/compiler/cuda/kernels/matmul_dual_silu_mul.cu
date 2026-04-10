__global__ void __KERNEL_NAME__(
    const float* __restrict__ A,
    const float* __restrict__ Wg,
    const float* __restrict__ Wu,
    float* __restrict__ out,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = A[row * K + k];
            gate_acc += a * Wg[k * N + col];
            up_acc   += a * Wu[k * N + col];
        }
        // silu(gate) * up — gate and up stay in registers.
        out[row * N + col] = (gate_acc / (1.0f + expf(-gate_acc))) * up_acc;
    }
}
