__global__ void __KERNEL_NAME__(
    const float* __restrict__ A,
    const float* __restrict__ Wq,
    const float* __restrict__ Wk,
    const float* __restrict__ Wv,
    float* __restrict__ Q,
    float* __restrict__ K_out,
    float* __restrict__ V_out,
    int M, int K,
    int Nq, int Nk, int Nv
) {
    int which = blockIdx.z;  // 0=Q, 1=K, 2=V
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    const float* W;
    float* out;
    int N;

    if (which == 0) { W = Wq; out = Q;     N = Nq; }
    else if (which == 1) { W = Wk; out = K_out; N = Nk; }
    else { W = Wv; out = V_out; N = Nv; }

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += A[row * K + k] * W[k * N + col];
        out[row * N + col] = acc;
    }
}
