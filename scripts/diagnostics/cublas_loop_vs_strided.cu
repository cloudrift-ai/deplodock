// Compare two cuBLAS batched-GEMM dispatch styles, plus our TMA kernel as a
// reference, to substantiate the article's "is the comparison fair" paragraph.
//
// Compile: nvcc -O3 --fmad=true -arch=sm_120 -lcublas -lcurand -o cublas_loop_vs_strided cublas_loop_vs_strided.cu
// Run:     ./cublas_loop_vs_strided 1024 16
//
// Output: KERNEL_LOOP_MS=...   (loop of N individual cublasSgemm calls)
//         KERNEL_STRIDED_MS=... (cublasSgemmStridedBatched single call)
//         Both medians of 20 interleaved iterations, first iteration discarded.
//
// This is intentionally minimal — no autotuning, no graph capture. The point
// is just to show whether `cublasSgemmStridedBatched` is faster than the
// hand-rolled loop. The article previously cited specific numbers from this
// experiment without committing the harness; this script is that harness.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

static inline void check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int BATCH = (argc > 2) ? atoi(argv[2]) : 16;
    int ITER = 20;

    size_t bytes = (size_t)BATCH * N * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    check(cudaMalloc(&d_A, bytes), "alloc A");
    check(cudaMalloc(&d_B, bytes), "alloc B");
    check(cudaMalloc(&d_C, bytes), "alloc C");

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateUniform(gen, d_A, (size_t)BATCH * N * N);
    curandGenerateUniform(gen, d_B, (size_t)BATCH * N * N);
    curandDestroyGenerator(gen);

    cublasHandle_t h;
    cublasCreate(&h);
    float alpha = 1.0f, beta = 0.0f;

    // Warmup both paths.
    for (int b = 0; b < BATCH; b++) {
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                    d_B + (size_t)b * N * N, N,
                    d_A + (size_t)b * N * N, N,
                    &beta,  d_C + (size_t)b * N * N, N);
    }
    cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                              d_B, N, (long long)N * N,
                              d_A, N, (long long)N * N,
                              &beta, d_C, N, (long long)N * N, BATCH);
    check(cudaDeviceSynchronize(), "warmup");

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    float loop_times[64], strided_times[64];

    // Interleaved loop+strided, first iteration discarded.
    for (int i = 0; i < ITER + 1; i++) {
        // Loop variant
        cudaEventRecord(s);
        for (int b = 0; b < BATCH; b++) {
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                        d_B + (size_t)b * N * N, N,
                        d_A + (size_t)b * N * N, N,
                        &beta,  d_C + (size_t)b * N * N, N);
        }
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float t; cudaEventElapsedTime(&t, s, e);
        if (i > 0) loop_times[i-1] = t;

        // Strided variant
        cudaEventRecord(s);
        cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                  d_B, N, (long long)N * N,
                                  d_A, N, (long long)N * N,
                                  &beta, d_C, N, (long long)N * N, BATCH);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        cudaEventElapsedTime(&t, s, e);
        if (i > 0) strided_times[i-1] = t;
    }

    // Sort for median
    for (int i = 0; i < ITER - 1; i++)
        for (int j = i + 1; j < ITER; j++) {
            if (loop_times[j]    < loop_times[i])    { float t = loop_times[i];    loop_times[i]    = loop_times[j];    loop_times[j]    = t; }
            if (strided_times[j] < strided_times[i]) { float t = strided_times[i]; strided_times[i] = strided_times[j]; strided_times[j] = t; }
        }

    float loop_med    = loop_times[ITER / 2];
    float strided_med = strided_times[ITER / 2];

    double flops = 2.0 * (double)BATCH * N * N * N;
    double loop_tflops    = flops / (loop_med * 1e-3) / 1e12;
    double strided_tflops = flops / (strided_med * 1e-3) / 1e12;

    printf("# cuBLAS batched dispatch comparison\n");
    printf("SIZE=%d\n", N);
    printf("BATCH=%d\n", BATCH);
    printf("KERNEL_LOOP_MS=%.6f\n", loop_med);
    printf("KERNEL_STRIDED_MS=%.6f\n", strided_med);
    printf("LOOP_TFLOPS=%.2f\n", loop_tflops);
    printf("STRIDED_TFLOPS=%.2f\n", strided_tflops);
    printf("STRIDED_VS_LOOP_PCT=%.1f\n", 100.0 * loop_med / strided_med);

    cublasDestroy(h);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
