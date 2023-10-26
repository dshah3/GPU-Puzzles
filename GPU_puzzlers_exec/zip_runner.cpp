#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

extern __global__ void VecAdd(float* A, float* B, float* C);

void runKernel() {
    const int N = 3;
    float A[N], B[N], C[N];

    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeof(float) * N);
    cudaMalloc(&d_B, sizeof(float) * N);
    cudaMalloc(&d_C, sizeof(float) * N);

    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    VecAdd<<<1, N>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    for (int i = 0; i < N; i++) {
      assert(C[i] == A[i] + B[i]);
    }

    std::cout << "Vector addition successful!" << std::endl;

    return 0;
}

int main() {
    runKernel();
    return 0;
}