#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

extern __global__ void ScalarAdd(float* A, float* C);

void runKernel() {
    const int N = 3;
    float A[N], C[N];

    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>(i);
    }

    float *d_A, *d_C;

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);

    ScalarAdd<<<1, N>>>(d_A, d_C);

    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);

    for (int i = 0; i < N; i++) {
        assert(C[i] == A[i] + 10);
    }

    std::cout << "Scalar addition is successful!" << std::endl;

    return 0;
}

int main() {
    runKernel();
    return 0;
}