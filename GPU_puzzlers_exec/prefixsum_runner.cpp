#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

extern __global__ void PrefixSum(float* A, float* C, int size);

void runKernel() {
    const int size = 5;

    float A[size], C[size];

    for (int i = 0; i < size; i++) {
        A[i] = i;
    }

    float *d_A, *d_C;

    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));

    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = size;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int shared_size = threadsPerBlock * sizeof(float);

    PrefixSum<<<blocksPerGrid, threadsPerBlock, shared_size>>>(
        d_A, d_C, size
    );

    cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);

    assert(C[0] == 10);

    std::cout << "Prefix sum successful!" << std::endl;

    return 0;
}

int main() {
    runKernel();
    return 0;
}