#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

extern __global__ void DotProduct(float* A, float* B, float* C, float size);

void runKernel() {
    const int size = 8;
    float A[size], B[size], C[1];

    for (int i = 0; i < size; i++) {
        A[i] = i;
    }

    for (int j = 0; j < size; j++) {
        B[j] = j;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, sizeof(float));

    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = size;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int shared_size = threadsPerBlock * sizeof(float);

    DotProduct<<<blocksPerGrid, threadsPerBlock, shared_size>>>(d_A, d_B, d_C, size);

    cudaMemcpy(C, d_C, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    int expected_dot_product = 0;
    for (int k = 0; k < size; k++) {
        expected_dot_product += A[k] * B[k];
    }
    assert(C[0] == expected_dot_product);

    std::cout << "Dot product successful!" << std::endl;
}

int main() {
    runKernel();
    return 0;
}