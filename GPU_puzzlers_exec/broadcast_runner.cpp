#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

extern __global__ void Broadcast(float* A, float* B, float* C, int size);

void runKernel() {
    const int size = 4;
    float A[size][1], B[1][size], C[size][size];

    for (int i = 0; i < size; i++) {
        A[i][0] = static_cast<float>(i);
    }

    for (int j = 0; j < size; j++) {
        B[0][j] = static_cast<float>(j);
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, (size * size) * sizeof(float));

    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(size, size);

    Broadcast<<<1, blockDim>>>(d_A, d_B, d_C, size);

    cudaMemcpy(C, d_C, (size * size) * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
        assert(C[i][j] == A[i][0] + B[0][j]);
        }
    }

    std::cout << "Broadcast successful" << std::endl;
    return 0;
}

int main() {
    runKernel();
    return 0;
}