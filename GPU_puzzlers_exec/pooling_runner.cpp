#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

extern __global__ void Pooling(float* A, float* C, float size);

void runKernel() {
    const int size = 4;
    float A[size], C[size];

    for (int i = 0; i < size; i++) {
        A[i] = static_cast<float>(i);
    }

    float *d_A, *d_C;

    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));

    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = size;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int shared_size = threadsPerBlock * sizeof(float);

    Pooling<<<blocksPerGrid, threadsPerBlock, shared_size>>>(d_A, d_C, size);

    cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);

    for (int i = 0; i < size; i++) {
        if (i >= 2) {
            assert(C[i] == A[i] + A[i-1] + A[i-2]);
        }
    }

    std::cout << "Pooling successful!" << std::endl;
    return 0;
}

int main() {
    runKernel();
    return 0;
}