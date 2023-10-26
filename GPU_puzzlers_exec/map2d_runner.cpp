#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

extern __global__ void Map2D(float* A, float* C, float size);

void runKernel() {
    const int size = 4;
    float A[size][size], C[size][size];

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
        A[i][j] = static_cast<float>(i) + static_cast<float>(j);
        }
    }

    float *d_A, *d_C;

    cudaMalloc(&d_A, (size * size) * sizeof(float));
    cudaMalloc(&d_C, (size * size) * sizeof(float));

    dim3 blockDim(size, size);

    cudaMemcpy(d_A, A, (size * size) * sizeof(float), cudaMemcpyHostToDevice);

    Map2D<<<1, blockDim>>>(d_A, d_C, size);

    cudaMemcpy(C, d_C, (size * size) * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
        assert(C[i][j] == A[i][j] + 10);
        }
    }

    std::cout << "2D mapping successful" << std::endl;
    return 0;
}

int main() {
    runKernel();
    return 0;
}