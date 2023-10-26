#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

extern __global__ void Map2DBlock(float* A, float* C, float size);

void runKernel() {
    const int size = 6;
    float A[size][size], C[size][size];

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
        A[i][j] = static_cast<float>(i) + static_cast<float>(j);
        }
    }

    float *d_A, *d_C;

    cudaMalloc(&d_A, (size * size) * sizeof(float));
    cudaMalloc(&d_C, (size * size) * sizeof(float));

    dim3 threadsPerBlock(size - 1, size - 1);
    dim3 blocksPerGrid(((size + threadsPerBlock.x - 1) / threadsPerBlock.x),
                        (((size + threadsPerBlock.y - 1) / threadsPerBlock.y)));

    cudaMemcpy(d_A, A, (size * size) * sizeof(float), cudaMemcpyHostToDevice);

    Map2DBlock<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, size);

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