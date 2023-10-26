#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

const int TPB = 3;

extern __global__ void Matmul(float* A, float* B, float* C, int size);

void runKernel() {
    const int size = 2;
    float A[size][size], B[size][size], C[size][size];

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
        A[i][j] = i * j;
        B[i][j] = i + j;
        }
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, (size * size) * sizeof(float));
    cudaMalloc(&d_B, (size * size) * sizeof(float));
    cudaMalloc(&d_C, (size * size) * sizeof(float));

    cudaMemcpy(d_A, A, (size * size) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, (size * size) * sizeof(float), cudaMemcpyHostToDevice);

    int BpG = (size + TPB - 1) / TPB;
    dim3 blocksPerGrid(BpG, BpG);
    dim3 threadsPerBlock(TPB, TPB);
    int sharedMemSize = 2 * (TPB * TPB) * sizeof(float);

    Matmul<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_A, d_B, d_C, size);

    cudaMemcpy(C, d_C, (size * size) * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    assert(C[0][0] == 0);
    assert(C[0][1] == 0);
    assert(C[1][0] == 1);
    assert(C[1][1] == 2);

    std::cout << "Matrix multiplication successful!" << std::endl;
}

int main() {
    runKernel();
    return 0;
}