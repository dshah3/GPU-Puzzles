#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

extern __global__ void AxisSum(float* A, float* C, int size);

void runKernel() {
    const int size = 5;
    const int numBatches = 1;

    float A[size * numBatches], C[numBatches];

    for (int j = 0; j < numBatches; j++) {
        for (int i = 0; i < size; i++) {
        A[j * size + i] = i;
        }
    }

    float *d_A, *d_C;

    cudaMalloc(&d_A, size * numBatches * sizeof(float));
    cudaMalloc(&d_C, numBatches * sizeof(float));

    cudaMemcpy(d_A, A, size * numBatches * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(size, 1);
    dim3 blocksPerGrid(1, numBatches);
    int shared_size = threadsPerBlock.x * sizeof(float);

    AxisSum<<<blocksPerGrid, threadsPerBlock, shared_size>>>(d_A, d_C, size);

    cudaMemcpy(C, d_C, numBatches * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);

    assert(C[0] == 10);

    std::cout << "Axis sum successful!" << std::endl;
}

int main() {
    runKernel();
    return 0;
}