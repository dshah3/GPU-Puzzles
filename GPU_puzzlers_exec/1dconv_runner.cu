#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

extern __global__ void Conv1D(float* A, float* B, float* C, int a_size, int b_size);

const int TPB = 8;
const int MAX_CONV = 4;
const int TPB_MAX_CONV = TPB + MAX_CONV;

void runKernel() {
    const int size = 5;

    float A[size], B[size-2], C[1];

    for (int i = 0; i < size; i++) {
        A[i] = i;
    }

    for (int j = 0; j < size-2; j++) {
        B[j] = j;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, (size - 2) * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));

    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, (size - 2) * sizeof(float), cudaMemcpyHostToDevice);

    int blocksPerGrid = (size + TPB - 1) / TPB;

    int shared_size_a = sizeof(float) * (TPB + MAX_CONV);
    int shared_size_b = sizeof(float) * MAX_CONV;

    Conv1D<<<blocksPerGrid, TPB, shared_size_a + shared_size_b>>>(
        d_A, d_B, d_C, size, (size - 2)
    );

    cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    float host_C[] = {5, 8, 11, 4, 0};

    for (int i = 0; i < size; i++) {
        assert(host_C[i] == C[i]);
    }

    std::cout << "1D Convolution successful!" << std::endl;
}

int main() {
    runKernel();
    return 0;
}