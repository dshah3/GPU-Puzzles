#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

extern __global__ void Guards(float* A, float* C, int size);

void runKernel() {
    const int size = 3;
    float A[size], C[size];

    for (int i = 0; i < size; i++) {
        A[i] = static_cast<float>(i);
    }

    float *d_A, *d_C;

    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));

    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);

    Guards<<<1, 10>>>(d_A, d_C, size);

    cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);

    for (int i = 0; i < size; i++){
        assert(C[i] == A[i] + 10);
    }

    std::cout << "Guards successful!" << std::endl;
}

int main() {
    runKernel();
    return 0;
}
