#include <cuda_runtime.h>
#include <stdio.h>

__global__ void activeMaskExample() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int active_mask = __activemask();

    if (active_mask & (1 << threadIdx.x)) {
        printf("Thread %d is active.\n", tid);
    }
}

int main() {
    int numBlocks = 1;
    int threadsPerBlock = 32;
    activeMaskExample<<<numBlocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();
    return 0;
}
