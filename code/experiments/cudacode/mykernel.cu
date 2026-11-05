#include "stdio.h"

__global__ void convolution(float* input, float* kernel, float* output, int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int k_radius = kernel_size / 2;
    float sum = 0.0;
    for (int i = -k_radius; i <= k_radius; i++) {
        for (int j = -k_radius; j <= k_radius; j++) {
            int cur_x = x + i;
            int cur_y = y + j;
            if (cur_x < 0 || cur_x >= width || cur_y < 0 || cur_y >= height) {
                continue;
            }
            float val = input[cur_y * width + cur_x];
            float kernel_val = kernel[(j + k_radius) * kernel_size + (i + k_radius)];
            sum += val * kernel_val;
        }
    }
    output[y * width + x] = sum;
}

__global__ void helloCUDA(float f){
    printf ("Hello thread %d, f=%f\n", threadIdx.x, f);
}