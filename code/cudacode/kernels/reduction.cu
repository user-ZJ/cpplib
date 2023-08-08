#include "reduction.h"
#include <stdio.h>
#include <stdlib.h>


// 使用global memory进行reduction sum
__global__ void global_reduction_kernel(float *data_out, float *data_in, int stride, int size) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_x + stride < size) { data_out[idx_x] += data_in[idx_x + stride]; }
}

void global_reduction(float *d_out, float *d_in, int n_threads, int size) {
  int n_blocks = (size + n_threads - 1) / n_threads;
  for (int stride = 1; stride < size; stride *= 2) {
    global_reduction_kernel<<<n_blocks, n_threads>>>(d_out, d_in, stride, size);
  }
}

// 使用shared memory进行reduction sum
__global__ void reduction_kernel(float *d_out, float *d_in, unsigned int size) {
  unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  // 申请共享内存，每个block在一个SM执行，有单独的共享内存
  extern __shared__ float s_data[];
  s_data[threadIdx.x] = (idx_x < size) ? d_in[idx_x] : 0.f;
  __syncthreads();

  // 对每个block进行do reduction
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    // thread synchronous reduction
    // 功能同idx_x % (stride * 2)，但速度更快
    if ((idx_x & (stride * 2)) == 0) s_data[threadIdx.x] += s_data[threadIdx.x + stride];
    // 同步block中所有线程
    __syncthreads();
  }
  // 将每个block的reduce结果进行gather
  if (threadIdx.x == 0) d_out[blockIdx.x] = s_data[0];
}

void reduction(float *d_out, float *d_in, int n_threads, int size) {
  cudaMemcpy(d_out, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice);
  while (size > 1) {
    // 将数据分成n_blocks块，每块分配给一个block执行
    int n_blocks = (size + n_threads - 1) / n_threads;
    reduction_kernel<<<n_blocks, n_threads, n_threads * sizeof(float), 0>>>(d_out, d_out, size);
    size = n_blocks;
  }
}