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



__global__ void
reduction_kernel_sequential(float *g_out, float *g_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_data[];

    s_data[threadIdx.x] = (idx_x < size) ? g_in[idx_x] : 0.f;

    __syncthreads();

    // do reduction
    // sequential addressing
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];

        __syncthreads();
    }

    if (threadIdx.x == 0)
        g_out[blockIdx.x] = s_data[0];
}

int reduction_sequential(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{
    int n_blocks = (size + n_threads - 1) / n_threads;
    reduction_kernel_sequential<<< n_blocks, n_threads, n_threads * sizeof(float), 0 >>>(g_outPtr, g_inPtr, size);
    return n_blocks;
}

__global__ void
reduction_kernel_grid(float *g_out, float *g_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_data[];

    // 不同的grid之间进行reduce
    // cumulates input with grid-stride loop and save to share memory
    float input = 0.f;
    for (int i = idx_x; i < size; i += blockDim.x * gridDim.x)
        input += g_in[i];
    s_data[threadIdx.x] = input;

    __syncthreads();

    // grid中的block进行reduce
    // do reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride) 
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        g_out[blockIdx.x] = s_data[0];
    }
}

int reduction_grid(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{
    int num_sms;
    int num_blocks_per_sm;
    // 查询设备的多处理器数量
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    // 查询每个SM可同时执行的最大block数
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_kernel_grid, n_threads, n_threads*sizeof(float));
    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);

    reduction_kernel_grid<<<n_blocks, n_threads, n_threads * sizeof(float), 0>>>(g_outPtr, g_inPtr, size);
    reduction_kernel_grid<<<1, n_threads, n_threads * sizeof(float), 0>>>(g_outPtr, g_inPtr, n_blocks);

    return 1;
}


#define NUM_LOAD 4
__global__ void
reduction_kernel_register(float *g_out, float *g_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_data[];

    // cumulates input with grid-stride loop and save to share memory
    float input[NUM_LOAD] = {0.f};
    for (int i = idx_x; i < size; i += blockDim.x * gridDim.x * NUM_LOAD)
    {
        for (int step = 0; step < NUM_LOAD; step++)
            input[step] += (i + step * blockDim.x * gridDim.x < size) ? g_in[i + step * blockDim.x * gridDim.x] : 0.f;
    }
    for (int i = 1; i < NUM_LOAD; i++)
        input[0] += input[i];
    s_data[threadIdx.x] = input[0];

    __syncthreads();

    // do reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride) 
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        g_out[blockIdx.x] = s_data[0];
    }
}

int reduction_register(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{
    int num_sms;
    int num_blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_kernel_register, n_threads, n_threads*sizeof(float));
    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);

    reduction_kernel_register<<<n_blocks, n_threads, n_threads * sizeof(float), 0>>>(g_outPtr, g_inPtr, size);
    reduction_kernel_register<<<1, n_threads, n_threads * sizeof(float), 0>>>(g_outPtr, g_inPtr, n_blocks);

    return 1;
}