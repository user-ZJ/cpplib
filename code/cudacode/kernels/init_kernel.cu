#include "init_kernel.h"


#ifndef BLOCK_DIM
#define BLOCK_DIM 32
#endif


__global__ void init_vec_kernel(float *d_vec, size_t length,float c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= length) return;
  d_vec[i] = c;
}

void init_vec(float *d_vec, size_t length,float c){
  int n_blocks = (length + BLOCK_DIM - 1) / BLOCK_DIM;
  init_vec_kernel<<<n_blocks, BLOCK_DIM>>>(d_vec, length,c);
}