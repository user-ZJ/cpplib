#include <stdio.h>
#include <stdlib.h>

// #define N 2048
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

__global__ void matrix_transpose_naive(float *input, float *output, int M, int N) {
  int indexX = threadIdx.x + blockIdx.x * blockDim.x;
  int indexY = threadIdx.y + blockIdx.y * blockDim.y;
  int index = indexY * M + indexX;
  int transposedIndex = indexX * N + indexY;
  if (indexX >= M || indexY >= N) return;

  // this has discoalesced global memory store
  output[transposedIndex] = input[index];

  // this has discoalesced global memore load
  // output[index] = input[transposedIndex];
}

__global__ void matrix_transpose_shared(float *input, float *output, int M, int N) {
  // 使用BLOCK_SIZE + 1，解决wrap的bank冲突
  __shared__ int sharedMemory[BLOCK_SIZE][BLOCK_SIZE + 1];

  // global index
  int indexX = threadIdx.x + blockIdx.x * blockDim.x;
  int indexY = threadIdx.y + blockIdx.y * blockDim.y;
  if (indexX >= M || indexY >= N) return;

  // local index
  int localIndexX = threadIdx.x;
  int localIndexY = threadIdx.y;

  int index = indexY + indexX * N;
  int transposedIndex = indexY*M + indexX;
//   int transposedIndex = tindexY*M + tindexX;

  // reading from global memory in coalesed manner and performing tanspose in shared memory
  sharedMemory[localIndexX][localIndexY] = input[index];
//   printf("%d,%d,%d,%d,%d,%d\n", indexX, indexY,localIndexX, localIndexY, index,
//          transposedIndex);

  __syncthreads();

  // writing into global memory in coalesed fashion via transposed data in shared memory
  output[transposedIndex] = sharedMemory[localIndexX][localIndexY];
}

int matrix_transpose(float *d_in, float *d_out, int M, int N) {
  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

  matrix_transpose_shared<<<gridSize, blockSize>>>(d_in, d_out, M, N);
  return 0;
}

int global_matrix_transpose(float *d_in, float *d_out, int M, int N) {
  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

  matrix_transpose_naive<<<gridSize, blockSize>>>(d_in, d_out, M, N);
  return 0;
}
