


__global__ void init_one_vec(float *d_one_vec, size_t length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= length) return;

  d_one_vec[i] = 1.f;
}