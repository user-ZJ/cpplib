#include "Dense.h"

#include <random>

#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>

#include <fstream>
#include <iostream>
#include <sstream>

namespace CUDA_NAMESPACE {

/****************************************************************
 * Dense Layer                                                  *
 ****************************************************************/

__global__ void init_one_vec(float *d_one_vec, size_t length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= length) return;

  d_one_vec[i] = 1.f;
}

Dense::Dense(std::string name, int input_size, int output_size) {
  name_ = name;
  input_size_ = input_size;
  output_size_ = output_size;
  // initialize weight, bias, and output
  weights_ptr_ = new CuTensor({1, 1, input_size_, output_size_});
  biases_ptr_ = new CuTensor({1, 1, output_size_});
  weights_.reset(weights_ptr_);
  biases_.reset(biases_ptr_);
  
}

Dense::~Dense() {
  if (d_one_vec != nullptr) {
    cudaFree(d_one_vec);
    d_one_vec = nullptr;
  }
}

int Dense::load_parameter() {
  std::stringstream filename_weights, filename_biases;

  // load weights and biases pretrained parameters
  filename_weights << name_ << ".bin";
  CTensorfi weight(weights_->shapes()), bias(biases_->shapes());
  if (weight.readFile(filename_weights.str().c_str())) return -1;

  filename_biases << name_ << ".bias.bin";
  if (bias.readFile(filename_biases.str().c_str())) return -2;

  weights_->fromcpu(weight);
  biases_->fromcpu(bias);

  std::cout << ".. loaded " << name_ << " pretrain parameter.." << std::endl;

  return 0;
}

int Dense::save_parameter() {
  std::stringstream filename_weights, filename_biases;

  std::cout << ".. saving " << name_ << " parameter ..";

  // Write weights file
  filename_weights << name_ << ".bin";
  if (weights_->cpu().writeFile(filename_weights.str().c_str())) return -1;

  // Write bias file

  filename_biases << name_ << ".bias.bin";
  if (biases_->cpu().writeFile(filename_biases.str().c_str())) return -2;

  std::cout << " done .." << std::endl;

  return 0;
}

void Dense::fwd_initialize(const std::vector<int> &inputShape) {
  batch_size_ = inputShape[0];
  if (d_one_vec != nullptr) cudaFree(d_one_vec);
  checkCudaErrors(cudaMalloc((void **)&d_one_vec, sizeof(float) * batch_size_));
  init_one_vec<<<(batch_size_ + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D, BLOCK_DIM_1D>>>(d_one_vec, batch_size_);
}

int Dense::forward(CudaContext &context, CuTensor *input, CuTensor *output) {
  //   LOG(INFO) << "Dense::forward\n";
  // output = weights^T * input (without biases)
  checkCublasErrors(cublasSgemm(context.cublas(), CUBLAS_OP_T, CUBLAS_OP_N, output_size_, batch_size_, input_size_,
                                &context.one, (float *)weights_->data(), input_size_, (float *)input->data(), input_size_, &context.zero,
                                (float *)output->data(), output_size_));
  //   checkCudaErrors(cudaDeviceSynchronize());
  // output += biases * d_one_vec^T
  checkCublasErrors(cublasSgemm(context.cublas(), CUBLAS_OP_N, CUBLAS_OP_N, output_size_, batch_size_, 1, &context.one,
                                (float *)biases_->data(), output_size_, d_one_vec, 1, &context.one,(float *)output->data(),
                                output_size_));
  //   checkCudaErrors(cudaDeviceSynchronize());
  return 0;
}

int Dense::set_input_shape(const std::vector<int> &inputShape) {
  batch_size_ = inputShape[0];
  input_shapes_ = inputShape;
  output_shapes_ = std::vector<int>{batch_size_, output_size_};
  fwd_initialize(inputShape);
  return 0;
}

}  // namespace CUDA_NAMESPACE
