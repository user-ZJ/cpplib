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


Dense::Dense(std::string name, int input_size, int output_size):input_size_(input_size),output_size_(output_size) {
  batch_size_ = 1;
  name_ = name;
  // initialize weight, bias, and output
  weights_ptr_ = new NDTensor({input_size_, output_size_},DataType::FLOAT,DeviceType::CUDA);
  biases_ptr_ = new NDTensor({output_size_},DataType::FLOAT,DeviceType::CUDA);
  weights_.reset(weights_ptr_);
  biases_.reset(biases_ptr_);
  
}

Dense::~Dense() {
  if (d_one_vec != nullptr) {
    cudaFree(d_one_vec);
    d_one_vec = nullptr;
  }
}

int Dense::load_parameter(const std::vector<char> &buff) {
  long offset = 0;
  NDTensor weight(weights_->shapes()),biase(biases_->shapes());
  memcpy(weight.data<char>(),buff.data(),weight.byteSize());
  offset += weight.byteSize();
  memcpy(biase.data<char>(),&buff[offset],biase.byteSize());
  

  *weights_ = weight.cuda();
  *biases_ = biase.cuda();
  weights_->cpu().dump2File<float>("weight.txt");
  biases_->cpu().dump2File<float>("biase.txt");
  std::cout << ".. loaded " << name_ << " pretrain parameter.." << std::endl;

  return 0;
}

int Dense::save_parameter() {
  std::stringstream filename_weights, filename_biases;

  std::cout << ".. saving " << name_ << " parameter ..";

  // Write weights file
  filename_weights << name_ << ".bin";
  weights_->cpu();
  if (weights_->writeFile(filename_weights.str().c_str())) return -1;

  // Write bias file

  filename_biases << name_ << ".bias.bin";
  biases_->cpu();
  if (biases_->writeFile(filename_biases.str().c_str())) return -2;

  std::cout << " done .." << std::endl;

  return 0;
}

void Dense::fwd_initialize(const std::vector<int> &inputShape) {
  batch_size_ = inputShape[0];
  if (d_one_vec != nullptr) cudaFree(d_one_vec);
  checkCudaErrors(cudaMalloc((void **)&d_one_vec, sizeof(float) * batch_size_));
  init_one_vec<<<(batch_size_ + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D, BLOCK_DIM_1D>>>(d_one_vec, batch_size_);
}

int Dense::forward(CudaContext &context, NDTensor *input, NDTensor *output) {
  std::cout << "batch_size_:"<<batch_size_<<" input_size:"<<input_size_<<" output_size:"<<output_size_<<std::endl;
  std::cout << "weight_ptr:"<<weights_->data<float>()<<std::endl;
  std::cout << "input:"<<input->data<float>()<<std::endl;
  std::cout << "output:"<<output->data<float>()<<std::endl;
  // weights:output_size x input_size
  // input: input_size x batch_size
  // output: output_size x batch_size
  // output = weights^T * input  (without biases)
  checkCublasErrors(cublasSgemm(context.cublas(), CUBLAS_OP_T, CUBLAS_OP_N, output_size_, batch_size_, input_size_,
                                &context.one, weights_->data<float>(), input_size_, input->data<float>(), batch_size_, &context.zero,
                                output->data<float>(), batch_size_));
  //   checkCudaErrors(cudaDeviceSynchronize());
  // output += biases * d_one_vec^T
  // checkCublasErrors(cublasSgemm(context.cublas(), CUBLAS_OP_N, CUBLAS_OP_N, output_size_, batch_size_, 1, &context.one,
  //                               biases_->data<float>(), 1, d_one_vec, batch_size_, &context.one,output->data<float>(),
  //                               batch_size_));
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
