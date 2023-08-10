#include <random>

#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>

#include "Pooling.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace CUDA_NAMESPACE {

/****************************************************************
 * Pooling Layer                                             *
 ****************************************************************/

Pooling::Pooling(std::string name, int kernel_size, int padding, int stride, cudnnPoolingMode_t mode) :
  kernel_size_(kernel_size), padding_(padding), stride_(stride), mode_(mode) {
  name_=name;
  cudnnCreatePoolingDescriptor(&pool_desc_);
  cudnnSetPooling2dDescriptor(pool_desc_, mode_, CUDNN_PROPAGATE_NAN, kernel_size_, kernel_size_, padding_, padding_,
                              stride_, stride_);
}

Pooling::~Pooling() {
  cudnnDestroyPoolingDescriptor(pool_desc_);
}

void Pooling::fwd_initialize(const std::vector<int> &inputShape) {}

int Pooling::forward(CudaContext &context, NDTensor *input, NDTensor *output) {
  //   LOG(INFO) << "Pooling::forward\n";
  auto input_desc = createTensorDesc(input->shapes());
  auto output_desc = createTensorDesc(output->shapes());
  cudnnPoolingForward(context.cudnn(), pool_desc_,
		&context.one,   input_desc.tensor_desc_,  input->data<float>(),
		&context.zero,  output_desc.tensor_desc_, output->data<float>());
  return 0;
}

int Pooling::set_input_shape(const std::vector<int> &inputShape) {
  batch_size_ = inputShape[0];
  input_shapes_ = inputShape;
  cudnnSetTensor4dDescriptor(input_desc_raii_.tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_shapes_[0],
                             input_shapes_[1], input_shapes_[2], input_shapes_[3]);
  output_shapes_.resize(inputShape.size());
  cudnnGetPooling2dForwardOutputDim(pool_desc_, input_desc_raii_.tensor_desc_, 
			&output_shapes_[0], &output_shapes_[1], &output_shapes_[2], &output_shapes_[3]);
  cudnnSetTensor4dDescriptor(output_desc_raii_.tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_shapes_[0],
                             output_shapes_[1], output_shapes_[2], output_shapes_[3]);
  return 0;
}

}  // namespace CUDA_NAMESPACE
