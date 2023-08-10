#include <random>

#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include "Activation.h"

namespace CUDA_NAMESPACE {

/****************************************************************
 * Activation Layer                                             *
 ****************************************************************/

Activation::Activation(std::string name, cudnnActivationMode_t mode, float coef) {
  name_ = name;
  act_mode_ = mode;
  act_coef_ = coef;

  cudnnCreateActivationDescriptor(&act_desc_);
  cudnnSetActivationDescriptor(act_desc_, act_mode_, CUDNN_PROPAGATE_NAN, act_coef_);
}

Activation::~Activation() {
  cudnnDestroyActivationDescriptor(act_desc_);
}

void Activation::fwd_initialize(const std::vector<int> &inputShape) {
}

int Activation::forward(CudaContext &context, NDTensor *input, NDTensor *output) {
  //   LOG(INFO) << "Activation::forward\n";
  auto input_desc = createTensorDesc(input->shapes());
  auto output_desc = createTensorDesc(output->shapes());
  cudnnActivationForward(context.cudnn(), act_desc_, &context.one, input_desc.tensor_desc_, input->data<float>(), &context.zero,
                         output_desc.tensor_desc_, output->data<float>());
  return 0;
}

int Activation::set_input_shape(const std::vector<int> &inputShape) {
  batch_size_ = inputShape[0];
  input_shapes_ = inputShape;
  output_shapes_ = inputShape;
  fwd_initialize(inputShape);
  return 0;
}

}  // namespace CUDA_NAMESPACE
