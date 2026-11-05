#include <random>

#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include "Softmax.h"

namespace CUDA_NAMESPACE {

/****************************************************************
 * Softmax Layer                                             *
 ****************************************************************/

Softmax::Softmax(std::string name) {
  name_ = name;
}

Softmax::~Softmax() {
}

void Softmax::fwd_initialize(const std::vector<int> &inputShape) {
}

int Softmax::forward(CudaContext &context, NDTensor *input, NDTensor *output) {
  //   LOG(INFO) << "Softmax::forward\n";
  auto input_desc = createTensorDesc(input->shapes());
  auto output_desc = createTensorDesc(output->shapes());
  checkCudnnErrors(
		cudnnSoftmaxForward(context.cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
			&context.one,  input_desc.tensor_desc_,  input->data<float>(),
			&context.zero, output_desc.tensor_desc_, output->data<float>()));
  return 0;
}

int Softmax::set_input_shape(const std::vector<int> &inputShape) {
  batch_size_ = inputShape[0];
  input_shapes_ = inputShape;
  output_shapes_ = inputShape;
  fwd_initialize(inputShape);
  return 0;
}

}  // namespace CUDA_NAMESPACE
