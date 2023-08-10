#include "Dropout.h"

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
 * dropout Layer                                             *
 ****************************************************************/

Dropout::Dropout(const std::string name, float ratio) {
  name_ = name;
  dropRatio = ratio;
  cudnnCreateDropoutDescriptor(&dropout_descriptor);
}

Dropout::~Dropout() {
  cudnnDestroyDropoutDescriptor(dropout_descriptor);
  if(tensor_desc_) cudnnDestroyTensorDescriptor(tensor_desc_);
}

void Dropout::fwd_initialize(const std::vector<int> &inputShapes) {
  if (inputShapes.size() <= 4) {
    std::array<int, 4> nchw{1, 1, 1, 1};
    for (int i = 0; i < inputShapes.size(); i++)
      nchw[i] = inputShapes[i];
    cudnnCreateTensorDescriptor(&tensor_desc_);
    cudnnSetTensor4dDescriptor(tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, nchw[0], nchw[1], nchw[2], nchw[3]);
  }

  
}

int Dropout::forward(CudaContext &context,NDTensor *input,NDTensor *output) {
  //   LOG(INFO) << "Dropout::forward\n";
  cudnnDropoutGetStatesSize(context.cudnn(), &dropout_state_size);
  cudnnDropoutGetReserveSpaceSize(tensor_desc_, &dropout_reserve_size);
  // Allocate memory for states and reserve space
  cudaMalloc(&states, dropout_state_size);
  cudaMalloc(&dropout_reserve_space, dropout_reserve_size);
  cudnnSetDropoutDescriptor(dropout_descriptor, context.cudnn(), dropRatio, states, dropout_state_size,
                            /*Seed*/ time(NULL));
  auto input_desc =createTensorDesc(input->shapes());
  auto output_desc = createTensorDesc(output->shapes());
  cudnnDropoutForward(context.cudnn(), dropout_descriptor, input_desc.tensor_desc_, input->data<float>(), output_desc.tensor_desc_, output->data<float>(),
                      dropout_reserve_space, dropout_reserve_size);
  return 0;
}

int Dropout::set_input_shape(const std::vector<int> &inputShapes) {
  batch_size_ = inputShapes[0];
  input_shapes_ = inputShapes;
  output_shapes_ = inputShapes;
  fwd_initialize(input_shapes_);
  return 0;
}

int Dropout::load_parameter() {
  return 0;
}
int Dropout::save_parameter() {
  return 0;
}

/****************************************************************
 * Layer definition                                             *
 ****************************************************************/

}  // namespace CUDA_NAMESPACE
