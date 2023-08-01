#include <random>

#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>

#include "Conv2D.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace CUDA_NAMESPACE {

/****************************************************************
 * Conv2D Layer                                             *
 ****************************************************************/

Conv2D::Conv2D(std::string name, int out_channels, int kernel_size, int stride, int padding, int dilation) :
  out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding), dilation_(dilation) {
  name_ = name;
  // create cudnn container handles
  cudnnCreateFilterDescriptor(&filter_desc_);
  cudnnCreateConvolutionDescriptor(&conv_desc_);
  checkCudnnErrors(cudnnSetConvolution2dDescriptor(conv_desc_, padding_, padding_, stride_, stride_, dilation_,
                                                   dilation_, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  // setting cudnn convolution math type
  // CUDNN_DEFAULT_MATH operates convolution with FP32.
  // If you use A100, CUDNN utilise tensor cores with TF32.
  checkCudnnErrors(cudnnSetConvolutionMathType(conv_desc_, CUDNN_DEFAULT_MATH));
}

Conv2D::~Conv2D() {
  // distroy cudnn container resources
  cudnnDestroyFilterDescriptor(filter_desc_);
  cudnnDestroyConvolutionDescriptor(conv_desc_);
}

CuTensor Conv2D::set_workspace(CudaContext &context, CuTensor *input, CuTensor *output) {
  auto input_desc_ = input->tensor_desc();
  auto output_desc_ = output->tensor_desc();
  size_t temp_size = 0;
  size_t workspace_size=0;

  // forward
  std::vector<cudnnConvolutionFwdAlgoPerf_t> fwd_algo_perf_results(CUDNN_CONVOLUTION_FWD_ALGO_COUNT);

  int algo_max_count;
  int returnedAlgoCount = 0;
  checkCudnnErrors(cudnnGetConvolutionForwardAlgorithmMaxCount(context.cudnn(), &algo_max_count));
  checkCudnnErrors(cudnnGetConvolutionForwardAlgorithm_v7(context.cudnn(), input_desc_, filter_desc_, conv_desc_,
                                                          output_desc_, algo_max_count, &returnedAlgoCount,
                                                          &fwd_algo_perf_results[0]));
  // shoose the fastest algorithm
  conv_fwd_algo_ = fwd_algo_perf_results[0].algo;

  checkCudnnErrors(cudnnGetConvolutionForwardWorkspaceSize(context.cudnn(), input_desc_, filter_desc_, conv_desc_,
                                                           output_desc_, conv_fwd_algo_, &temp_size));
  workspace_size = std::max(workspace_size, temp_size);

  return CuTensor({1,(int)workspace_size},DataType::INT8);
}

void Conv2D::fwd_initialize(const std::vector<int> &inputShape) {
  int channel = inputShape[1];
  // initialize weights and bias
  if (weights_ == nullptr) {
    // initialize containers handles
    checkCudnnErrors(cudnnSetFilter4dDescriptor(filter_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels_,
                                                channel, kernel_size_, kernel_size_));

    weights_ptr_ = new CuTensor({out_channels_, channel, kernel_size_, kernel_size_});
    biases_ptr_ = new CuTensor({1, out_channels_});  // bias size
    weights_.reset(weights_ptr_);
    biases_.reset(biases_ptr_);
  }
}

int Conv2D::forward(CudaContext &context, CuTensor *input, CuTensor *output) {
  //   LOG(INFO) << "Conv2D::forward\n";
  CuTensor workspace = set_workspace(context,input,output);
  auto bias_desc_ = biases_->tensor_desc();
  auto input_desc_ = input->tensor_desc();
  auto output_desc_ = output->tensor_desc();
  checkCudnnErrors(cudnnConvolutionForward(context.cudnn(), &context.one, input_desc_, input->data(), filter_desc_,
                                           weights_->data(), conv_desc_, conv_fwd_algo_, workspace.data(), workspace.byteSize(),
                                           &context.zero, output_desc_, output->data()));

  checkCudnnErrors(cudnnAddTensor(context.cudnn(), &context.one, bias_desc_, biases_->data(), &context.one,
                                  output_desc_, output->data()));
  return 0;
}

int Conv2D::set_input_shape(const std::vector<int> &inputShape) {
  batch_size_ = inputShape[0];
  input_shapes_ = inputShape;
  cudnnSetTensor4dDescriptor(input_desc_raii_.tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_shapes_[0],
                             input_shapes_[1], input_shapes_[2], input_shapes_[3]);
  output_shapes_.resize(inputShape.size());
  checkCudnnErrors(cudnnGetConvolution2dForwardOutputDim(conv_desc_, input_desc_raii_.tensor_desc_, filter_desc_,
                                                         &output_shapes_[0], &output_shapes_[1], &output_shapes_[2],
                                                         &output_shapes_[3]));
  cudnnSetTensor4dDescriptor(output_desc_raii_.tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_shapes_[0],
                             output_shapes_[1], output_shapes_[2], output_shapes_[3]);
  return 0;
}

}  // namespace CUDA_NAMESPACE
