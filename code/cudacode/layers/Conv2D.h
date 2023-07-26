#ifndef _Conv2D_LAYER_H_
#define _Conv2D_LAYER_H_

#include <string>

#include <cublas_v2.h>
#include <cudnn.h>

#include "Layer.h"

namespace CUDA_NAMESPACE {

class Conv2D : public Layer {
 public:
  Conv2D(std::string name, int out_channels, int kernel_size, int stride = 1, int padding = 0, int dilation = 1);
  virtual ~Conv2D();

  virtual int forward(CudaContext &context, CuTensor *input, CuTensor *output) override;

  virtual int set_input_shape(const std::vector<int> &inputShape) override;

 private:
  void fwd_initialize(const std::vector<int> &inputShape);
  virtual int load_parameter() {
    return 0;
  }
  virtual int save_parameter() {
    return 0;
  }

  std::unique_ptr<CuTensor> weights_; /* w */
  std::unique_ptr<CuTensor> biases_;  /* b */
  CuTensor *weights_ptr_;
  CuTensor *biases_ptr_;

  int out_channels_;
  int kernel_size_;
  int stride_;
  int padding_;
  int dilation_;
  // convolution
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionFwdAlgo_t conv_fwd_algo_;
  

  size_t workspace_size_ = 0;
  void **d_workspace_ = nullptr;
  void set_workspace(CudaContext &context,CuTensor *input, CuTensor *output);
};

}  // namespace CUDA_NAMESPACE

#endif  // _LAYER_H_