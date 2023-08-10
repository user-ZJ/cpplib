#ifndef _Pooling_LAYER_H_
#define _Pooling_LAYER_H_

#include <string>

#include <cublas_v2.h>
#include <cudnn.h>

#include "Layer.h"

namespace CUDA_NAMESPACE {

class Pooling : public Layer {
 public:
  Pooling(std::string name, int kernel_size, int padding, int stride, cudnnPoolingMode_t mode);
  virtual ~Pooling();

  virtual int forward(CudaContext &context, NDTensor *input, NDTensor *output) override;

  virtual int set_input_shape(const std::vector<int> &inputShape) override;

 private:
  void fwd_initialize(const std::vector<int> &inputShape);
  virtual int load_parameter() {
    return 0;
  }
  virtual int save_parameter() {
    return 0;
  }

  int kernel_size_;
  int padding_;
  int stride_;
  cudnnPoolingMode_t mode_;
  cudnnPoolingDescriptor_t pool_desc_;
};

}  // namespace CUDA_NAMESPACE

#endif  // _LAYER_H_