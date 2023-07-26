#ifndef _DROPOUT_LAYER_H_
#define _DROPOUT_LAYER_H_

#include <string>

#include <cublas_v2.h>
#include <cudnn.h>
#include "Layer.h"

namespace CUDA_NAMESPACE {

class Dropout : public Layer {
 public:
  Dropout(const std::string name, float ratio);
  virtual ~Dropout();
  virtual int forward(CudaContext &context,CuTensor *input,CuTensor *output) override;
  virtual int set_input_shape(const std::vector<int> &inputShape) override;
  virtual void fwd_initialize(const std::vector<int> &inputShape) override;
  virtual int load_parameter() override;
  virtual int save_parameter() override;

 private:
  size_t dropout_state_size;
  size_t dropout_reserve_size;
  void *states;
  void *dropout_reserve_space;
  float dropRatio;
  cudnnDropoutDescriptor_t dropout_descriptor;
  cudnnTensorDescriptor_t tensor_desc_;
};

}  // namespace DMAI

#endif  // _LAYER_H_