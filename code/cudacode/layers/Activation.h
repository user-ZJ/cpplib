#ifndef _ACTIVATION_LAYER_H_
#define _ACTIVATION_LAYER_H_

#include <string>

#include <cublas_v2.h>
#include <cudnn.h>

#include "Layer.h"

namespace CUDA_NAMESPACE {


class Activation : public Layer {
 public:
  Activation(std::string name, cudnnActivationMode_t mode, float coef = 0.f);
  virtual ~Activation();

  virtual int forward(CudaContext &context,NDTensor *input,NDTensor *output) override;

  virtual int set_input_shape(const std::vector<int> &inputShape) override;

 private:
  void fwd_initialize(const std::vector<int> &inputShape);
  virtual int load_parameter(const std::vector<char> &buff) {
    return 0;
  }
  virtual int save_parameter() {
    return 0;
  }

  cudnnActivationDescriptor_t act_desc_;
  cudnnActivationMode_t act_mode_;
  float act_coef_;
};



}  // namespace DMAI

#endif  // _LAYER_H_