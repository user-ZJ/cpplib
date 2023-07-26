#ifndef _Softmax_LAYER_H_
#define _Softmax_LAYER_H_

#include <string>

#include <cublas_v2.h>
#include <cudnn.h>

#include "Layer.h"

namespace CUDA_NAMESPACE {


class Softmax : public Layer {
 public:
  Softmax(std::string name);
  virtual ~Softmax();

  virtual int forward(CudaContext &context,CuTensor *input,CuTensor *output) override;

  virtual int set_input_shape(const std::vector<int> &inputShape) override;

 private:
  void fwd_initialize(const std::vector<int> &inputShape);
  virtual int load_parameter() {
    return 0;
  }
  virtual int save_parameter() {
    return 0;
  }
};



}  // namespace DMAI

#endif  // _LAYER_H_