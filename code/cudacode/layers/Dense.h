#ifndef _DENSE_LAYER_H_
#define _DENSE_LAYER_H_

#include <string>

#include <cublas_v2.h>
#include <cudnn.h>
#include "Layer.h"

namespace CUDA_NAMESPACE {


class Dense : public Layer {
 public:
  Dense(std::string name, int in_size, int out_size);
  virtual ~Dense();

  virtual int forward(CudaContext &context,CuTensor *input,CuTensor *output) override;
  virtual int set_input_shape(const std::vector<int> &inputShape) override;

 private:
  virtual void fwd_initialize(const std::vector<int> &inputShape) override;

  int input_size_ = 0;  // 输入维度
  int output_size_ = 0;  // 输出维度
  std::unique_ptr<CuTensor> weights_; /* w */
  std::unique_ptr<CuTensor> biases_;  /* b */
  CuTensor *weights_ptr_;
  CuTensor *biases_ptr_;
  virtual int load_parameter();
  virtual int save_parameter();
  float *d_one_vec = nullptr;
};


}  // namespace DMAI

#endif  // _LAYER_H_