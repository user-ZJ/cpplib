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

  virtual int forward(CudaContext &context,NDTensor *input,NDTensor *output) override;
  virtual int set_input_shape(const std::vector<int> &inputShape) override;
  virtual int load_parameter(const std::vector<char> &buff);
  virtual int save_parameter();

 private:
  virtual void fwd_initialize(const std::vector<int> &inputShape) override;

  int input_size_ = 0;  // 输入维度
  int output_size_ = 0;  // 输出维度
  std::unique_ptr<NDTensor> weights_; /* w */
  std::unique_ptr<NDTensor> biases_;  /* b */
  NDTensor *weights_ptr_;
  NDTensor *biases_ptr_;
  
  float *d_one_vec = nullptr;
};


}  // namespace DMAI

#endif  // _LAYER_H_