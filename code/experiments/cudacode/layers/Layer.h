#ifndef _Layer_H_
#define _Layer_H_

#include <string>

#include "base/NDTensor.h"
#include "base/cuhelper.h"

namespace CUDA_NAMESPACE {

// 各种层的接口
// Layer设计原则：
// 1. Layer只持有本身权重，不持有输入和输出Tensor
// 2. Layer中forward是线程安全的
class Layer {
 public:
  Layer();
  virtual ~Layer();

  virtual int forward(CudaContext &context,NDTensor *input,NDTensor *output) = 0;
  // 设置输入维度，计算输出维度
  virtual int set_input_shape(const std::vector<int> &shape) = 0;
  virtual int load_parameter(const std::vector<char> &buff) = 0;
  virtual int save_parameter() = 0;
  // 申请前向推理需要的内存空间(不包含输入和输出Tensor)
  virtual void fwd_initialize(const std::vector<int> &inputShapes) = 0;
  std::vector<int> get_output_shape();
  std::vector<int> get_input_shape();
  std::string get_name();

  /* Weight Freeze or Unfreeze */
  void freeze() {
    freeze_ = true;
  }
  void unfreeze() {
    freeze_ = false;
  }

 protected:
  // name of Layer
  std::string name_;
  bool freeze_; /* control parameter updates */
  int batch_size_;  // mini-batch size
  std::vector<int> input_shapes_;
  std::vector<int> output_shapes_;
  // int input_num_;
  // int output_num_;
  TensorDescriptorRAII input_desc_raii_;
  TensorDescriptorRAII output_desc_raii_;
};

}  // namespace CUDA_NAMESPACE

#endif  // _Layer_H_