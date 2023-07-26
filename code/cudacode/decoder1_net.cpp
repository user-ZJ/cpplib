#include "decoder1_net.h"

#include "helper.h"
#include "layer.h"

#include "utils/logging.h"
#include "utils/string-util.h"
#include <iomanip>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

namespace CUDA_NAMESPACE {

Decoder1Net::Decoder1Net() {
  model.add_layer(new Dense("dense1",80, 256));
  // model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
  // model.add_layer(new Dropout("dropout", 0.5));
  model.add_layer(new Dense("dense2",256, 256));
  // model.add_layer(new Activation("relu2", CUDNN_ACTIVATION_RELU));
  // model.add_layer(new Dropout("dropout2", 0.5));
  model.cuda();
  model.load_pretrain();
  input_shape_ = std::vector<int>{1,80};
  model.set_input_shape(input_shape_);
  input_ptr_ = new CuTensor<float>(input_shape_);
  input_.reset(input_ptr_);
}







CuTensor<float> *Decoder1Net::forward() {
  return model.forward(input_ptr_);
}



















}  // namespace DMAI
