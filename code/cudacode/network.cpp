#include "network.h"

#include "helper.h"
#include "layer.h"

#include "utils/logging.h"
#include "utils/string-util.h"
#include <iomanip>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

namespace CUDA_NAMESPACE {

Network::Network() {
  // do nothing
}

Network::~Network() {
  // destroy network
  for (auto layer : layers_)
    delete layer;

  // terminate CUDA context
  if (cuda_ != nullptr) delete cuda_;
}

void Network::add_layer(Layer *layer) {
  layers_.push_back(layer);
}

CuTensor<float> *Network::forward(CuTensor<float> *input) {
  input_ptr_ = input;

  for (auto layer : layers_) {
    input_ptr_ = layer->forward(input_ptr_);
    // checkCudaErrors(cudaDeviceSynchronize());
    // output_->cpu().dump2File(layer->get_name().c_str());
  }
  *output_ptr_ = *input_ptr_;

  return output_ptr_;
}





int Network::write_file() {
  LOG(INFO) << ".. store weights to the storage .." << std::endl;
  for (auto layer : layers_) {
    int err = layer->save_parameter();

    if (err != 0) {
      LOG(INFO) << "-> error code: " << err << std::endl;
      exit(err);
    }
  }

  return 0;
}

int Network::load_pretrain() {
  for (auto layer : layers_) {
    layer->load_parameter();
  }
  return 0;
}

// 1. initialize cuda resource container
// 2. register the resource container to all the layers
void Network::cuda() {
  cuda_ = new CudaContext();

  LOG(INFO) << ".. model Configuration .." << std::endl;
  for (auto layer : layers_) {
    LOG(INFO) << "CUDA: " << layer->get_name() << std::endl;
    layer->set_cuda_context(cuda_);
  }
}



void Network::test() {
  phase_ = inference;

  // freeze all layers
  for (auto layer : layers_) {
    layer->freeze();
  }
}

std::vector<Layer *> Network::layers() {
  return layers_;
}



int Network::set_input_shape(const std::vector<int> &input_shape) {
  input_shape_ = input_shape;
  output_shape_ = input_shape_;
  for (auto layer : layers_) {
	layer->set_input_shape(output_shape_);
    output_shape_ = layer->get_output_shape();
  }
  output_ptr_ = new CuTensor<float>(output_shape_);
  output_.reset(output_ptr_);
  return 0;
}

}  // namespace DMAI
