#include "Layer.h"

#include <random>

#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>

#include <fstream>
#include <iostream>
#include <sstream>

namespace CUDA_NAMESPACE {

/****************************************************************
 * Layer definition                                             *
 ****************************************************************/
Layer::Layer() : freeze_(false), batch_size_(0),input_num_(0),output_num_(0) {}

Layer::~Layer() {}

std::string Layer::get_name() {
  return name_;
}

std::vector<int> Layer::get_output_shape() {
  return input_shapes_;
}
std::vector<int> Layer::get_input_shape() {
  return output_shapes_;
}

}  // namespace CUDA_NAMESPACE
