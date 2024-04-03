#pragma once

#include "engine/onnx/onnx_util.h"
#include "utils/ctensor.h"
#include <memory>
#include <string>
#include <vector>

namespace BASE_NAMESPACE {

class ONNXEncoderEngine : private ONNXEngine {
public:
  using ONNXEngine::loadModel;
  int loadModel(const std::string &modelPath, bool is_crypto = false);
  int infer(const CTensorfl &input, CTensorfl *out);

};

}; // namespace BASE_NAMESPACE