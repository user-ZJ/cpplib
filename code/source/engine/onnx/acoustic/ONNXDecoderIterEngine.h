#pragma once

#include "engine/onnx/onnx_util.h"
#include "utils/ctensor.h"
#include <memory>
#include <string>
#include <vector>

namespace BASE_NAMESPACE {

class ONNXDecoderIterEngine : private ONNXEngine {
public:
  using ONNXEngine::loadModel;
  int loadModel(const std::string &modelPath, bool is_crypto = false);
  int infer(const CTensorfl &frame, const CTensorfl &memory,
            const CTensorfl &rnn_h0, const CTensorfl &rnn_h1, CTensorfl *mel,
            CTensorfl *rnn_out0, CTensorfl *rnn_out1);

};

}; // namespace BASE_NAMESPACE