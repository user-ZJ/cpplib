#pragma once

#include "engine/onnx/onnx_util.h"
#include "utils/ctensor.h"
#include "ONNXEncoderEngine.h"
#include "ONNXDecoderIterEngine.h"
#include <memory>
#include <string>
#include <vector>

namespace BASE_NAMESPACE {

class ONNXAcousticEngine : private ONNXEngine {
public:
  using ONNXEngine::loadModel;
  ONNXAcousticEngine() : is_init_(false) {}
  ~ONNXAcousticEngine() {}
  int loadModel(const std::string &encoderPath,const std::string &decoderPath, bool is_crypto = false);
  int infer(const CTensorfl &input,int bss_len, CTensorfl *out);

private:
  ONNXEncoderEngine encoder;
  ONNXDecoderIterEngine decoder_iter;
  bool is_init_;
};

}; // namespace BASE_NAMESPACE