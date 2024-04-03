#pragma once

#include "engine/onnx/onnx_util.h"
#include "utils/ctensor.h"
#include "TRTEncoderEngine.h"
#include "TRTDecoderIterEngine.h"
#include <memory>
#include <string>
#include <vector>

namespace BASE_NAMESPACE {

class TRTAcousticEngine : private ONNXEngine {
public:
  using ONNXEngine::loadModel;
  TRTAcousticEngine() : is_init_(false) {}
  ~TRTAcousticEngine() {}
  int loadModel(const std::string &encoderPath,const std::string &decoderPath, bool is_crypto = false);
  int infer(const CTensorfl &input,int bss_len, CTensorfl *out);

private:
  TRTEncoderEngine encoder;
  TRTDecoderIterEngine decoder_iter;
  bool is_init_;
};

}; // namespace BASE_NAMESPACE