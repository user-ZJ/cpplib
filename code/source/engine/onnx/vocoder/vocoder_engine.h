#pragma once

#include "onnx_util.h"
#include "utils/ctensor.h"
#include <memory>
#include <string>
#include <vector>

namespace DMAI {

class VocoderEngine {
 public:
  VocoderEngine() : is_init_(false) {}
  ~VocoderEngine() {}
  int loadModel(const std::string &modelPath,int num_threads=4);
  int infer(const CTensorfl &mels,CTensorfl *out);

 private:
  std::unique_ptr<Ort::Session> hifigan_session;
  std::vector<const char *> hifigan_inputNames;
  std::vector<const char *> hifigan_outputNames;
  bool is_init_;
  int n_mel_channels = 80;
};

};  // namespace DMAI