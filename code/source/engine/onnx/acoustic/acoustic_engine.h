#pragma once

#include "onnx_util.h"

#include <memory>
#include <string>
#include <vector>
#include "utils/ctensor.h"

#define ONNX_

namespace DMAI {

class AcousticEngine {
 public:
  AcousticEngine() :
    is_init_(false) {}
  ~AcousticEngine() {
  }
  int loadModel(const std::string &encoderPath,
                const std::string &decoderPath,
                const std::string &postnetPath,
                int num_threads=4);
  int infer(const CTensorll &seqs,const CTensorll &skips,const std::vector<std::string> &phonemes,CTensorfl *out,std::vector<int> *dur=nullptr);
  int encoderInfer(const CTensorll &seqs,const CTensorll &skips,const std::vector<std::string> &phonemes,CTensorfl *out,std::vector<int> *dur);
  int decoderInfer(const CTensorfl &memories,CTensorfl *out);
  int postnetInfer(const CTensorfl &mels,CTensorfl *out);


 private:
  std::unique_ptr<Ort::Session> encoder_session;
  std::vector<const char *> encoder_inputNames;
  std::vector<const char *> encoder_outputNames;
  std::unique_ptr<Ort::Session> decoder_session;
  std::vector<const char *> decoder_inputNames;
  std::vector<const char *> decoder_outputNames;
  std::unique_ptr<Ort::Session> postnet_session;
  std::vector<const char *> postnet_inputNames;
  std::vector<const char *> postnet_outputNames;
  int n_mel_channels = 80;
  int decoder_rnn_dim = 256;
  int n_frames_per_step = 3;
  bool is_init_;
};

};  // namespace DMAI
