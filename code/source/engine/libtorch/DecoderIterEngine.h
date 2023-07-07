#pragma once

#include <torch/script.h>
#include "utils/ctensor.h"
#include <memory>
#include <string>
#include <vector>

namespace DMAI {

class DecoderIterEngine {
public:
  using TorchModule = torch::jit::script::Module;
  int loadModel(const std::string &modelPath, bool is_crypto = false);
  int infer(const CTensorfl &frame, const CTensorfl &memory,
            const CTensorfl &rnn_h0, const CTensorfl &rnn_h1, CTensorfl *mel,
            CTensorfl *rnn_out0, CTensorfl *rnn_out1);
private:
  std::shared_ptr<TorchModule> model_;
  bool is_init_=false;
};

}; // namespace DMAI