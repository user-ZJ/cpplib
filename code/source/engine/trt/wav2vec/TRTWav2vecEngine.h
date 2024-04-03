#pragma once


#include "utils/ctensor.h"
#include <memory>
#include <string>
#include <vector>
#include "engine/trt/trt_util.h"
#include "utils/blocking_queue.h"

namespace BASE_NAMESPACE {

class TRTWav2vecEngine {
public:
  int loadModel(const std::string &modelPath, bool is_crypto = false);
  int loadModel(const std::vector<char> &modelBuff);
  int infer(const CTensorfl &input, CTensorfl *out);
private:
  int instance_num = 3;
  BlockingQueue<TRTInstance> ins_queue;
  bool is_init_;
};

}; // namespace BASE_NAMESPACE