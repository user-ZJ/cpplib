#pragma once

#include "engine/trt/trt_util.h"
#include "utils/ScopeGuard.h"
#include "utils/blocking_queue.h"
#include "utils/ctensor.h"
#include "utils/logging.h"
#include <memory>
#include <string>
#include <vector>

namespace DMAI {

class EmbeddingEngine {
public:
  EmbeddingEngine() : is_init_(false) {}
  ~EmbeddingEngine() {}
  int loadModel(const std::string &modelPath, int num_threads = 4,
                bool is_crypto = true);
  int loadModel(const std::vector<char> &modelBuff, int num_threads = 4);
  int infer(const std::vector<std::vector<int>> &ids, CTensorfl &out);

private:
  BlockingQueue<TRTInstance> ins_queue;
  int instance_num = 2;
  bool is_init_;
};

}; // namespace DMAI