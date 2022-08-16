#pragma once
#include "MNN/Interpreter.hpp"
#include <memory>
#include <string>
#include <vector>

using namespace MNN;

namespace DMAI {

class ProsodyEngine {
 public:
  ProsodyEngine();
  ~ProsodyEngine();
  int loadModel(const std::string &modelPath);
  int loadModel(const void *buffer, size_t size);
  vector<vector<float>> infer(const vector<int> &input);

 private:
  std::unique_ptr<Interpreter> net;
  Session *session;
  bool is_init_;
};

};  // namespace DMAI
