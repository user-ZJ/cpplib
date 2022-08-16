#pragma once
#include "MNN/Interpreter.hpp"
#include <string>
#include <vector>

using namespace MNN;

namespace DMAI {

class ZHG2PStartEngine {
 public:
  ZHG2PStartEngine();
  ~ZHG2PStartEngine();
  int loadModel(const std::string &modelPath);
  int loadModel(const char *buffer, size_t size);
  vector<float> infer(const vector<int> &input);

 private:
  std::unique_ptr<Interpreter> net;
  Session *session;
  bool is_init_;
};

class ZHG2PEndEngine {
 public:
  ZHG2PEndEngine();
  ~ZHG2PEndEngine();
  int loadModel(const std::string &modelPath);
  int loadModel(const char *buffer, size_t size);
  vector<int> infer(const vector<float> &input, const vector<int> &x_ids, const vector<int> &x_cm);

 private:
  std::unique_ptr<Interpreter> net;
  Session *session;
  bool is_init_;
};

};  // namespace DMAI
