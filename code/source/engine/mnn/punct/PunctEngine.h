/*
 * @Author: zack 
 * @Date: 2022-06-16 09:56:20 
 * @Last Modified by: zack
 * @Last Modified time: 2022-06-16 09:56:47
 */
#pragma once
#include "MNN/Interpreter.hpp"
#include <memory>
#include <string>
#include <vector>

using namespace MNN;

namespace BASE_NAMESPACE {

class PunctEngine {
 public:
  PunctEngine();
  ~PunctEngine();
  int loadModel(const std::string &modelPath);
  int loadModel(const void *buffer, size_t size);
  std::vector<float> infer(const std::vector<float> &input);

 private:
  std::unique_ptr<Interpreter> net;
  Session *session;
  bool is_init_;
};

};  // namespace BASE_NAMESPACE
