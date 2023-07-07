#ifndef ONNX_UTIL_ENV_H_
#define ONNX_UTIL_ENV_H_
#include "onnxruntime_cxx_api.h"
#include <mutex>
#include <thread>
#include <iostream>

class ONNXENV {
 private:
  ONNXENV(){};
  ~ONNXENV(){};
  ONNXENV(const ONNXENV &) = delete;
  ONNXENV &operator=(const ONNXENV &) = delete;

  inline static Ort::Env *env = nullptr;
  inline static std::mutex mutex;

 public:
  static Ort::Env *getInstance() {
    if (env == nullptr) {
      std::lock_guard<std::mutex> lock(mutex);
      if (env == nullptr) {
        const int thread_pool_size = std::thread::hardware_concurrency() - 2;
        const OrtApi *g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        std::unique_ptr<OrtStatus, decltype(OrtApi::ReleaseStatus)> st_ptr(nullptr, g_ort->ReleaseStatus);
        OrtThreadingOptions *tp_options;
        st_ptr.reset(g_ort->CreateThreadingOptions(&tp_options));
        st_ptr.reset(g_ort->SetGlobalIntraOpNumThreads(tp_options, thread_pool_size));
        st_ptr.reset(g_ort->SetGlobalInterOpNumThreads(tp_options, thread_pool_size));
        env = new Ort::Env(tp_options, ORT_LOGGING_LEVEL_WARNING, "Default");
      }
    }
    return env;
  }
};

class ONNXEngine {
 public:
  int loadModel(const std::vector<char> &modelBuff) {
    try {
      auto env = ONNXENV::getInstance();
      Ort::SessionOptions session_options;
      session_options.DisablePerSessionThreads();
      session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
      // session_options.SetExecutionMode(ORT_SEQUENTIAL);
      session = std::make_unique<Ort::Session>(*env, modelBuff.data(), modelBuff.size(), session_options);
      Ort::AllocatorWithDefaultOptions allocator;
      size_t num_input_nodes = session->GetInputCount();
      for (int i = 0; i < num_input_nodes; i++) {
        char *input_name = session->GetInputName(i, allocator);
        inputNames.push_back(input_name);
      }
      size_t num_output_nodes = session->GetOutputCount();
      for (int i = 0; i < num_output_nodes; i++) {
        char *output_name = session->GetOutputName(i, allocator);
        outputNames.push_back(output_name);
      }
      is_init_ = true;
      return 0;
    }
    catch (Ort::Exception &e) {
      std::cout << __FILE__ << e.what() << std::endl;
      return 1;
    }
  }
  std::unique_ptr<Ort::Session> session;
  std::vector<const char *> inputNames;
  std::vector<const char *> outputNames;
  bool is_init_ = false;
};

#endif