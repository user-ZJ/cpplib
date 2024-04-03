#ifndef ONNX_UTIL_ENV_H_
#define ONNX_UTIL_ENV_H_
#include "onnxruntime_cxx_api.h"
#include <mutex>
#include <thread>
#include "utils/logging.h"

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
        std::unique_ptr<OrtStatus, decltype(OrtApi::ReleaseStatus)> st_ptr(
            nullptr, g_ort->ReleaseStatus);
        OrtThreadingOptions *tp_options;
        st_ptr.reset(g_ort->CreateThreadingOptions(&tp_options));
        st_ptr.reset(
            g_ort->SetGlobalIntraOpNumThreads(tp_options, thread_pool_size));
        st_ptr.reset(
            g_ort->SetGlobalInterOpNumThreads(tp_options, thread_pool_size));
        env = new Ort::Env(tp_options, ORT_LOGGING_LEVEL_WARNING, "Default");
      }
    }
    return env;
  }
};

class ONNXEngine {
public:
  int loadModel(const std::vector<char> &modelBuff) {
    auto env = ONNXENV::getInstance();
    Ort::SessionOptions session_options;
    session_options.DisablePerSessionThreads();
    // session_options.SetGraphOptimizationLevel(
    //     GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    // session_options.SetExecutionMode(ORT_SEQUENTIAL);
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_options.gpu_mem_limit = static_cast<int>(4 * 1024 * 1024);
    cuda_options.arena_extend_strategy = 1;
    cuda_options.default_memory_arena_cfg = nullptr;
    // session_options.AppendExecutionProvider_CUDA(cuda_options);
    session = std::make_unique<Ort::Session>(*env, modelBuff.data(),
                                             modelBuff.size(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session->GetInputCount();
    for (int i = 0; i < num_input_nodes; i++) {
      auto input_name = session->GetInputNameAllocated(i, allocator);
      LOG(INFO)<<input_name.get();
      inputNames_str.push_back(std::string(input_name.get()));
      inputNames.push_back(inputNames_str[i].c_str());
    }
    size_t num_output_nodes = session->GetOutputCount();
    for (int i = 0; i < num_output_nodes; i++) {
      auto output_name = session->GetOutputNameAllocated(i, allocator);
      LOG(INFO)<<output_name.get();
      outputNames_str.push_back(std::string(output_name.get()));
      outputNames.push_back(outputNames_str[i].c_str());
    }
    is_init_ = true;
    return 0;
  }
  std::unique_ptr<Ort::Session> session;
  std::vector<std::string> inputNames_str;
  std::vector<const char *> inputNames;
  std::vector<std::string> outputNames_str;
  std::vector<const char *> outputNames;
  bool is_init_=false;
};

#endif