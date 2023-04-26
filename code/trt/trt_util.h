#ifndef _TRT_UTIL_H_
#define _TRT_UTIL_H_
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <numeric>
#include <memory>
#include <algorithm>
#include <cassert>

using namespace nvinfer1;

namespace BASE_NAMESPACE {

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
 public:
  Logger(Severity severity = Severity::kINFO) : reportableSeverity(severity) {}

  void log(Severity severity, const char *msg) noexcept override {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportableSeverity) return;

    switch (severity) {
    case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
    case Severity::kERROR: std::cerr << "ERROR: "; break;
    case Severity::kWARNING: std::cerr << "WARNING: "; break;
    case Severity::kINFO: std::cerr << "INFO: "; break;
    default: std::cerr << "UNKNOWN: "; break;
    }
    std::cerr << msg << std::endl;
  }

  Severity getReportableSeverity() const {
    return reportableSeverity;
  }

 private:
  Severity reportableSeverity;
};

struct StreamDeleter {
  void operator()(cudaStream_t *stream) {
    if (stream) {
      cudaStreamDestroy(*stream);
      delete stream;
    }
  }
};

template <typename T>
struct TrtDeleter {
  void operator()(T *p) noexcept {
    p->destroy();
  }
};

template <typename T>
struct CuMemDeleter {
  void operator()(T *p) noexcept {
    cudaFree(p);
  }
};

template <typename T, template <typename> class DeleterType = TrtDeleter>
using UniqPtr = std::unique_ptr<T, DeleterType<T>>;

inline void checkCudaErrors(cudaError_t err) {
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorName(err));
}

template <typename T>
inline UniqPtr<T, CuMemDeleter> mallocCudaMem(size_t nbElems) {
  T *ptr = nullptr;
  checkCudaErrors(cudaMalloc((void **)&ptr, sizeof(T) * nbElems));
  return UniqPtr<T, CuMemDeleter>{ptr};
}

inline std::unique_ptr<cudaStream_t, StreamDeleter> makeCudaStream(int flags = cudaStreamDefault) {
  // cudaStream_t stream;
  // checkCudaErrors(cudaStreamCreateWithFlags(&stream, flags));
  // return std::unique_ptr<cudaStream_t, StreamDeleter>{ stream };
  std::unique_ptr<cudaStream_t, StreamDeleter> pStream(new cudaStream_t);
  if (cudaStreamCreateWithFlags(pStream.get(), flags) != cudaSuccess) { pStream.reset(nullptr); }
  return pStream;
}

inline void enableDLA(nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config, int useDLACore,
                      bool allowGPUFallback = true) {
  if (useDLACore >= 0) {
    if (builder->getNbDLACores() == 0) {
      std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores"
                << std::endl;
      return;
    }
    if (allowGPUFallback) { config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK); }
    if (!config->getFlag(nvinfer1::BuilderFlag::kINT8)) {
      // User has not requested INT8 Mode.
      // By default run in FP16 mode. FP32 mode is not permitted.
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    config->setDLACore(useDLACore);
  }
}

inline nvinfer1::Dims toDims(std::vector<int32_t> const &vec) {
  int32_t limit = static_cast<int32_t>(nvinfer1::Dims::MAX_DIMS);
  if (static_cast<int32_t>(vec.size()) > limit) {
    std::cerr << "Vector too long, only first 8 elements are used in dimension." << std::endl;
  }
  // Pick first nvinfer1::Dims::MAX_DIMS elements
  nvinfer1::Dims dims{std::min(static_cast<int32_t>(vec.size()), limit), {}};
  std::copy_n(vec.begin(), dims.nbDims, std::begin(dims.d));
  return dims;
}

inline int elemSize(DataType dataType) {
  switch (dataType) {
  case DataType::kFLOAT: return 4;
  case DataType::kHALF: return 2;
  default: throw std::runtime_error("invalid data type");
  }
}

static Logger gLogger;

inline UniqPtr<ICudaEngine> make_engine_from_onnx(const std::vector<char> &modelBuff) {
  auto builder = UniqPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
  assert((builder != nullptr) && "create builder error");
  std::cout << "create builder success" << std::endl;
  // 动态维度输入需要使用--explicitBatch
  const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = UniqPtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
  assert((network != nullptr) && "create network error");
  std::cout << "create network success" << std::endl;

  auto config = UniqPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  assert((config != nullptr) && "create build config error");
  std::cout << "create config success" <<std::endl;
  auto parser = UniqPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
  assert((parser != nullptr) && "create parser error");
  std::cout << "create parser success"<<std::endl;

  auto parsed = parser->parse(modelBuff.data(), modelBuff.size());
  assert((parsed) && "parser onnx error");
  std::cout << "parser onnx success" <<std::endl;

  auto profile = builder->createOptimizationProfile();
  profile->setDimensions(network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMIN, Dims3({1, 80, 1}));
  profile->setDimensions(network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kOPT, Dims3({1, 80, 800}));
  profile->setDimensions(network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMAX, Dims3({1, 80, 2000}));
  config->addOptimizationProfile(profile);

  // --fp16
  config->setFlag(BuilderFlag::kFP16);

  std::unique_ptr<cudaStream_t, StreamDeleter> profileStream = makeCudaStream(cudaStreamDefault);
  assert((profileStream != nullptr) && "create stream error");
  std::cout << "create stream success" <<std::endl;

  config->setProfileStream(*profileStream);
  UniqPtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  assert((plan != nullptr) && "buildSerializedNetwork error");
  std::cout << "create plan success" <<std::endl;

  UniqPtr<IRuntime> runtime = UniqPtr<IRuntime>{createInferRuntime(gLogger)};
  assert((runtime != nullptr) && "createInferRuntime error");
  std::cout << "create runtime success" <<std::endl;

  UniqPtr<ICudaEngine> engine{runtime->deserializeCudaEngine(plan->data(), plan->size())};
  assert((engine != nullptr) && "create engine error");
  std::cout << "create engine success" <<std::endl;
  return engine;
}

inline std::string GetDeviceName() {
  try {
    cudaError_t status = cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.name;
  }
  catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return "";
  }
}

class TRTInstance {
 public:
  TRTInstance(const std::vector<char> &modelBuff) {
    engine.reset(runtime->deserializeCudaEngine(modelBuff.data(), modelBuff.size()));
    assert((engine != nullptr) && "create engine error");
    std::cout << "create engine success"<<std::endl;

    context.reset(engine->createExecutionContext());
    assert((context != nullptr) && "create context error");
    std::cout << "create context success" <<std::endl;

    const int tensor_num = engine->getNbIOTensors();
    for (int i = 0; i < tensor_num; i++) {
      auto nodename = engine->getIOTensorName(i);
      auto iomode = engine->getTensorIOMode(nodename);
      if (iomode == TensorIOMode::kINPUT)
        inputNames.push_back(nodename);
      else if (iomode == TensorIOMode::kOUTPUT)
        outputNames.push_back(nodename);
    }
    for (auto &inputname : inputNames) {
      auto dims = engine->getProfileShape(inputname.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
      // std::string str = "";
      // for(int i=0;i<dims.nbDims;i++)
      //   str+=std::to_string(dims.d[i])+",";
      // LOG(INFO)<<inputname<<":"<<str;
      auto dataType = engine->getTensorDataType(inputname.c_str());
      int bindingSize = elemSize(dataType) * std::accumulate(dims.d, &dims.d[dims.nbDims], 1, std::multiplies<int>{});
      bindings.emplace_back(mallocCudaMem<char>(bindingSize));
      bindingArray.emplace_back(bindings.back().get());
      // 使用最大维度设置动态维度
      context->setInputShape(inputname.c_str(), dims);
    }
    for (auto &outputname : outputNames) {
      auto dims = context->getTensorShape(outputname.c_str());
      auto dataType = engine->getTensorDataType(outputname.c_str());
      int bindingSize = elemSize(dataType) * std::accumulate(dims.d, &dims.d[dims.nbDims], 1, std::multiplies<int>{});
      bindings.emplace_back(mallocCudaMem<char>(bindingSize));
      bindingArray.emplace_back(bindings.back().get());
    }
    context->executeV2(bindingArray.data());
  }
  TRTInstance(const TRTInstance &) = delete;
  TRTInstance &operator=(const TRTInstance &) = delete;
  TRTInstance(TRTInstance &&) noexcept = default;
  TRTInstance &operator=(TRTInstance &&) noexcept = default;
  int elemSize(const DataType &dataType) {
    switch (dataType) {
    case DataType::kFLOAT: return 4;
    case DataType::kHALF: return 2;
    default: throw std::runtime_error("invalid data type");
    }
  }
  inline static Logger gLogger;
  std::unique_ptr<IRuntime> runtime = std::unique_ptr<IRuntime>{createInferRuntime(gLogger)};
  std::unique_ptr<ICudaEngine> engine = nullptr;
  std::unique_ptr<IExecutionContext> context = nullptr;
  std::vector<UniqPtr<char, CuMemDeleter>> bindings;
  std::vector<void *> bindingArray;  // same content as bindings
  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;
};

}  // namespace BASE_NAMESPACE

#endif