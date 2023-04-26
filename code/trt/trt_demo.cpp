#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "trt_util.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace nvinfer1;
using namespace BASE_NAMESPACE;

int main(int argc, char *argv[]) {
  // 获取显卡型号
  cudaError_t status = cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Device name: " << prop.name << std::endl;
  std::cout << "totalGlobalMem: " << prop.totalGlobalMem / 1024.0 / 1024 << "MB" << std::endl;
  size_t free_byte, total_byte;
  cudaMemGetInfo(&free_byte, &total_byte);
  std::cout << "Total memory: " << total_byte / (1024.0 * 1024.0) << " MB" << std::endl;
  std::cout << "Free memory: " << free_byte / (1024.0 * 1024.0) << " MB" << std::endl;
  std::cout << "used memory: " << (total_byte-free_byte) / (1024.0 * 1024.0) << " MB" << std::endl;


  static Logger gLogger;

  std::string onnx_path = "resnet.onnx";

  // 解析onnx文件
  auto builder = UniqPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
  if (!builder) std::cout << "create builder error\n";
  // 动态维度输入需要使用--explicitBatch
  const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = UniqPtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
  if (!network) std::cout << "create network error\n";

  auto config = UniqPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) std::cout << "create build config error\n";
  auto parser = UniqPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
  if (!parser) std::cout << "create parser error\n";

  auto parsed = parser->parseFromFile(onnx_path.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
  if (!parsed) std::cout << "parser onnx error\n";

  int nb_inputs = network->getNbInputs();
  int nb_outputs = network->getNbInputs();
  std::cout << "input num:" << nb_inputs << " output num:" << nb_outputs << std::endl;
  std::cout << "input dims:\n";
  for (int i = 0; i < nb_inputs; i++) {
    std::cout << "name:" << network->getInput(i)->getName() << " ";
    const auto dims = network->getInput(i)->getDimensions();
    for (int j = 0; j < dims.nbDims; j++)
      std::cout << dims.d[j] << " x ";
    std::cout << std::endl;
  }
  std::cout << "output dims:\n";
  for (int i = 0; i < nb_outputs; i++) {
    std::cout << "name:" << network->getOutput(i)->getName() << " ";
    const auto dims = network->getOutput(i)->getDimensions();
    for (int j = 0; j < dims.nbDims; j++)
      std::cout << dims.d[j] << " x ";
    std::cout << std::endl;
  }
  // 设置动态维度区间
  auto profile = builder->createOptimizationProfile();
  profile->setDimensions("feats", nvinfer1::OptProfileSelector::kMIN, Dims3({1, 10, 80}));
  profile->setDimensions("feats", nvinfer1::OptProfileSelector::kOPT, Dims3({1, 200, 80}));
  profile->setDimensions("feats", nvinfer1::OptProfileSelector::kMAX, Dims3({1, 500, 80}));
  config->addOptimizationProfile(profile);

  // --fp16
  config->setFlag(BuilderFlag::kFP16);

  // dla
  // enableDLA(builder.get(), config.get(), 1);

  std::unique_ptr<cudaStream_t, StreamDeleter> profileStream = makeCudaStream(cudaStreamDefault);
  if (!profileStream) std::cout << "create stream error\n";

  config->setProfileStream(*profileStream);
  UniqPtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  if (!plan) std::cout << "buildSerializedNetwork error\n";

  UniqPtr<IRuntime> runtime = UniqPtr<IRuntime>{createInferRuntime(gLogger)};
  if (!runtime) std::cout << "createInferRuntime error\n";

  // 直接加载trt文件
  // std::string filename = "gpu.engine";
  // std::ifstream fin(filename, std::ios::binary);
  // std::vector<char> inBuffer((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
  // UniqPtr<ICudaEngine> engine{runtime->deserializeCudaEngine(inBuffer.data(), inBuffer.size(), nullptr)};

  // 从内存中加载onnx文件
  UniqPtr<ICudaEngine> engine{runtime->deserializeCudaEngine(plan->data(), plan->size())};
  if (!engine) std::cout << "create engine error\n";

  // 创建运行上下文
  UniqPtr<IExecutionContext> context{engine->createExecutionContext()};

  // 模型输入/输出数据个数
  const int tensor_num = engine->getNbIOTensors();
  std::cout << "tensor_num:" << tensor_num << std::endl;
  for (int i = 0; i < tensor_num; i++) {
    std::cout << "name:" << engine->getIOTensorName(i) << " ";
    const auto dataType = engine->getTensorDataType(engine->getIOTensorName(i));
    const auto dims = engine->getTensorShape(engine->getIOTensorName(i));
    for (int j = 0; j < dims.nbDims; j++)
      std::cout << dims.d[j] << " x ";
    std::cout << std::endl;
  }
  //使用最大维度推理一遍
  auto inputname = engine->getIOTensorName(0);
  auto inputtype = engine->getTensorDataType(inputname);
  //   auto inputdims = engine->getTensorShape(inputname);
  auto inputdims = engine->getProfileShape(inputname, 0, nvinfer1::OptProfileSelector::kMAX);
  auto outputname = engine->getIOTensorName(1);
  auto outputtype = engine->getTensorDataType(outputname);
  auto outputdims = engine->getTensorShape(outputname);
  const int inbindingSize =
    elemSize(inputtype) * std::accumulate(inputdims.d, &inputdims.d[inputdims.nbDims], 1, std::multiplies<int>{});
  context->setInputShape(inputname, inputdims);
  outputdims = context->getTensorShape(outputname);
  int outbindingSize =
    elemSize(outputtype) * std::accumulate(outputdims.d, &outputdims.d[outputdims.nbDims], 1, std::multiplies<int>{});
  std::cout << "inbindingSize:" << inbindingSize << " outbindingSize:" << outbindingSize << std::endl;
  std::vector<void *> bindingArray;
  auto input_ptr = mallocCudaMem<char>(inbindingSize);
  auto output_ptr = mallocCudaMem<char>(outbindingSize);
  bindingArray.push_back(input_ptr.get());
  bindingArray.push_back(output_ptr.get());

  // 使用自定义维度推理
  Dims3 sdim{1, 215, 80};
  context->setInputShape(inputname, sdim);
  std::vector<float> dinput(1 * 125 * 80);
  cudaMemcpy(bindingArray[0], dinput.data(), dinput.size() * sizeof(float), cudaMemcpyHostToDevice);
  auto statu = context->executeV2(bindingArray.data());
  outputdims = context->getTensorShape(outputname);
  outbindingSize = std::accumulate(outputdims.d, &outputdims.d[outputdims.nbDims], 1, std::multiplies<int>{});
  std::vector<float> doutput(outbindingSize);
  cudaMemcpy(doutput.data(), bindingArray[1], elemSize(outputtype) * doutput.size(), cudaMemcpyDeviceToHost);
  for (auto &d : doutput) {
    std::cout << d << " ";
  }
  std::cout << std::endl;

  return 0;
}