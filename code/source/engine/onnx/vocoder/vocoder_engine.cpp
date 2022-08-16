/*
 * @Author: dm.liujunshi
 * @Date: 2021-07-29 09:15:27
 * @Last Modified by: dm.liujunshi
 * @Last Modified time: 2021-09-07 20:38:29
 */

#include "vocoder_engine.h"

#include "utils/logging.h"

#include <cstdint>
#include <map>
#include <numeric>

namespace DMAI {

int VocoderEngine::loadModel(const std::string &modelPath, int num_threads) {
  LOG(INFO) << "load vocoder model:" << modelPath;
  Ort::Env &env = ONNXENV::getInstance();
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(num_threads);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  hifigan_inputNames = {"mel"};
  hifigan_outputNames = {"y"};
  hifigan_session = std::make_unique<Ort::Session>(env, modelPath.c_str(), session_options);

  LOG(INFO) << "load vocoder model success";
  is_init_ = true;
  return 0;
}

 int VocoderEngine::infer(const CTensorfl &mels,CTensorfl *out) {
  LOG(INFO) << "VocoderEngine::infer";
  try {
    if (!is_init_) {
      LOG(ERROR) << "infer error;please call loadModel() before";
      return -1;
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> hifigan_inputTensors;
    Ort::MemoryInfo memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    hifigan_inputTensors.push_back(Ort::Value::CreateTensor<float>(
      memoryInfo, const_cast<float *>(mels.data()), mels.size(), mels.shapes().data(), mels.shapes().size()));
    auto hifigan_out = hifigan_session->Run(Ort::RunOptions{nullptr}, hifigan_inputNames.data(),
                                            hifigan_inputTensors.data(), 1, hifigan_outputNames.data(), 1);
    std::vector<int64_t> output_shape = hifigan_out[0].GetTensorTypeAndShapeInfo().GetShape();
    out->resize(output_shape);
    memcpy(out->data(), hifigan_out[0].GetTensorMutableData<float>(), out->byteSize());

    LOG(INFO) << "VocoderEngine::infer end";
    return 0;
  } catch (Ort::Exception e) { 
    LOG(ERROR) << e.what(); 
    return -1;
  }
}

};  // namespace DMAI