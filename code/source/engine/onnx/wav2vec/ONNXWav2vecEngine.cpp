
#include "ONNXWav2vecEngine.h"

#include "crypto/AESCryptoWrapper.h"
#include "utils/file-util.h"
#include "utils/logging.h"

#include <cstdint>
#include <map>
#include <numeric>

namespace BASE_NAMESPACE {

int ONNXWav2vecEngine::loadModel(const std::string &modelPath, bool is_crypto) {
  LOG(INFO) << "load model:" << modelPath<<" is_crypto:"<<is_crypto;
  CHECK(is_exist(modelPath.c_str()))<<" file not exist!";
  auto buff = file_to_buff(modelPath.c_str());
  if (is_crypto)
    buff = AESCipherDecrypt(buff);
  int res = loadModel(buff);
  if (res == 0)
    is_init_ = true;
  LOG(INFO) << "load model end";
  return res;
}

int ONNXWav2vecEngine::infer(const CTensorfl &input, CTensorfl *out) {
  LOG(INFO) << "ONNXWav2vecEngine::infer";
  try {
    if (!is_init_) {
      LOG(ERROR) << "infer error;please call loadModel() before";
      return -1;
    }

    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float *>(input.data()), input.size(),
        input.shapes().data(), input.shapes().size()));
    auto outputTensors =
        session->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                     inputTensors.data(), inputNames.size(), outputNames.data(), outputNames.size());
    std::vector<int64_t> output_shape =
        outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    out->resize(output_shape);
    memcpy(out->data(), outputTensors[0].GetTensorMutableData<float>(),
           out->byteSize());

    LOG(INFO) << "ONNXWav2vecEngine::infer end";
    return 0;
  } catch (Ort::Exception e) {
    LOG(ERROR) << e.what();
    return -1;
  }
}

}; // namespace BASE_NAMESPACE