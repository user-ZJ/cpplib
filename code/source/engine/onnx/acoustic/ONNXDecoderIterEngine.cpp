
#include "ONNXDecoderIterEngine.h"

#include "crypto/AESCryptoWrapper.h"
#include "utils/file-util.h"
#include "utils/logging.h"

#include <cstdint>
#include <map>
#include <numeric>

namespace BASE_NAMESPACE {

int ONNXDecoderIterEngine::loadModel(const std::string &modelPath, bool is_crypto) {
  LOG(INFO) << "load model:" << modelPath;
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

int ONNXDecoderIterEngine::infer(const CTensorfl &frame, const CTensorfl &memory,
            const CTensorfl &rnn_h0, const CTensorfl &rnn_h1, CTensorfl *mel,
            CTensorfl *rnn_out0, CTensorfl *rnn_out1) {
  // LOG(INFO) << "ONNXDecoderIterEngine::infer";
  try {
    if (!is_init_) {
      LOG(ERROR) << "infer error;please call loadModel() before";
      return -1;
    }

    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float *>(frame.data()), frame.size(),
        frame.shapes().data(), frame.shapes().size()));
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float *>(memory.data()), memory.size(),
        memory.shapes().data(), memory.shapes().size()));
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float *>(rnn_h0.data()), rnn_h0.size(),
        rnn_h0.shapes().data(), rnn_h0.shapes().size()));
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float *>(rnn_h1.data()), rnn_h1.size(),
        rnn_h1.shapes().data(), rnn_h1.shapes().size()));
    auto outputTensors =
        session->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                     inputTensors.data(), inputNames.size(), outputNames.data(), outputNames.size());
    mel->resize(outputTensors[0].GetTensorTypeAndShapeInfo().GetShape());
    rnn_out0->resize(outputTensors[1].GetTensorTypeAndShapeInfo().GetShape());
    rnn_out1->resize(outputTensors[2].GetTensorTypeAndShapeInfo().GetShape());
    memcpy(mel->data(), outputTensors[0].GetTensorMutableData<float>(),mel->byteSize());
    memcpy(rnn_out0->data(), outputTensors[1].GetTensorMutableData<float>(),rnn_out0->byteSize());
    memcpy(rnn_out1->data(), outputTensors[2].GetTensorMutableData<float>(),rnn_out1->byteSize());

    // LOG(INFO) << "ONNXDecoderIterEngine::infer end";
    return 0;
  } catch (Ort::Exception e) {
    LOG(ERROR) << e.what();
    return -1;
  }
}

}; // namespace BASE_NAMESPACE