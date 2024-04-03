
#include "TRTDecoderIterEngine.h"

#include "crypto/AESCryptoWrapper.h"
#include "utils/ScopeGuard.h"
#include "utils/file-util.h"
#include "utils/logging.h"
#include "utils/regex-util.h"
#include <cstdint>
#include <map>
#include <numeric>

namespace BASE_NAMESPACE {

int TRTDecoderIterEngine::loadModel(const std::string &modelPath, bool is_crypto) {
  std::string newPath = "";
  std::string device_name = GetDeviceName();
  if (REGEX::search(device_name, "GeForce RTX 2080 Ti")) {
    newPath = REGEX::replace(modelPath, "t4", "2080ti");
  } else if (REGEX::search(device_name, "Tesla T4")) {
    newPath = modelPath;
  }
  CHECK(!newPath.empty()) << "load model on not support device:" << device_name;
  LOG(INFO) << "load model:" << newPath;
  CHECK(is_exist(newPath.c_str())) << " file not exist!";
  auto buff = file_to_buff(newPath.c_str());
  if (is_crypto) buff = AESCipherDecrypt(buff);
  int res = loadModel(buff);
  if (res == 0) is_init_ = true;
  LOG(INFO) << "load model end";
  return res;
}

int TRTDecoderIterEngine::loadModel(const std::vector<char> &modelBuff) {
  for (int i = 0; i < instance_num; i++) {
    ins_queue.Push(TRTInstance(modelBuff));
  }
  LOG(INFO) << "load vocoder model success";
  is_init_ = true;
  return 0;
}

int TRTDecoderIterEngine::infer(const CTensorfl &memories,CTensorfl *out) {
  LOG(INFO) << "TRTDecoderIterEngine::infer";
  try {
    if (!is_init_) {
      LOG(ERROR) << "infer error;please call loadModel() before";
      return -1;
    }

    auto decoder_ins = ins_queue.Pop();
    ScopeGuard scopeguard([&]() { ins_queue.Push(std::move(decoder_ins)); });
    CTensorfl frame({1,52}),rnn_h0({1,1024}),rnn_h1({1,1024})/*,rnn_out0({1,1024}),rnn_out1({1,1024})*/;
    
    cudaMemcpy(decoder_ins.bindingArray[0], frame.data(), frame.byteSize(),cudaMemcpyHostToDevice);
    // cudaMemcpy(decoder_ins.bindingArray[1], memory.data(), memory.byteSize(), cudaMemcpyHostToDevice);
    cudaMemcpy(decoder_ins.bindingArray[2], rnn_h0.data(), rnn_h0.byteSize(), cudaMemcpyHostToDevice);
    cudaMemcpy(decoder_ins.bindingArray[3], rnn_h1.data(), rnn_h1.byteSize(), cudaMemcpyHostToDevice);
    out->resize({1,memories.shapes()[1],52});
    for(int i=0;i<memories.shapes()[1];i++){
      cudaMemcpy(decoder_ins.bindingArray[1],&memories.at({0,i,0}),memories.shapes()[2]*sizeof(float),cudaMemcpyHostToDevice);
      auto statu = decoder_ins.context->executeV2(decoder_ins.bindingArray.data());
      cudaMemcpy(&(out->at({0,i,0})), decoder_ins.bindingArray[6], frame.byteSize(), cudaMemcpyDeviceToHost);
      cudaMemcpy(decoder_ins.bindingArray[0], decoder_ins.bindingArray[6], frame.byteSize(), cudaMemcpyDeviceToDevice);
      cudaMemcpy(decoder_ins.bindingArray[2], decoder_ins.bindingArray[4], rnn_h0.byteSize(), cudaMemcpyDeviceToDevice);
      cudaMemcpy(decoder_ins.bindingArray[3], decoder_ins.bindingArray[5], rnn_h1.byteSize(), cudaMemcpyDeviceToDevice);
     
    }
    LOG(INFO) << "TRTDecoderIterEngine::infer end";
    return 0;
  } catch (std::exception &e) {
    LOG(ERROR) << e.what();
    return -1;
  }
}

};  // namespace BASE_NAMESPACE