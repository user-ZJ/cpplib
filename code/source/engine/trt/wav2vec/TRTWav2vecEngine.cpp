
#include "TRTWav2vecEngine.h"

#include "crypto/AESCryptoWrapper.h"
#include "utils/file-util.h"
#include "utils/logging.h"
#include "utils/ScopeGuard.h"
#include "utils/regex-util.h"
#include <cstdint>
#include <map>
#include <numeric>

namespace BASE_NAMESPACE {

int TRTWav2vecEngine::loadModel(const std::string &modelPath, bool is_crypto) {
  std::string newPath = "";
  std::string device_name = GetDeviceName();
  if(REGEX::search(device_name,"GeForce RTX 2080 Ti")){
    newPath = REGEX::replace(modelPath,"t4","2080ti");
  }else if(REGEX::search(device_name,"Tesla T4")){
    newPath = modelPath;
  }
  CHECK(!newPath.empty())<<"load model on not support device:"<<device_name;
  LOG(INFO) << "load model:" << newPath<<" is_crypto:"<<is_crypto;
  CHECK(is_exist(newPath.c_str()))<<" file not exist!";
  auto buff = file_to_buff(newPath.c_str());
  if (is_crypto)
    buff = AESCipherDecrypt(buff);
  int res = loadModel(buff);
  if (res == 0)
    is_init_ = true;
  LOG(INFO) << "load model end";
  return res;
}

int TRTWav2vecEngine::loadModel(const std::vector<char> &modelBuff) {
  for(int i=0;i<instance_num;i++){
    ins_queue.Push(TRTInstance(modelBuff));
  }
  LOG(INFO) << "load vocoder model success";
  is_init_ = true;
  return 0;
}

int TRTWav2vecEngine::infer(const CTensorfl &input, CTensorfl *out) {
  LOG(INFO) << "TRTWav2vecEngine::infer";
  try {
    if (!is_init_) {
      LOG(ERROR) << "infer error;please call loadModel() before";
      return -1;
    }

    auto ins = ins_queue.Pop();
    ScopeGuard scopeguard([&](){
      ins_queue.Push(std::move(ins));
    });

    auto inputshape = input.shapes();
    Dims2 inputdims{inputshape[0], inputshape[1]};
    ins.context->setInputShape(ins.inputNames[0].c_str(), inputdims);
    auto outputdims = ins.context->getTensorShape(ins.outputNames[0].c_str());

    cudaMemcpy(ins.bindingArray[0], input.data(), input.byteSize(), cudaMemcpyHostToDevice);
    auto statu = ins.context->executeV2(ins.bindingArray.data());
    out->resize({outputdims.d[0], outputdims.d[1], outputdims.d[2]});
    cudaMemcpy(out->data(), ins.bindingArray[1], out->byteSize(), cudaMemcpyDeviceToHost);

    LOG(INFO) << "TRTWav2vecEngine::infer end";
    return 0;
  } catch (std::exception &e) {
    LOG(ERROR) << e.what();
    return -1;
  }
}

}; // namespace BASE_NAMESPACE