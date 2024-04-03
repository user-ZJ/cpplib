
#include "DecoderIterEngine.h"

#include "crypto/AESCryptoWrapper.h"
#include "utils/file-util.h"
#include "utils/logging.h"
#include <exception>
#include <cstdint>
#include <map>
#include <numeric>

namespace BASE_NAMESPACE {

int DecoderIterEngine::loadModel(const std::string &modelPath, bool is_crypto) {
  LOG(INFO) << "load model:" << modelPath;
  CHECK(is_exist(modelPath.c_str()))<<" file not exist!";
  // auto buff = file_to_buff(modelPath.c_str());
  // if (is_crypto)
  //   buff = AESCipherDecrypt(buff);
  // int res = loadModel(modelPath);
  // if (res == 0)
  torch::jit::script::Module model = torch::jit::load(modelPath);
  model_ = std::make_shared<TorchModule>(std::move(model));
  is_init_ = true;
  LOG(INFO) << "load model end";
  return 0;
}



int DecoderIterEngine::infer(const CTensorfl &frame, const CTensorfl &memory,
            const CTensorfl &rnn_h0, const CTensorfl &rnn_h1, CTensorfl *mel,
            CTensorfl *rnn_out0, CTensorfl *rnn_out1) {
  // LOG(INFO) << "DecoderIterEngine::infer";
  try {
    if (!is_init_) {
      LOG(ERROR) << "infer error;please call loadModel() before";
      return -1;
    }
    auto tframe = torch::from_blob(frame.data(),{1,52},torch::kFloat);
    auto tmemory = torch::from_blob(memory.data(),{1,512},torch::kFloat);
    auto trnn_h0 = torch::from_blob(rnn_h0.data(),{1,1024},torch::kFloat);
    auto trnn_h1 = torch::from_blob(rnn_h1.data(),{1,1024},torch::kFloat);
    
    std::vector<torch::jit::IValue> torch_inputs{tframe,tmemory,trnn_h0,trnn_h1};
    auto outputs = model_->get_method("forward")(torch_inputs).toTuple()->elements();
    mel->resize({1,52});
    rnn_out0->resize({1,1024});
    rnn_out1->resize({1,1024});
    memcpy(mel->data(),outputs[0].toTensor().data_ptr(),mel->byteSize());
    memcpy(rnn_out0->data(),outputs[1].toTensor().data_ptr(),rnn_out0->byteSize());
    memcpy(rnn_out1->data(),outputs[2].toTensor().data_ptr(),rnn_out1->byteSize());

    
    // LOG(INFO) << "DecoderIterEngine::infer end";
    return 0;
  } catch (std::exception &e) {
    LOG(ERROR) << e.what();
    return -1;
  }
}

}; // namespace BASE_NAMESPACE