/*
 * @Author: dm.liujunshi
 * @Date: 2021-07-29 09:15:27
 * @Last Modified by: dm.liujunshi
 * @Last Modified time: 2021-09-07 20:38:29
 */

#include "embedding_engine.h"
#include "crypto/AESCryptoWrapper.h"
#include "utils/file-util.h"
#include "utils/logging.h"
#include "utils/regex-util.h"
#include "utils/timer.h"
#include <cstdint>
#include <map>
#include <numeric>

namespace DMAI {

int EmbeddingEngine::loadModel(const std::string &modelPath, int num_threads,
                               bool is_crypto) {
  std::string newPath = "";
  std::string device_name = GetDeviceName();
  if (REGEX::search(device_name, "GeForce RTX 2080 Ti")) {
    newPath = REGEX::replace(modelPath, "t4", "2080ti");
  } else if (REGEX::search(device_name, "Tesla T4")) {
    newPath = modelPath;
  } else if (REGEX::search(device_name, "1080 Ti")) {
    newPath = REGEX::replace(modelPath, "t4", "1080ti");
  }
  CHECK(!newPath.empty()) << "load model on not support device:" << device_name;
  CHECK(is_exist(newPath.c_str())) << "file not exist:" << newPath;
  LOG(INFO) << "load model:" << newPath;
  std::vector<char> model_buff = file_to_buff(newPath.c_str());
  if (is_crypto) {
    model_buff = AESCipherDecrypt(model_buff);
  }
  return loadModel(model_buff, num_threads);
}

int EmbeddingEngine::loadModel(const std::vector<char> &modelBuff,
                               int num_threads) {

  for (int i = 0; i < instance_num; i++) {
    ins_queue.Push(TRTInstance(modelBuff));
  }
  LOG(INFO) << "load vocoder model success";
  is_init_ = true;
  return 0;
}

int EmbeddingEngine::infer(const std::vector<std::vector<int>> &ids,
                           CTensorfl &out) {
  LOG(INFO) << "EmbeddingEngine::infer";
  Timer timer;
  try {
    if (!is_init_) {
      LOG(ERROR) << "infer error;please call loadModel() before";
      return -1;
    }
    int batch_size = ids.size();
    int max_len = 0;
    for (auto &v : ids)
      max_len = std::max(max_len, int(v.size()));
    CTensorii input_ids({batch_size, max_len});
    CTensorii token_type_ids({batch_size, max_len});
    CTensorii attention_mask({batch_size, max_len});

    for (int i = 0; i < ids.size(); i++) {
      for (int j = 0; j < ids[i].size(); j++) {
        input_ids.at({i, j}) = ids[i][j];
        attention_mask.at({i, j}) = 1;
      }
    }

    auto ins = ins_queue.Pop();
    ScopeGuard scopeguard([&]() { ins_queue.Push(std::move(ins)); });

    Dims2 inputdims{batch_size, max_len};
    ins.context->setInputShape(ins.inputNames[0].c_str(), inputdims);
    ins.context->setInputShape(ins.inputNames[1].c_str(), inputdims);
    ins.context->setInputShape(ins.inputNames[2].c_str(), inputdims);
    auto outputdims = ins.context->getTensorShape(ins.outputNames[0].c_str());

    cudaMemcpyAsync(ins.bindingArray[0], input_ids.data(), input_ids.byteSize(),
                    cudaMemcpyHostToDevice, *ins.stream);
    cudaMemcpyAsync(ins.bindingArray[1], token_type_ids.data(), token_type_ids.byteSize(),
                    cudaMemcpyHostToDevice, *ins.stream);
    cudaMemcpyAsync(ins.bindingArray[2], attention_mask.data(), attention_mask.byteSize(),
                    cudaMemcpyHostToDevice, *ins.stream);
    auto statu = ins.context->enqueueV2(ins.bindingArray.data(), *ins.stream,nullptr);
    out.resize({outputdims.d[0], outputdims.d[1]});
    cudaMemcpyAsync(out.data(), ins.bindingArray[3], out.byteSize(),
               cudaMemcpyDeviceToHost, *ins.stream);
    cudaStreamSynchronize(*ins.stream);

    LOG(INFO) << "EmbeddingEngine::infer end.timecost:"<<timer.Elapsed()<<"ms";
    return 0;
  } catch (std::exception e) {
    LOG(ERROR) << e.what();
    return -1;
  }
}

}; // namespace DMAI