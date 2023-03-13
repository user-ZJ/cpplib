#pragma once
extern "C" {
#include "rnnoise.h"
}
#include <memory>
#include <vector>

namespace BASE_NAMESPACE {

class RnnoiseWrapper {
 public:
  RnnoiseWrapper();
  ~RnnoiseWrapper();

  int RemoveNoise(std::vector<char> &buff, const std::string &taskid);
  int RemoveNoise(const std::string &infile, const std::string &outfile, const std::string &taskid);

 private:
  /**
   * @brief 对语音数据进行降噪，只支持48k的语音数据
   *
   * @param data 48k的语音数据
   */
  void process(std::vector<int16_t> &data, const std::string &taskid);

  
  int FRAME_SIZE = 480;
};

}  // namespace BASE_NAMESPACE