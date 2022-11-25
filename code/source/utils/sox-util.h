#ifndef BASE_SOX_UTIL_H_
#define BASE_SOX_UTIL_H_
#include "sox.h"
#include "utils/logging.h"
#include <algorithm>
#include <vector>

namespace BASE_NAMESPACE {

/***
使用该头文件需要安装libsox-dev. 如：apt install libsox-dev
另外需要链接libsox.so
*/

class WavInfo {
 public:
  unsigned int sample_rate = 0;
  unsigned int channel = 0;
  unsigned int sample_num = 0;
  unsigned int precision = 0;
};

class SoxUtil {
 public:
  static SoxUtil &instance() {
    static SoxUtil ins;
    return ins;
  }
  // 获取音频信息，成功返回0
  int GetWavInfo(const std::string &filename, WavInfo &info);
  int GetWavInfo(const std::vector<char> &buff, WavInfo &info);
  // 获取音频数据，成功返回0
  int GetData(const std::string &filename, std::vector<int16_t> &out);
  int GetData(const std::string &filename, std::vector<double> &out);
  int GetData(const std::vector<char> &buff, std::vector<int16_t> &out);
  int GetData(const std::vector<char> &buff, std::vector<double> &out);
  // 将音频写入buff
  std::vector<char> Write2Buff(const WavInfo &info, const std::vector<sox_sample_t> &data);
  std::vector<char> Write2Buff(const WavInfo &info, const std::vector<double> &data);
  std::vector<char> Write2Buff(const WavInfo &info, const std::vector<int16_t> &data);
  // 将音频写入文件
  int Write2File(const WavInfo &info, const std::vector<sox_sample_t> &data, const char *filepath);
  int Write2File(const WavInfo &info, const std::vector<double> &data, const char *filepath);
  int Write2File(const WavInfo &info, const std::vector<int16_t> &data, const char *filepath);
  // 处理音频，采样率，语速，音量
  std::vector<char> ProcessWav(const WavInfo &info, const std::vector<double> &data, const int sample_rate,
                               const float volume, const float speed);
  std::vector<char> ProcessWav(const WavInfo &info, const std::vector<int16_t> &data, const int sample_rate,
                               const float volume, const float speed);
  std::vector<char> ProcessWav(const WavInfo &info, const std::vector<sox_sample_t> &audio_data, const int sample_rate,
                               const float volume, const float speed);
  std::vector<char> ProcessWav(const std::vector<char> &buffer, const int sample_rate, const float volume,
                               const float speed);
  // wav转为mp3
  std::vector<char> Wav2Mp3(const std::vector<char> &buffer);
  std::vector<char> Mp3ToWav(const std::vector<char> &buffer);

 private:
  SoxUtil();
  ~SoxUtil();
};

};  // namespace BASE_NAMESPACE

#endif
