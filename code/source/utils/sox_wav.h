#include "sox.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace BASE_NAMESPACE {

/***
使用该头文件需要安装libsox-dev. 如：apt install libsox-dev
另外需要链接libsox.so 
*/
class SoxWavReader {
public:
  explicit SoxWavReader(const std::string &filename) {
    sox_format_t *in;
    assert(sox_init() == SOX_SUCCESS);
    assert(in = sox_open_read(filename.c_str(), NULL, NULL, NULL));
    sample_rate_ = in->signal.rate;
    num_channel_ = in->signal.channels;
    num_sample_ = in->signal.length;
    bits_per_sample_ = in->signal.precision;
    std::vector<sox_sample_t> audio_data(num_sample_);
    data_.resize(num_sample_);
    assert(sox_read(in, audio_data.data(), num_sample_)==num_sample_);
    for(size_t i=0;i<num_sample_;i++){
        data_[i] = SOX_SAMPLE_TO_FLOAT_64BIT(audio_data[i],)*(std::numeric_limits<int16_t>::max()+1.0);
    }
    sox_close(in);
    sox_quit();
  }

  int num_channel() const { return num_channel_; }
  int sample_rate() const { return sample_rate_; }
  int bits_per_sample() const { return bits_per_sample_; }
  int num_sample() const { return num_sample_; }
  std::vector<int16_t> data() { return data_; }

private:
  int num_channel_;
  int sample_rate_;
  int bits_per_sample_;
  long num_sample_; 
  std::vector<int16_t> data_;
};

}; // namespace wenet