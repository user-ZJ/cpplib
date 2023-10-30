#include "audio-util.h"
#include "utils/logging.h"
#include <algorithm>
#include <fstream>
#include <mutex>
#include <vector>

namespace BASE_NAMESPACE {

/***
使用该头文件需要安装libsox-dev. 如：apt install libsox-dev
另外需要链接libsox.so
*/

static std::mutex sox_mutex;

SoxUtil::SoxUtil() {
  int res = sox_init();
  if (res != SOX_SUCCESS) { LOG(ERROR) << "sox_init failed."; }
}

SoxUtil::~SoxUtil() {
  sox_quit();
}

int SoxUtil::GetWavInfo(const std::string &filename, WavInfo &info) {
  sox_format_t *in;
  in = sox_open_read(filename.c_str(), NULL, NULL, NULL);
  if (in == nullptr) {
    LOG(ERROR) << "read audio:" << filename << " error";
    return -1;
  }
  info.sample_rate = in->signal.rate;
  info.channel = in->signal.channels;
  info.sample_num = in->signal.length;
  info.precision = in->signal.precision;
  sox_close(in);
  return 0;
}

int SoxUtil::GetWavInfo(const std::vector<char> &buff, WavInfo &info) {
  sox_format_t *in;
  in = sox_open_mem_read(const_cast<char *>(buff.data()), buff.size(), NULL, NULL, NULL);
  if (in == nullptr) {
    LOG(ERROR) << "read audio error";
    return -1;
  }
  info.sample_rate = in->signal.rate;
  info.channel = in->signal.channels;
  info.sample_num = in->signal.length;
  info.precision = in->signal.precision;
  sox_close(in);
  return 0;
}

int SoxUtil::GetData(const std::string &filename, std::vector<int16_t> &out) {
  out.clear();
  sox_format_t *in;
  in = sox_open_read(filename.c_str(), NULL, NULL, NULL);
  if (in == nullptr || in->signal.channels == 0) {
    LOG(ERROR) << "read audio:" << filename << " error";
    return -1;
  }
  int channel = in->signal.channels;
  std::vector<sox_sample_t> audio_data(in->signal.length, 0);
  size_t read_num = sox_read(in, audio_data.data(), audio_data.size());
  if (read_num != audio_data.size()) {
    LOG(WARNING) << "can not get enough audio data:expect " << audio_data.size() << " but got " << read_num;
  }
  if (channel != 1) {
    LOG(WARNING) << "only read channel 0 data";
    read_num /= channel;  // 仅读第一个通道数据
  }
  out.resize(read_num);
  for (size_t i = 0; i < read_num; i++) {
    out[i] = SOX_SAMPLE_TO_FLOAT_64BIT(audio_data[i * channel], ) * (std::numeric_limits<int16_t>::max() + 1.0);
  }
  sox_close(in);
  return 0;
}

int SoxUtil::GetData(const std::string &filename, std::vector<float> &out){
  std::vector<double> o;
  int res = GetData(filename,o);
  out.resize(o.size());
  std::copy_n(o.begin(),o.size(),out.begin());
  return res;
}

int SoxUtil::GetData(const std::string &filename, std::vector<double> &out) {
  out.clear();
  sox_format_t *in = sox_open_read(filename.c_str(), NULL, NULL, NULL);
  if (in == nullptr || in->signal.channels == 0) {
    LOG(ERROR) << "read audio:" << filename << " error";
    return -1;
  }
  int channel = in->signal.channels;
  std::vector<sox_sample_t> audio_data(in->signal.length, 0);
  size_t read_num = sox_read(in, audio_data.data(), audio_data.size());
  if (read_num != audio_data.size()) {
    LOG(WARNING) << "can not get enough audio data:expect " << audio_data.size() << " but got " << read_num;
  }
  if (channel != 1) {
    LOG(WARNING) << "only read channel 0 data";
    read_num /= channel;  // 仅读第一个通道数据
  }
  out.resize(read_num);
  for (size_t i = 0; i < read_num; i++) {
    out[i] = SOX_SAMPLE_TO_FLOAT_64BIT(audio_data[i * channel], );
  }
  sox_close(in);
  return 0;
}

int SoxUtil::GetData(const std::vector<char> &buff, std::vector<int16_t> &out) {
  out.clear();
  sox_format_t *in = sox_open_mem_read(const_cast<char *>(buff.data()), buff.size(), NULL, NULL, NULL);
  if (in == nullptr || in->signal.channels == 0) {
    LOG(ERROR) << "read audio from buffer error";
    return -1;
  }
  int channel = in->signal.channels;
  std::vector<sox_sample_t> audio_data(in->signal.length, 0);
  size_t read_num = sox_read(in, audio_data.data(), audio_data.size());
  if (read_num != audio_data.size()) {
    LOG(WARNING) << "can not get enough audio data:expect " << audio_data.size() << " but got " << read_num;
  }
  LOG(INFO) << "read buff:sample_rate " << in->signal.rate << " channel " << in->signal.channels << " sample_num "
            << read_num;
  if (channel != 1) {
    LOG(WARNING) << "only read channel 0 data";
    read_num /= channel;  // 仅读第一个通道数据
  }
  out.resize(read_num);
  for (size_t i = 0; i < read_num; i++) {
    out[i] = SOX_SAMPLE_TO_FLOAT_64BIT(audio_data[i * channel], ) * (std::numeric_limits<int16_t>::max() + 1.0);
  }
  sox_close(in);
  return 0;
}

int SoxUtil::GetData(const std::vector<char> &buff, std::vector<float> &out){
  std::vector<double> o;
  int res = GetData(buff,o);
  out.resize(o.size());
  std::copy_n(o.begin(),o.size(),out.begin());
  return res;
}

int SoxUtil::GetData(const std::vector<char> &buff, std::vector<double> &out) {
  out.clear();
  sox_format_t *in = sox_open_mem_read(const_cast<char *>(buff.data()), buff.size(), NULL, NULL, NULL);
  if (in == nullptr || in->signal.channels == 0) {
    LOG(ERROR) << "read audio from buffer error";
    return -1;
  }
  int channel = in->signal.channels;
  std::vector<sox_sample_t> audio_data(in->signal.length, 0);
  size_t read_num = sox_read(in, audio_data.data(), audio_data.size());
  if (read_num != audio_data.size()) {
    LOG(WARNING) << "can not get enough audio data:expect " << audio_data.size() << " but got " << read_num;
  }
  if (channel != 1) {
    LOG(WARNING) << "only read channel 0 data";
    read_num /= channel;  // 仅读第一个通道数据
  }
  out.resize(read_num);
  for (size_t i = 0; i < read_num; i++) {
    out[i] = SOX_SAMPLE_TO_FLOAT_64BIT(audio_data[i * channel], );
  }
  sox_close(in);
  return 0;
}

std::vector<char> SoxUtil::Write2Buff(const WavInfo &info, const std::vector<sox_sample_t> &data) {
  try {
    std::vector<char> buffer(data.size() * 5);
    sox_signalinfo_t out_signal = {static_cast<sox_rate_t>(info.sample_rate), info.channel, info.precision, data.size(),
                                   NULL};
    sox_format_t *out = sox_open_mem_write(buffer.data(), buffer.size(), &out_signal, NULL, "wav", NULL);
    if (out == nullptr) {
      LOG(ERROR) << "Opens an encoding session error";
      return {};
    }
    size_t len = sox_write(out, data.data(), data.size());
    if (len != data.size()) {
      LOG(ERROR) << "write audio error";
      return {};
    }
    buffer.resize(out->tell_off);
    sox_close(out);
    return buffer;
  }
  catch (...) {
    LOG(ERROR) << "write audio buff error";
    return {};
  }
}

std::vector<char> SoxUtil::Write2Buff(const WavInfo &info, const std::vector<float> &data) {
  std::vector<sox_sample_t> audio_data(data.size());
  for (int i = 0; i < data.size(); i++) {
    audio_data[i] = SOX_SIGNED_16BIT_TO_SAMPLE(data[i] * (std::numeric_limits<int16_t>::max() + 1.0), );
  }
  return Write2Buff(info, audio_data);
}

std::vector<char> SoxUtil::Write2Buff(const WavInfo &info, const std::vector<double> &data) {
  std::vector<sox_sample_t> audio_data(data.size());
  for (int i = 0; i < data.size(); i++) {
    audio_data[i] = SOX_SIGNED_16BIT_TO_SAMPLE(data[i] * (std::numeric_limits<int16_t>::max() + 1.0), );
  }
  return Write2Buff(info, audio_data);
}

std::vector<char> SoxUtil::Write2Buff(const WavInfo &info, const std::vector<int16_t> &data) {
  std::vector<sox_sample_t> audio_data(data.size());
  for (int i = 0; i < data.size(); i++) {
    audio_data[i] = SOX_SIGNED_16BIT_TO_SAMPLE(data[i], );
  }
  return Write2Buff(info, audio_data);
}

int SoxUtil::Write2File(const WavInfo &info, const std::vector<sox_sample_t> &data, const char *filepath) {
  try {
    sox_signalinfo_t out_signal = {static_cast<sox_rate_t>(info.sample_rate), info.channel, info.precision, 0, NULL};
    sox_format_t *out = sox_open_write(filepath, &out_signal, NULL, NULL, NULL, NULL);
    if (out == nullptr) {
      LOG(ERROR) << "Opens an encoding session error";
      return -1;
    }
    size_t len = sox_write(out, data.data(), data.size());
    if (len != data.size()) {
      LOG(ERROR) << "write audio error";
      return -1;
    }
    sox_close(out);
    return 0;
  }
  catch (...) {
    LOG(ERROR) << "write audio buff error";
    return -1;
  }
}

int SoxUtil::Write2File(const WavInfo &info, const std::vector<float> &data, const char *filepath) {
  std::vector<sox_sample_t> audio_data(data.size());
  for (int i = 0; i < data.size(); i++) {
    audio_data[i] = SOX_SIGNED_16BIT_TO_SAMPLE(data[i] * (std::numeric_limits<int16_t>::max() + 1.0), );
  }
  return Write2File(info, audio_data, filepath);
}

int SoxUtil::Write2File(const WavInfo &info, const std::vector<double> &data, const char *filepath) {
  std::vector<sox_sample_t> audio_data(data.size());
  for (int i = 0; i < data.size(); i++) {
    audio_data[i] = SOX_SIGNED_16BIT_TO_SAMPLE(data[i] * (std::numeric_limits<int16_t>::max() + 1.0), );
  }
  return Write2File(info, audio_data, filepath);
}

int SoxUtil::Write2File(const WavInfo &info, const std::vector<int16_t> &data, const char *filepath) {
  std::vector<sox_sample_t> audio_data(data.size());
  for (int i = 0; i < data.size(); i++) {
    audio_data[i] = SOX_SIGNED_16BIT_TO_SAMPLE(data[i], );
  }
  return Write2File(info, audio_data, filepath);
}

std::vector<char> SoxUtil::ProcessWav(const WavInfo &info, const std::vector<float> &data, const int sample_rate,
                                      const float volume, const float speed) {
  std::vector<sox_sample_t> audio_data(data.size());
  for (int i = 0; i < data.size(); i++) {
    audio_data[i] = SOX_SIGNED_16BIT_TO_SAMPLE(data[i] * (std::numeric_limits<int16_t>::max() + 1.0), );
  }
  return ProcessWav(info, audio_data, sample_rate, volume, speed);
}

std::vector<char> SoxUtil::ProcessWav(const WavInfo &info, const std::vector<double> &data, const int sample_rate,
                                      const float volume, const float speed) {
  std::vector<sox_sample_t> audio_data(data.size());
  for (int i = 0; i < data.size(); i++) {
    audio_data[i] = SOX_SIGNED_16BIT_TO_SAMPLE(data[i] * (std::numeric_limits<int16_t>::max() + 1.0), );
  }
  return ProcessWav(info, audio_data, sample_rate, volume, speed);
}

std::vector<char> SoxUtil::ProcessWav(const WavInfo &info, const std::vector<int16_t> &data, const int sample_rate,
                                      const float volume, const float speed) {
  std::vector<sox_sample_t> audio_data(data.size());
  for (int i = 0; i < data.size(); i++) {
    audio_data[i] = SOX_SIGNED_16BIT_TO_SAMPLE(data[i], );
  }
  return ProcessWav(info, audio_data, sample_rate, volume, speed);
}

std::vector<char> SoxUtil::ProcessWav(const WavInfo &info, const std::vector<sox_sample_t> &audio_data,
                                      const int sample_rate, const float volume, const float speed) {
  auto buff = Write2Buff(info, audio_data);
  return ProcessWav(buff, sample_rate, volume, speed);
}

std::vector<char> SoxUtil::ProcessWav(const std::vector<char> &buffer, const int sample_rate, const float volume,
                                      const float speed) {
  sox_format_t *in, *out;
  WavInfo info;
  int res = GetWavInfo(buffer, info);
  if (res != 0) {
    LOG(ERROR) << "read audio error";
    return {};
  }
  // effect
  sox_effects_chain_t *chain;
  sox_effect_t *e;
  char *args[10];
  sox_signalinfo_t interm_signal; /* @ intermediate points in the chain. */
  uint64_t tgt_sample_num = 1.0 / speed * sample_rate / info.sample_rate * info.sample_num;
  sox_encodinginfo_t out_encoding{SOX_ENCODING_SIGN2, 16,       0, sox_option_default, sox_option_default,
                                  sox_option_default, sox_false};
  sox_signalinfo_t out_signal = {static_cast<sox_rate_t>(sample_rate), 1, 16, tgt_sample_num, NULL};
  in = sox_open_mem_read(const_cast<char *>(buffer.data()), buffer.size(), NULL, NULL, NULL);
  if (in == nullptr) {
    LOG(ERROR) << "read audio error";
    return {};
  }
  std::vector<char> out_buff(static_cast<size_t>(buffer.size() / speed *sample_rate/info.sample_rate + 1), 0);
  out = sox_open_mem_write(out_buff.data(), out_buff.size(), &out_signal, &out_encoding, "wav", NULL);
  if (out == nullptr) {
    LOG(ERROR) << "write audio buffer error";
    return {};
  }
  chain = sox_create_effects_chain(&in->encoding, &out->encoding);
  interm_signal = in->signal; /* NB: deep copy */

  e = sox_create_effect(sox_find_effect("input"));
  args[0] = (char *)in;
  if (sox_effect_options(e, 1, args) != SOX_SUCCESS) {
    LOG(ERROR) << "creat effect option error";
    return {};
  }
  if (sox_add_effect(chain, e, &interm_signal, &in->signal) != SOX_SUCCESS) {
    LOG(ERROR) << "add effect error";
    return {};
  }
  free(e);

  if (volume != 1.0) {
    e = sox_create_effect(sox_find_effect("vol"));
    args[0] = const_cast<char *>(std::to_string(volume).c_str());
    if (sox_effect_options(e, 1, args) != SOX_SUCCESS) {
      LOG(ERROR) << "creat effect option error";
      return {};
    }
    /* Add the effect to the end of the effects processing chain: */
    if (sox_add_effect(chain, e, &interm_signal, &in->signal) != SOX_SUCCESS) {
      LOG(ERROR) << "add effect error";
      return {};
    }
    free(e);
  }

  if (speed != 1.0) {
    e = sox_create_effect(sox_find_effect("speed"));
    args[0] = const_cast<char *>(std::to_string(speed).c_str());
    if (sox_effect_options(e, 1, args) != SOX_SUCCESS) {
      LOG(ERROR) << "creat effect option error";
      return {};
    }
    /* Add the effect to the end of the effects processing chain: */
    if (sox_add_effect(chain, e, &interm_signal, &in->signal) != SOX_SUCCESS) {
      LOG(ERROR) << "add effect error";
      return {};
    }
    free(e);
    e = sox_create_effect(sox_find_effect("rate"));
    if (sox_effect_options(e, 0, NULL) != SOX_SUCCESS) {
      LOG(ERROR) << "creat effect option error";
      return {};
    }
    if (sox_add_effect(chain, e, &interm_signal, &out->signal) != SOX_SUCCESS) {
      LOG(ERROR) << "add effect error";
      return {};
    }
    free(e);
  }

  if (in->signal.rate != out->signal.rate) {
    e = sox_create_effect(sox_find_effect("rate"));
    if (sox_effect_options(e, 0, NULL) != SOX_SUCCESS) {
      LOG(ERROR) << "creat effect option error";
      return {};
    }
    if (sox_add_effect(chain, e, &interm_signal, &out->signal) != SOX_SUCCESS) {
      LOG(ERROR) << "creat effect option error";
      return {};
    }
    free(e);
  }

  if (in->signal.channels != out->signal.channels) {
    e = sox_create_effect(sox_find_effect("channels"));
    if (sox_effect_options(e, 0, NULL) != SOX_SUCCESS) {
      LOG(ERROR) << "creat effect option error";
      return {};
    }
    if (sox_add_effect(chain, e, &interm_signal, &out->signal) != SOX_SUCCESS) {
      LOG(ERROR) << "creat effect option error";
      return {};
    }
    free(e);
  }

  e = sox_create_effect(sox_find_effect("output"));
  args[0] = (char *)out;
  if (sox_effect_options(e, 1, args) != SOX_SUCCESS) {
    LOG(ERROR) << "creat effect option error";
    return {};
  }
  if (sox_add_effect(chain, e, &interm_signal, &out->signal) != SOX_SUCCESS) {
    LOG(ERROR) << "creat effect option error";
    return {};
  }
  free(e);

  sox_flow_effects(chain, NULL, NULL);
  out_buff.resize(out->tell_off);
  sox_delete_effects_chain(chain);
  sox_close(out);
  sox_close(in);

  return out_buff;
}

std::vector<char> SoxUtil::Wav2Mp3(const std::vector<char> &buffer) {
  std::vector<char> result(buffer.size());
  sox_format_t *in, *out;
  sox_encodinginfo_t mp3_encoding{SOX_ENCODING_MP3,   0,        0, sox_option_default, sox_option_default,
                                  sox_option_default, sox_false};
  in = sox_open_mem_read(const_cast<char *>(buffer.data()), buffer.size(), NULL, NULL, NULL);
  if (in == nullptr) {
    LOG(ERROR) << "read audio error";
    return {};
  }
  out = sox_open_mem_write(result.data(), result.size(), &in->signal, &mp3_encoding, "mp3", NULL);
  if (out == nullptr) {
    LOG(ERROR) << "write audio buffer error";
    return {};
  }
  int channel = in->signal.channels;
  std::vector<sox_sample_t> audio_data(in->signal.length);
  size_t read_num = sox_read(in, audio_data.data(), in->signal.length);
  if (read_num != audio_data.size()) {
    LOG(WARNING) << "can not get enough audio data:expect " << audio_data.size() << " but got " << read_num;
  }
  size_t write_num = sox_write(out, audio_data.data(), audio_data.size());
  if (write_num != audio_data.size()) {
    LOG(WARNING) << "not write all data:expect " << audio_data.size() << " but got " << write_num;
  }
  sox_close(out);
  sox_close(in);
  result.resize(out->tell_off);
  return result;
}

std::vector<char> SoxUtil::Mp3ToWav(const std::vector<char> &buffer) {
  std::vector<char> result(buffer.size() * 15);
  sox_format_t *in, *out;
  {
    sox_encodinginfo_t mp3_encoding{SOX_ENCODING_MP3,   0,        0, sox_option_default, sox_option_default,
                                    sox_option_default, sox_false};
    in = sox_open_mem_read(const_cast<char *>(buffer.data()), buffer.size(), NULL, &mp3_encoding, "mp3");
    in->signal.length = buffer.size() * 6;
    if (in == nullptr) {
      LOG(ERROR) << "read audio error";
      return {};
    }

    std::vector<sox_sample_t> audio_data(in->signal.length);
    size_t read_num = sox_read(in, audio_data.data(), in->signal.length);
    if (read_num != audio_data.size()) {
      in->signal.length = read_num;
      audio_data.resize(read_num);
    }

    out = sox_open_mem_write(result.data(), result.size(), &in->signal, NULL, "sox", NULL);
    if (out == nullptr) {
      LOG(ERROR) << "write audio buffer error";
      return {};
    }
    size_t write_num = sox_write(out, audio_data.data(), audio_data.size());
    if (write_num != audio_data.size()) {
      LOG(WARNING) << "not write all data:expect " << audio_data.size() << " but got " << write_num;
    }
  }
  sox_close(out);
  sox_close(in);
  result.resize(out->tell_off);
  return result;
}

namespace SonicUtil {

std::vector<int16_t> process(const WavInfo &info, const std::vector<int16_t> &data, const float volume,
                                  const float speed) {
  LOG(INFO) << "sonic process volume:" << volume << " speed:" << speed;
  std::vector<int16_t> result;
  if (data.empty()) {
    LOG(ERROR) << "audio data is empty";
    return result;
  }
  sonicStream stream = sonicCreateStream(info.sample_rate, info.channel);
  if (stream == nullptr) {
    LOG(ERROR) << "create sonic stream error";
    return result;
  }
  sonicSetSpeed(stream, speed);
  sonicSetPitch(stream, 1.0);
  sonicSetRate(stream, 1.0);
  sonicSetVolume(stream, volume);
  sonicSetChordPitch(stream, 0);
  sonicSetQuality(stream, 0);

  static int BUFFER_SIZE = 2048;

  int sample_index = 0, sample_out = 0;
  auto input_len = data.size();
  std::vector<std::int16_t> buffer(BUFFER_SIZE, 0);
  do {
    if (sample_index >= input_len) {
      sonicFlushStream(stream);
    } else if (sample_index + BUFFER_SIZE > input_len) {
      sonicWriteShortToStream(stream, const_cast<short *>(&data[sample_index]), input_len - sample_index);
      sample_index += (input_len - sample_index);
    } else {
      sonicWriteShortToStream(stream, const_cast<short *>(&data[sample_index]), BUFFER_SIZE);
      sample_index += BUFFER_SIZE;
    }
    do {
      sample_out = sonicReadShortFromStream(stream, buffer.data(), BUFFER_SIZE);
      if (sample_out > 0) { result.insert(result.end(), buffer.begin(), buffer.begin() + sample_out); }
    } while (sample_out > 0);
  } while (sample_index < input_len);
  sonicDestroyStream(stream);
  LOG(INFO) << "sonic process end";
  return result;
}

}  // namespace SonicUtil
};  // namespace BASE_NAMESPACE
