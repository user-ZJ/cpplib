#pragma once
#include "sox.h"
#include "utils/logging.h"
#include <algorithm>
#include <cassert>
#include <vector>

namespace BASE_NAMESPACE {

/***
使用该头文件需要安装libsox-dev. 如：apt install libsox-dev
另外需要链接libsox.so
*/

class WavInfo {
 public:
  unsigned int sample_rate;
  unsigned int channel;
  unsigned int sample_num;
  unsigned int precision;
};

class SoxUtil {
 public:
  static SoxUtil &instance() {
    static SoxUtil ins;
    return ins;
  }

  WavInfo GetWavInfo(const std::string &filename) {
    WavInfo info;
    sox_format_t *in;
    assert(in = sox_open_read(filename.c_str(), NULL, NULL, NULL));
    info.sample_rate = in->signal.rate;
    info.channel = in->signal.channels;
    info.sample_num = in->signal.length;
    info.precision = in->signal.precision;
    sox_close(in);
    return info;
  }

  int GetData(const std::string &filename, std::vector<int16_t> *out) {
    sox_format_t *in;
    assert(in = sox_open_read(filename.c_str(), NULL, NULL, NULL));
    std::vector<sox_sample_t> audio_data(in->signal.length);
    out->resize(in->signal.length);
    assert(sox_read(in, audio_data.data(), audio_data.size()) == audio_data.size());
    for (size_t i = 0; i < audio_data.size(); i++) {
      (*out)[i] = SOX_SAMPLE_TO_FLOAT_64BIT(audio_data[i], ) * (std::numeric_limits<int16_t>::max() + 1.0);
    }
    sox_close(in);
    return 0;
  }

  int GetData(const std::string &filename, std::vector<float> *out) {
    sox_format_t *in;
    assert(in = sox_open_read(filename.c_str(), NULL, NULL, NULL));
    std::vector<sox_sample_t> audio_data(in->signal.length);
    out->resize(in->signal.length);
    assert(sox_read(in, audio_data.data(), audio_data.size()) == audio_data.size());
    for (size_t i = 0; i < audio_data.size(); i++) { (*out)[i] = SOX_SAMPLE_TO_FLOAT_64BIT(audio_data[i], ); }
    sox_close(in);
    return 0;
  }

  std::vector<char> ProcessWav(const WavInfo &info, const std::vector<float> &data, const int sample_rate,
                               const float volume, const float speed) {
    sox_format_t *tmp, *in, *out;
    std::vector<sox_sample_t> audio_data(data.size());
    for (int i = 0; i < data.size(); i++) {
      audio_data[i] = SOX_SIGNED_16BIT_TO_SAMPLE(data[i] * (std::numeric_limits<int16_t>::max() + 1.0), );
    }
    sox_signalinfo_t tmp_signal = {static_cast<sox_rate_t>(info.sample_rate), info.channel, info.precision, 0, NULL};
    std::vector<char> buffer(data.size() * 5);
    assert(tmp = sox_open_mem_write(buffer.data(), buffer.size(), &tmp_signal, NULL, "sox", NULL));
    assert(sox_write(tmp, audio_data.data(), audio_data.size()) == audio_data.size());
    sox_close(tmp);

    // effect
    sox_effects_chain_t *chain;
    sox_effect_t *e;
    char *args[10];
    sox_signalinfo_t interm_signal; /* @ intermediate points in the chain. */
    uint64_t tgt_sample_num = 1.0/speed * sample_rate / info.sample_rate * info.sample_num;
    sox_encodinginfo_t out_encoding{SOX_ENCODING_SIGN2,    16,        0, sox_option_default, sox_option_default,
                                    sox_option_default, sox_false};
    sox_signalinfo_t out_signal = {static_cast<sox_rate_t>(sample_rate), 1, 16, tgt_sample_num, NULL};
    assert(in = sox_open_mem_read(buffer.data(), buffer.size(), NULL, NULL, NULL));
    std::vector<char> out_buff(data.size() * 5, 0);
    assert(out =
             sox_open_mem_write(out_buff.data(), out_buff.size(), &out_signal, &out_encoding, "wav", NULL));
    chain = sox_create_effects_chain(&in->encoding, &out->encoding);
    interm_signal = in->signal; /* NB: deep copy */

    e = sox_create_effect(sox_find_effect("input"));
    args[0] = (char *)in, assert(sox_effect_options(e, 1, args) == SOX_SUCCESS);
    assert(sox_add_effect(chain, e, &interm_signal, &in->signal) == SOX_SUCCESS);
    free(e);

    if (volume != 1.0) {
      e = sox_create_effect(sox_find_effect("vol"));
      args[0] = const_cast<char *>(std::to_string(volume).c_str());
      assert(sox_effect_options(e, 1, args) == SOX_SUCCESS);
      /* Add the effect to the end of the effects processing chain: */
      assert(sox_add_effect(chain, e, &interm_signal, &in->signal) == SOX_SUCCESS);
      free(e);
    }

    if (speed != 1.0) {
      e = sox_create_effect(sox_find_effect("speed"));
      args[0] = const_cast<char *>(std::to_string(speed).c_str());
      assert(sox_effect_options(e, 1, args) == SOX_SUCCESS);
      /* Add the effect to the end of the effects processing chain: */
      assert(sox_add_effect(chain, e, &interm_signal, &in->signal) == SOX_SUCCESS);
      free(e);
      e = sox_create_effect(sox_find_effect("rate"));
      assert(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
      assert(sox_add_effect(chain, e, &interm_signal, &out->signal) == SOX_SUCCESS);
      free(e);
    }

    if (in->signal.rate != out->signal.rate) {
      e = sox_create_effect(sox_find_effect("rate"));
      assert(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
      assert(sox_add_effect(chain, e, &interm_signal, &out->signal) == SOX_SUCCESS);
      free(e);
    }

    if (in->signal.channels != out->signal.channels) {
      e = sox_create_effect(sox_find_effect("channels"));
      assert(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
      assert(sox_add_effect(chain, e, &interm_signal, &out->signal) == SOX_SUCCESS);
      free(e);
    }

    e = sox_create_effect(sox_find_effect("output"));
    args[0] = (char *)out, assert(sox_effect_options(e, 1, args) == SOX_SUCCESS);
    assert(sox_add_effect(chain, e, &interm_signal, &out->signal) == SOX_SUCCESS);
    free(e);

    sox_flow_effects(chain, NULL, NULL);
    out_buff.resize(out->tell_off);
    sox_delete_effects_chain(chain);
    sox_close(out);
    sox_close(in);
    
    return out_buff;
  }

  std::vector<char> Wav2Mp3(const std::vector<char> &buffer){
    std::vector<char> result(buffer.size());
    sox_format_t *in, *out;
     sox_encodinginfo_t mp3_encoding{SOX_ENCODING_MP3,    0,        0, sox_option_default, sox_option_default,
                                    sox_option_default, sox_false};
    assert(in = sox_open_mem_read(const_cast<char *>(buffer.data()), buffer.size(), NULL, NULL, NULL));
    assert(out =
             sox_open_mem_write(result.data(), result.size(), &in->signal, &mp3_encoding, "mp3", NULL));
    std::vector<sox_sample_t> audio_data(in->signal.length);
    assert(sox_read(in, audio_data.data(), in->signal.length)==in->signal.length);
    assert(sox_write(out, audio_data.data(), audio_data.size()) == audio_data.size());
    sox_close(out);
    sox_close(in);
    result.resize(out->tell_off);
    return result;
  }

 private:
  SoxUtil() {
    assert(sox_init() == SOX_SUCCESS);
  }
  ~SoxUtil() {
    sox_quit();
  }
};

};  // namespace BASE_NAMESPACE
