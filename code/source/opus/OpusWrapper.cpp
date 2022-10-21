#include "OpusWrapper.h"
#include "opus.h"
#include "utils/logging.h"

namespace BASE_NAMESPACE {

static void int_to_char(unsigned int i, std::vector<unsigned char> &ch) {
  ch.resize(4);
  ch[0] = i >> 24;
  ch[1] = (i >> 16) & 0xFF;
  ch[2] = (i >> 8) & 0xFF;
  ch[3] = i & 0xFF;
}

static opus_uint32 char_to_int(unsigned char ch[4]) {
  return ((opus_uint32)ch[0] << 24) | ((opus_uint32)ch[1] << 16) | ((opus_uint32)ch[2] << 8) | (opus_uint32)ch[3];
}

int OpusEncode(const std::vector<short> &audiodata, int sampleRate, std::vector<unsigned char> &buff) {
  // init encoder
  OpusEncoder *enc = nullptr;
  int error;
  int size;
  int channels = 1;
  int frame_size = sampleRate * 0.02;  // 20ms

  enc = opus_encoder_create(sampleRate, channels, OPUS_APPLICATION_AUDIO, &error);
  if (error != OPUS_OK) {
    LOG(INFO) << "Cannot create encoder: " << opus_strerror(error);
    free(enc);
    return error;
  }

  buff.clear();
  int max_payload_bytes = 1500;
  std::vector<unsigned char> chunk_out(max_payload_bytes);
  std::vector<short> chunk_in(frame_size);
  std::vector<unsigned char> len_buff(4);
  unsigned int enc_final_range;

  long offset = 0;
  while (offset < audiodata.size()) {
    memset(chunk_in.data(), 0, chunk_in.size() * sizeof(short));
    int chunk_size = offset + frame_size < audiodata.size() ? frame_size : audiodata.size() - offset;
    memcpy(chunk_in.data(), &audiodata[offset], chunk_size * sizeof(short));
    int len = opus_encode(enc, chunk_in.data(), frame_size, chunk_out.data(), chunk_out.size());
    // 写len
    int_to_char(len,len_buff);
    buff.insert(buff.end(),len_buff.begin(),len_buff.end());
    // 写enc_final_range
    opus_encoder_ctl(enc, OPUS_GET_FINAL_RANGE(&enc_final_range));
    int_to_char(enc_final_range,len_buff);
    buff.insert(buff.end(),len_buff.begin(),len_buff.end());
    buff.insert(buff.end(), chunk_out.begin(), chunk_out.begin() + len);
    offset += frame_size;
  }
  opus_encoder_destroy(enc);
  return 0;
}

int OpusDecode(const std::vector<unsigned char> &opusdata, int sampleRate, std::vector<short> &buff) {
  int err;
  int channels = 1;
  OpusDecoder *dec = nullptr;
  dec = opus_decoder_create(sampleRate, channels, &err);
  if (err != OPUS_OK) {
    LOG(ERROR) << "Cannot create decoder:" << opus_strerror(err);
    return err;
  }
  buff.clear();
  unsigned int enc_final_range;
  long offset = 0;
  unsigned char ch[4];
  int max_payload_bytes = 1500;
  std::vector<unsigned char> chunk_in(max_payload_bytes);
  std::vector<short> chunk_out(48000 * 2);
  while (offset < opusdata.size()) {
    // 获取一帧编码后的长度
    memcpy(ch, &opusdata[offset], 4);
    offset += 4;
    int len = char_to_int(ch);  // 4byte->int,单帧数据长度（负载）
    LOG(INFO) << "len:" << len;
    // 读enc_final_range
    memcpy(ch, &opusdata[offset], 4);
    offset += 4;
    enc_final_range = char_to_int(ch);
    LOG(INFO) << "enc_final_range:" << enc_final_range;
    // 读语音数据
    int num_read = offset + len <= opusdata.size() ? len : opusdata.size() - offset;
    LOG(INFO)<<"num_read:"<<num_read;
    memset(chunk_in.data(), 0, chunk_in.size());
    memcpy(chunk_in.data(), &opusdata[offset], num_read);
    int output_samples = opus_decode(dec, chunk_in.data(), num_read, chunk_out.data(), chunk_out.size(), 0);
    LOG(INFO) << "output_samples:" << output_samples;
    buff.insert(buff.end(), chunk_out.begin(), chunk_out.begin() + output_samples);
    offset += num_read;
  }
  opus_decoder_destroy(dec);
  return 0;
}

}  // namespace BASE_NAMESPACE