/*
 * @Author: zack
 * @Date: 2021-12-23 15:11:11
 * @Last Modified by: zack
 * @Last Modified time: 2021-12-23 18:23:53
 */

#include "FVadWrapper.h"
#include "utils/logging.h"
#include <numeric>


namespace BASE_NAMESPACE {

int FVadWrapper::Init(const std::string &path) {
  mode = 1;
  frame_ms = 30;  // duration is 20ms
  num_padding = 10;
  is_init = false;
  vad.reset(fvad_new(),vad_deleter());
  if (vad && fvad_set_mode(vad.get(), mode) == 0) {
    is_init = true;
    return 0;
  } else
    return -1;
}

std::vector<std::pair<size_t, size_t>> FVadWrapper::SplitAudio(const std::vector<int16_t> &audio, int sample_rate) {
  LOG(INFO)<<"fvad recive "<<audio.size()<<" samples with sample_rate "<<sample_rate;
  std::vector<std::pair<size_t, size_t>> results;
  if (!is_init) {
    LOG(ERROR) << "error! please init FVadWrapper first";
    return results;
  }
  if (fvad_set_sample_rate(vad.get(), sample_rate) < 0) {
    LOG(ERROR) << "invalid sample rate:" << sample_rate;
    return results;
  }

  std::vector<int> is_voice;
  size_t framelen = (size_t)(sample_rate / 1000.0 * frame_ms);
  size_t offset = 0;
  while (offset + framelen <= audio.size()) {
    int vadres = fvad_process(vad.get(), &audio[offset], framelen);
    vadres = !!vadres;  // make sure it is 0 or 1
    is_voice.push_back(vadres);
    offset += framelen;
  }
  // 使用滑动窗口对静音结果做一次平滑
  {
    int threshold = int(0.8 * num_padding);
    // 如果语音太短，小于窗口长度，直接返回
    if (is_voice.size() <= num_padding) {
      results.push_back({0, is_voice.size() - 1});
      return results;
    }
    size_t window_start = 0;
    size_t window_end = window_start + num_padding;
    size_t start, end;       // 音频的起始和结束的索引
    bool triggered = false;  // 是否检测到语音
    // 滑窗
    for (size_t i = 0; i < is_voice.size() - num_padding; i++) {
      window_start = i;
      window_end = window_start + num_padding;
      int sum = std::accumulate(is_voice.begin() + window_start, is_voice.begin() + window_end, 0);
      if (false == triggered && sum >= threshold) {  // 语音开始
        start = window_start * framelen;
        triggered = true;
      } else if (true == triggered && sum < threshold) {  // 语音结束
        end = window_end * framelen;
        triggered = false;
        i = window_end;
        // results.push_back({start, end});
        // 判断句子是否超过MAX_DURATION长度,如果超过MAX_DURATION，则切分为等长语音
        size_t part_len = end - start;
        size_t seg_num = part_len /(MAX_DURATION*sample_rate) + 1;
        size_t seg_len = part_len / seg_num;
        LOG(INFO)<<part_len<<" "<<seg_num<<" "<<seg_len<<" "<<MAX_DURATION;
        while (start + seg_len * 2 < part_len) {  // 如果有两个及以上的segment,则将第一个segment添加到列表
          results.push_back({start, start + seg_len});
          start += seg_len;
        }
        results.push_back({start, end}); //最后一个segment要包含未均分的数据
        LOG(INFO)<<"vad segment:"<<start<<","<<end;
      }
    }
  }
  return results;
}

std::vector<int16_t> FVadWrapper::RemoveSilence(const std::vector<int16_t> &audio, int sample_rate) {
  std::vector<int16_t> results;
  LOG(INFO)<<"fvad recive "<<audio.size()<<" samples with sample_rate "<<sample_rate;
  if (!is_init) {
    LOG(ERROR) << "error! please init FVadWrapper first";
    return results;
  }
  if (fvad_set_sample_rate(vad.get(), sample_rate) < 0) {
    LOG(ERROR) << "invalid sample rate:" << sample_rate;
    return results;
  }
  // 识别每帧语音是否是静音
  std::vector<int> is_voice;
  size_t framelen = (size_t)(sample_rate / 1000.0 * frame_ms);
  size_t offset = 0;
  while (offset + framelen <= audio.size()) {
    int vadres = fvad_process(vad.get(), &audio[offset], framelen);
    vadres = !!vadres;  // make sure it is 0 or 1
    is_voice.push_back(vadres);
    offset += framelen;
  }
  // 使用滑动窗口对静音结果做一次平滑
  {
    int threshold = int(0.8 * num_padding);
    // 如果语音太短，小于窗口长度，直接返回
    if (is_voice.size() <= num_padding) {
      return audio;
    }
    size_t window_start = 0;
    size_t window_end = window_start + num_padding;
    size_t start, end;       // 音频的起始和结束的索引
    bool triggered = false;  // 是否检测到语音
    // 滑窗
    for (size_t i = 0; i < is_voice.size() - num_padding; i++) {
      window_start = i;
      window_end = window_start + num_padding;
      int sum = std::accumulate(is_voice.begin() + window_start, is_voice.begin() + window_end, 0);
      if (false == triggered && sum >= threshold) {  // 语音开始
        start = window_start * framelen;
        triggered = true;
      } else if (true == triggered && sum < threshold) {  // 语音结束
        end = window_end * framelen;
        triggered = false;
        i = window_end;
        // append audio data to results
        for(size_t j=start;j<end;j++){
          results.push_back(audio[j]);
        }
      }
    }
  }
  return results;
}

};  // namespace BASE_NAMESPACE