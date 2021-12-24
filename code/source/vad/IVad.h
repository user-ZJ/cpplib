/*
 * @Author: zack
 * @Date: 2021-12-23 14:36:36
 * @Last Modified by: zack
 * @Last Modified time: 2021-12-23 16:19:32
 * vad接口类
 */
#pragma once
#include <string>
#include <vector>

namespace BASE_NAMESPACE {

#define MAX_DURATION 180  //使用vad断句，每个句子最长180s

class IVad {
 public:
  // 初始化Vad.
  // 有些VAD需要预先加载VAD模型
  virtual int Init(const std::string &path) = 0;
  // 切分语音为语音段
  virtual std::vector<std::pair<size_t, size_t>> SplitAudio(const std::vector<int16_t> &audio,
                                                            int sample_rate) = 0;
  // 移除语音中静音部分
  virtual std::vector<int16_t> RemoveSilence(const std::vector<int16_t> &audio, int sample_rate) = 0;
};

};  // namespace BASE_NAMESPACE
