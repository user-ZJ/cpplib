/*
 * @Author: zack
 * @Date: 2021-12-23 15:01:19
 * @Last Modified by: zack
 * @Last Modified time: 2021-12-23 18:18:37
 * 使用fvad作为静音检测工具
 */
#pragma once
#include "IVad.h"
#include <fvad.h>
#include <memory>


namespace BASE_NAMESPACE {

struct vad_deleter {
  void operator()(Fvad *vad) {
    if (vad) fvad_free(vad);
  }
};

class FVadWrapper : public IVad {
 public:
  ~FVadWrapper() {
  }
  // 初始化Vad.
  int Init(const std::string &path);
  // 切分语音为语音段
  // 返回的语音区间为左闭右开的集合,如[0,10)
  std::vector<std::pair<size_t, size_t>> SplitAudio(const std::vector<int16_t> &audio, int sample_rate);
  // 移除语音中静音部分
  std::vector<int16_t> RemoveSilence(const std::vector<int16_t> &audio, int sample_rate);

 private:
  std::shared_ptr<Fvad> vad;
  int mode;         // vad模式
  int frame_ms;     // 每帧长度
  int num_padding;  // 平滑窗口帧数
  bool is_init;
};

};  // namespace BASE_NAMESPACE