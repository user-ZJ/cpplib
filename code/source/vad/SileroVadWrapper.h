/*
 * @Author: zack
 * @Date: 2021-12-23 15:01:19
 * @Last Modified by: zack
 * @Last Modified time: 2021-12-23 18:18:37
 * 使用SileroVad作为静音检测工具
 */
#pragma once
#include <memory>
#include <vector>
#include "engine/onnx/onnx_util.h"

namespace BASE_NAMESPACE {



class SileroVadWrapper {
 public:
  SileroVadWrapper(const std::string ModelPath="../data/silero_vad.onnx", 
             int Sample_rate=16000, int frame_ms=30, 
             float Threshold=0.5, int min_silence_duration_ms=0, int speech_pad_ms=0);
  // 切分语音为语音段
  // 返回的语音区间为左闭右开的集合,如[0,10)
  std::vector<std::pair<size_t, size_t>> SplitAudio(const std::vector<int16_t> &audio, int sample_rate);
  // 移除语音中静音部分
  std::vector<int16_t> RemoveSilence(const std::vector<int16_t> &audio, int sample_rate);

 private:
  // model config
    int64_t window_size_samples;  // Assign when init, support 256 512 768 for 8k; 512 1024 1536 for 16k.
    int sample_rate;
    int sr_per_ms;  // Assign when init, support 8 or 16
    float threshold;
    int min_silence_samples; // sr_per_ms * #ms
    int speech_pad_samples; // usually a 


    std::vector<const char *> input_node_names = {"input", "sr", "h", "c"};
    std::vector<int64_t> sr;
    unsigned int size_hc = 2 * 1 * 64; // It's FIXED.
    int64_t input_node_dims[2] = {}; 
    const int64_t sr_node_dims[1] = {1};
    const int64_t hc_node_dims[3] = {2, 1, 64};
    std::vector<const char *> output_node_names = {"output", "hn", "cn"};

    // OnnxRuntime resources
    std::unique_ptr<Ort::Session> session = nullptr;

};

};  // namespace BASE_NAMESPACE