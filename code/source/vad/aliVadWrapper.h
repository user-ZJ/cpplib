/*
 * @Author: zack
 * @Date: 2021-12-23 15:01:19
 * @Last Modified by: zack
 * @Last Modified time: 2021-12-23 18:18:37
 * 使用aliVad作为静音检测工具
 */
#pragma once
#include "engine/onnx/onnx_util.h"
#include "kaldi/feat/kaldi-featlibs.h"
#include "e2e-vad.h"
#include <memory>
#include <vector>

namespace BASE_NAMESPACE {


class AliVadEngine {
 public:
  AliVadEngine(const std::string ModelPath = "../data/model/ali_vad.onnx",const std::string CmvnPath="../data/model/ali_vad_cmvn");
  void Infer(const std::vector<std::vector<float>> &chunk_feats, std::vector<std::vector<float>> *out_prob,
             std::vector<std::vector<float>> *in_cache);
  AliVadEngine(const AliVadEngine &) = delete;
  AliVadEngine(AliVadEngine &&) = delete;
  AliVadEngine &operator=(const AliVadEngine &) = delete;
  AliVadEngine &operator=(AliVadEngine &&) = delete;
  std::vector<float> GetMeans() { return means; }
  std::vector<float> GetVars() { return vars; }

 private:
  std::vector<const char *> input_node_names = {"speech", "in_cache0", "in_cache1", "in_cache2", "in_cache3"};
  std::vector<const char *> output_node_names = {"logits", "out_cache0", "out_cache1", "out_cache2", "out_cache3"};

  // OnnxRuntime resources
  std::unique_ptr<Ort::Session> session = nullptr;
  std::vector<float> means, vars;
};

class AliVadWrapper {
 public:
  AliVadWrapper();
  // 切分语音为语音段
  // 返回结果单位为ms
  std::vector<std::pair<long, long>> SplitAudio(const std::vector<int16_t> &audio);
  // 移除语音中静音部分
  std::vector<int16_t> RemoveSilence(const std::vector<int16_t> &audio);
  std::vector<std::vector<int>> AcceptAudio(const std::vector<int16_t> &audio, bool input_finished) ;
  // std::vector<std::vector<int>> GetSegments();
  void ExtractFeats(std::vector<std::vector<float>> &vad_feats, std::vector<float> &waves,bool input_finished);
  std::vector<std::vector<float>> LfrCmvn(std::vector<std::vector<float>> &vad_feats);
  int OnlineLfrCmvn(std::vector<std::vector<float>> &vad_feats, bool input_finished);
  // 返回结果单位为ms
  std::vector<std::pair<long, long>> OnlineSplitAudio(const std::vector<int16_t> &audio);
  void ResetCache();
  void Reset();
  void FbankKaldi(std::vector<std::vector<float>> &vad_feats, std::vector<float> &waves);


 private:
  inline static AliVadEngine engine;
  kaldi::FbankOptions opts;
  // std::unique_ptr<kaldi::OnlineFbank> onlinefbank;
  void InitCache();
  E2EVadModel vad_scorer;
  
  // std::vector<std::pair<long,long>> segments;
  // lfr reserved cache
  std::vector<std::vector<float>> lfr_splice_cache_;
  std::vector<float> reserve_waveforms_;
  std::vector<std::vector<float>> in_cache_;
  std::vector<float> input_cache_;

  std::vector<float> means, vars;
  int vad_sample_rate_ = 16000;
  int vad_silence_duration_ = 800;
  int vad_max_len_ = 60000;
  double vad_speech_noise_thres_ = 0.6;
  int lfr_m_=5;
  int lfr_n_=1;
  int frame_sample_length_ = 16000 / 1000 * 25;;
  int frame_shift_sample_length_ = 16000 / 1000 * 10;
};



};  // namespace BASE_NAMESPACE