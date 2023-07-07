/*
 * @Author: zack
 * @Date: 2021-12-23 15:11:11
 * @Last Modified by: zack
 * @Last Modified time: 2021-12-23 18:23:53
 */

#include "SileroVadWrapper.h"
#include "utils/logging.h"
#include "utils/string-util.h"
#include <numeric>

#define MAX_DURATION 180  //使用vad断句，每个句子最长180s

namespace BASE_NAMESPACE {

SileroVadWrapper::SileroVadWrapper(const std::string ModelPath, int Sample_rate, int frame_ms, float Threshold,
                                   int min_silence_duration_ms, int speech_pad_ms) {
  auto env = ONNXENV::getInstance();
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetInterOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session = std::make_unique<Ort::Session>(*env, ModelPath.c_str(), session_options);

  sample_rate = Sample_rate;
  sr_per_ms = sample_rate / 1000;
  threshold = Threshold;
  min_silence_samples = sr_per_ms * min_silence_duration_ms;
  speech_pad_samples = sr_per_ms * speech_pad_ms;
  window_size_samples = frame_ms * sr_per_ms;
  input_node_dims[0] = 1;
  input_node_dims[1] = window_size_samples;
  sr.resize(1);
  sr[0] = sample_rate;
}

std::vector<std::pair<size_t, size_t>> SileroVadWrapper::SplitAudio(const std::vector<int16_t> &audio,
                                                                    int sample_rate) {
  LOG(INFO) << "SileroVad recive " << audio.size() << " samples with sample_rate " << sample_rate;

  std::vector<std::pair<size_t, size_t>> results;
  if (audio.size()<window_size_samples) return results;
  std::vector<float> faudio(audio.size());
  for (int i = 0; i < audio.size(); i++)
    faudio[i] = static_cast<float>(audio[i]) / 32768;

  bool triggerd = false;
  unsigned int speech_start = 0;
  unsigned int speech_end = 0;
  unsigned int temp_end = 0;
  unsigned int current_sample = 0;
  std::vector<size_t> start_end;
  std::vector<float> _h(size_hc);
  std::vector<float> _c(size_hc);
  auto iter = faudio.begin();
  while (iter + window_size_samples < faudio.end()) {
    std::vector<float> input{iter, iter + window_size_samples};
    iter += window_size_samples;
    // predict
    {
      auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      std::vector<Ort::Value> ort_inputs;
      // Infer
      // Create ort tensors
      Ort::Value input_ort =
        Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), input_node_dims, 2);
      Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(memory_info, sr.data(), sr.size(), sr_node_dims, 1);
      Ort::Value h_ort = Ort::Value::CreateTensor<float>(memory_info, _h.data(), _h.size(), hc_node_dims, 3);
      Ort::Value c_ort = Ort::Value::CreateTensor<float>(memory_info, _c.data(), _c.size(), hc_node_dims, 3);
      ort_inputs.emplace_back(std::move(input_ort));
      ort_inputs.emplace_back(std::move(sr_ort));
      ort_inputs.emplace_back(std::move(h_ort));
      ort_inputs.emplace_back(std::move(c_ort));

      // Infer
      auto ort_outputs = session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),
                                 ort_inputs.size(), output_node_names.data(), output_node_names.size());

      // Output probability & update h,c recursively
      float output = ort_outputs[0].GetTensorMutableData<float>()[0];
      float *hn = ort_outputs[1].GetTensorMutableData<float>();
      std::memcpy(_h.data(), hn, size_hc * sizeof(float));
      float *cn = ort_outputs[2].GetTensorMutableData<float>();
      std::memcpy(_c.data(), cn, size_hc * sizeof(float));

      // Push forward sample index
      current_sample += window_size_samples;

      // Reset temp_end when > threshold
      if ((output >= threshold) && (temp_end != 0)) { temp_end = 0; }
      // 1) Silence
      if ((output < threshold) && (triggerd == false)) {
        // printf("{ silence: %.3f s }\n", 1.0 * current_sample / sample_rate);
      }
      // 2) Speaking
      if ((output >= (threshold - 0.15)) && (triggerd == true)) {
        // printf("{ speaking_2: %.3f s }\n", 1.0 * current_sample / sample_rate);
      }

      // 3) Start
      if ((output >= threshold) && (triggerd == false)) {
        triggerd = true;
        speech_start = current_sample - window_size_samples
                       - speech_pad_samples;  // minus window_size_samples to get precise start time point.
        start_end.push_back(speech_start);
        LOG(INFO) << "{ start: " << 1.0 * speech_start / sample_rate << " s }";
      }

      // 4) End
      if ((output < (threshold - 0.15)) && (triggerd == true)) {
        if (temp_end != 0) { temp_end = current_sample; }
        // a. silence < min_slience_samples, continue speaking
        if ((current_sample - temp_end) < min_silence_samples) {
          // printf("{ speaking_4: %.3f s }\n", 1.0 * current_sample / sample_rate);
          // printf("");
        }
        // b. silence >= min_slience_samples, end speaking
        else {
          speech_end = current_sample + speech_pad_samples;
          temp_end = 0;
          triggerd = false;
          start_end.push_back(speech_end);
          LOG(INFO) << "{ end: " << 1.0 * speech_end / sample_rate << " s }";
        }
      }
    }
  }
  if (start_end.size() % 2 != 0) start_end.push_back(audio.size());
  for (int i = 0; i < start_end.size() / 2; i++)
    results.push_back(std::make_pair(start_end[2 * i], start_end[2 * i + 1]));
  return results;
}

std::vector<int16_t> SileroVadWrapper::RemoveSilence(const std::vector<int16_t> &audio, int sample_rate) {
  std::vector<int16_t> results;
  LOG(INFO) << "SileroVad recive " << audio.size() << " samples with sample_rate " << sample_rate;
  auto segments = SplitAudio(audio, sample_rate);
  for (const auto &seg : segments) {
    results.insert(results.end(), audio.begin() + seg.first, audio.begin() + seg.second);
  }

  LOG(INFO) << "keep " << results.size() << " samples";
  return results;
}



};  // namespace BASE_NAMESPACE