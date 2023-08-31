#include "aliVadWrapper.h"
#include "e2e-vad.h"
#include "kaldi/feat/kaldi-featlibs.h"
#include "utils/file-util.h"
#include "utils/logging.h"
#include "utils/string-util.h"
#include <exception>
#include <numeric>

namespace BASE_NAMESPACE {

static int ComputeFrameNum(int sample_length, int frame_sample_length, int frame_shift_sample_length) {
  int frame_num = static_cast<int>((sample_length - frame_sample_length) / frame_shift_sample_length + 1);
  if (frame_num >= 1 && sample_length >= frame_sample_length)
    return frame_num;
  else
    return 0;
}

AliVadEngine::AliVadEngine(const std::string ModelPath,const std::string CmvnPath) {
  auto env = ONNXENV::getInstance();
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetInterOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options.DisableCpuMemArena();
  session = std::make_unique<Ort::Session>(*env, ModelPath.c_str(), session_options);
  auto buff = file_to_buff(CmvnPath.c_str());
  means.resize(400);
  vars.resize(400);
  memcpy(means.data(), buff.data(), 400 * sizeof(float));
  memcpy(vars.data(), buff.data() + 400 * sizeof(float), 400 * sizeof(float));
  LOG(INFO)<<"vad engine construct";
}

void AliVadEngine::Infer(const std::vector<std::vector<float>> &chunk_feats, std::vector<std::vector<float>> *out_prob,
                         std::vector<std::vector<float>> *in_cache) {
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  int num_frames = chunk_feats.size();
  const int feature_dim = chunk_feats[0].size();

  //  2. Generate input nodes tensor
  // vad node { batch,frame number,feature dim }
  const int64_t vad_feats_shape[3] = {1, num_frames, feature_dim};
  std::vector<float> vad_feats;
  for (const auto &chunk_feat : chunk_feats) {
    vad_feats.insert(vad_feats.end(), chunk_feat.begin(), chunk_feat.end());
  }
  Ort::Value vad_feats_ort =
    Ort::Value::CreateTensor<float>(memory_info, vad_feats.data(), vad_feats.size(), vad_feats_shape, 3);

  // 3. Put nodes into onnx input vector
  std::vector<Ort::Value> vad_inputs;
  vad_inputs.emplace_back(std::move(vad_feats_ort));
  // 4 caches
  // cache node {batch,128,19,1}
  const int64_t cache_feats_shape[4] = {1, 128, 19, 1};
  for (int i = 0; i < in_cache->size(); i++) {
    vad_inputs.emplace_back(std::move(Ort::Value::CreateTensor<float>(memory_info, (*in_cache)[i].data(),
                                                                      (*in_cache)[i].size(), cache_feats_shape, 4)));
  }

  // 4. Onnx infer
  std::vector<Ort::Value> vad_ort_outputs;
  try {
    vad_ort_outputs = session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), vad_inputs.data(),
                                   vad_inputs.size(), output_node_names.data(), output_node_names.size());
  }
  catch (std::exception const &e) {
    LOG(ERROR) << "Error when run vad onnx forword: " << (e.what());
    return;
  }

  // 5. Change infer result to output shapes
  float *logp_data = vad_ort_outputs[0].GetTensorMutableData<float>();
  auto type_info = vad_ort_outputs[0].GetTensorTypeAndShapeInfo();

  int num_outputs = type_info.GetShape()[1];
  int output_dim = type_info.GetShape()[2];
  out_prob->resize(num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    (*out_prob)[i].resize(output_dim);
    memcpy((*out_prob)[i].data(), logp_data + i * output_dim, sizeof(float) * output_dim);
  }

  // get 4 caches outputs,each size is 128*19
  {
    for (int i = 1; i < 5; i++) {
      float *data = vad_ort_outputs[i].GetTensorMutableData<float>();
      memcpy((*in_cache)[i - 1].data(), data, sizeof(float) * 128 * 19);
    }
  }
}

std::vector<std::vector<float>> AliVadWrapper::LfrCmvn(std::vector<std::vector<float>> &vad_feats) {
  // lfr
  int T = vad_feats.size();
  int T_lfr = std::ceil(1.0 * T / lfr_n_);
  // left_padding
  int left_padding_num = (lfr_m_ - 1) / 2;
  for (int i = 0; i < left_padding_num; i++) {
    vad_feats.insert(vad_feats.begin(), vad_feats[0]);
  }
  // right_padding
  T += left_padding_num;
  int right_padding_num = T_lfr * lfr_n_ + (lfr_m_ - lfr_n_) - T;
  for (int i = 0; i < right_padding_num; i++) {
    vad_feats.insert(vad_feats.end(), vad_feats[vad_feats.size() - 1]);
  }

  std::vector<std::vector<float>> out_feats(T_lfr, std::vector<float>(vad_feats[0].size() * lfr_m_));
  for (int i = 0; i < T_lfr; i++) {
    for (int j = 0; j < lfr_m_; j++)
      memcpy(&out_feats[i][j * vad_feats[0].size()], vad_feats[i * lfr_n_ + j].data(),
             vad_feats[0].size() * sizeof(float));
  }
  // cmvn
  for (int i = 0; i < out_feats.size(); i++) {
    for (int j = 0; j < out_feats[0].size(); j++) {
      out_feats[i][j] = (out_feats[i][j] + means[j]) * vars[j];
    }
  }
  return out_feats;
}

AliVadWrapper::AliVadWrapper() {
  means = AliVadWrapper::engine.GetMeans();
  vars = AliVadWrapper::engine.GetVars();

  opts.frame_opts.samp_freq = 16000;
  opts.frame_opts.dither = 0.0;
  opts.frame_opts.window_type = "hamming";
  opts.frame_opts.frame_shift_ms = 10;
  opts.frame_opts.frame_length_ms = 25;
  opts.frame_opts.snip_edges = true;
  opts.mel_opts.num_bins = 80;
  opts.energy_floor = 0;
  opts.mel_opts.debug_mel = false;
  // onlinefbank = std::make_unique<kaldi::OnlineFbank>(opts);

  vad_scorer = E2EVadModel();

  InitCache();
}

std::vector<std::vector<int>> AliVadWrapper::AcceptAudio(const std::vector<int16_t> &audio, bool input_finished) {
  std::vector<float> waves(audio.size());
  for (int i = 0; i < waves.size(); i++)
    waves[i] = audio[i] / 32768.0;
  std::vector<std::vector<int>> vad_segments;
  std::vector<std::vector<float>> vad_feats;
  std::vector<std::vector<float>> vad_probs;
  ExtractFeats(vad_feats, waves, input_finished);
  if (vad_feats.size() == 0) { return vad_segments; }
  AliVadWrapper::engine.Infer(vad_feats, &vad_probs, &in_cache_);
  if (vad_probs.size() == 0) { return vad_segments; }

  vad_segments = vad_scorer(vad_probs, waves, input_finished, true, vad_silence_duration_, vad_max_len_,
                            vad_speech_noise_thres_, vad_sample_rate_);
  return vad_segments;
}

void AliVadWrapper::FbankKaldi(std::vector<std::vector<float>> &vad_feats, std::vector<float> &waves) {
  kaldi::OnlineFbank fbank(opts);
  waves.insert(waves.begin(), input_cache_.begin(), input_cache_.end());
  int frame_number = ComputeFrameNum(waves.size(), frame_sample_length_, frame_shift_sample_length_);
  // Send the audio after the last frame shift position to the cache
  input_cache_.clear();
  input_cache_.insert(input_cache_.begin(), waves.begin() + frame_number * frame_shift_sample_length_, waves.end());
  if (frame_number == 0) { return; }
  // Delete audio that haven't undergone fbank processing
  waves.erase(waves.begin() + (frame_number - 1) * frame_shift_sample_length_ + frame_sample_length_, waves.end());

  std::vector<float> buf(waves.size());
  for (int32_t i = 0; i != waves.size(); ++i) {
    buf[i] = waves[i] * 32768;
  }
  kaldi::Vector<float> audiodata(buf.size());
  memcpy(audiodata.Data(), buf.data(), audiodata.SizeInBytes());
  fbank.AcceptWaveform(vad_sample_rate_, audiodata);
  int frames = fbank.NumFramesReady();
  kaldi::Vector<float> feat(opts.mel_opts.num_bins);
  for (int i = 0; i != frames; ++i) {
    fbank.GetFrame(i,&feat);
    std::vector<float> frame_vector(opts.mel_opts.num_bins);
    memcpy(frame_vector.data(), feat.Data(), feat.SizeInBytes());
    vad_feats.emplace_back(frame_vector);
  }
}

void AliVadWrapper::ExtractFeats(std::vector<std::vector<float>> &vad_feats, std::vector<float> &waves,
                                 bool input_finished) {
  FbankKaldi(vad_feats,waves);

  // cache deal & online lfr,cmvn
  if (vad_feats.size() > 0) {
    if (!reserve_waveforms_.empty()) {
      waves.insert(waves.begin(), reserve_waveforms_.begin(), reserve_waveforms_.end());
    }
    if (lfr_splice_cache_.empty()) {
      for (int i = 0; i < (lfr_m_ - 1) / 2; i++) {
        lfr_splice_cache_.emplace_back(vad_feats[0]);
      }
    }
    if (vad_feats.size() + lfr_splice_cache_.size() >= lfr_m_) {
      vad_feats.insert(vad_feats.begin(), lfr_splice_cache_.begin(), lfr_splice_cache_.end());
      int frame_from_waves = (waves.size() - frame_sample_length_) / frame_shift_sample_length_ + 1;
      int minus_frame = reserve_waveforms_.empty() ? (lfr_m_ - 1) / 2 : 0;
      int lfr_splice_frame_idxs = OnlineLfrCmvn(vad_feats, input_finished);
      int reserve_frame_idx = std::abs(lfr_splice_frame_idxs - minus_frame);
      reserve_waveforms_.clear();
      reserve_waveforms_.insert(reserve_waveforms_.begin(),
                                waves.begin() + reserve_frame_idx * frame_shift_sample_length_,
                                waves.begin() + frame_from_waves * frame_shift_sample_length_);
      int sample_length = (frame_from_waves - 1) * frame_shift_sample_length_ + frame_sample_length_;
      waves.erase(waves.begin() + sample_length, waves.end());
    } else {
      reserve_waveforms_.clear();
      reserve_waveforms_.insert(reserve_waveforms_.begin(),
                                waves.begin() + frame_sample_length_ - frame_shift_sample_length_, waves.end());
      lfr_splice_cache_.insert(lfr_splice_cache_.end(), vad_feats.begin(), vad_feats.end());
    }
  } else {
    if (input_finished) {
      if (!reserve_waveforms_.empty()) { waves = reserve_waveforms_; }
      vad_feats = lfr_splice_cache_;
      if (vad_feats.size() == 0) {
        LOG(ERROR) << "vad_feats's size is 0";
      } else {
        OnlineLfrCmvn(vad_feats, input_finished);
      }
    }
  }
  if (input_finished) {
    Reset();
    ResetCache();
  }
}

int AliVadWrapper::OnlineLfrCmvn(std::vector<std::vector<float>> &vad_feats, bool input_finished) {
  std::vector<std::vector<float>> out_feats;
  int T = vad_feats.size();
  int T_lrf = ceil((T - (lfr_m_ - 1) / 2) / (float)lfr_n_);
  int lfr_splice_frame_idxs = T_lrf;
  std::vector<float> p;
  for (int i = 0; i < T_lrf; i++) {
    if (lfr_m_ <= T - i * lfr_n_) {
      for (int j = 0; j < lfr_m_; j++) {
        p.insert(p.end(), vad_feats[i * lfr_n_ + j].begin(), vad_feats[i * lfr_n_ + j].end());
      }
      out_feats.emplace_back(p);
      p.clear();
    } else {
      if (input_finished) {
        int num_padding = lfr_m_ - (T - i * lfr_n_);
        for (int j = 0; j < (vad_feats.size() - i * lfr_n_); j++) {
          p.insert(p.end(), vad_feats[i * lfr_n_ + j].begin(), vad_feats[i * lfr_n_ + j].end());
        }
        for (int j = 0; j < num_padding; j++) {
          p.insert(p.end(), vad_feats[vad_feats.size() - 1].begin(), vad_feats[vad_feats.size() - 1].end());
        }
        out_feats.emplace_back(p);
      } else {
        lfr_splice_frame_idxs = i;
        break;
      }
    }
  }
  lfr_splice_frame_idxs = std::min(T - 1, lfr_splice_frame_idxs * lfr_n_);
  lfr_splice_cache_.clear();
  lfr_splice_cache_.insert(lfr_splice_cache_.begin(), vad_feats.begin() + lfr_splice_frame_idxs, vad_feats.end());

  // Apply cmvn
  for (auto &out_feat : out_feats) {
    for (int j = 0; j < means.size(); j++) {
      out_feat[j] = (out_feat[j] + means[j]) * vars[j];
    }
  }
  vad_feats = out_feats;
  return lfr_splice_frame_idxs;
}

void AliVadWrapper::InitCache() {
  std::vector<float> cache_feats(128 * 19 * 1, 0);
  for (int i = 0; i < 4; i++) {
    in_cache_.emplace_back(cache_feats);
  }
};

std::vector<std::pair<long, long>> AliVadWrapper::SplitAudio(const std::vector<int16_t> &audio) {
  std::vector<float> waves(audio.size());
  for (int i = 0; i < waves.size(); i++)
    waves[i] = audio[i] / 32768.0;
  std::vector<std::pair<long, long>> result;
  std::vector<std::vector<float>> vad_probs;
  std::vector<std::vector<float>> fbank;
  FbankKaldi(fbank,waves);
  auto feats = LfrCmvn(fbank);
  LOG(INFO) << "cmvn:" << feats.size() << " " << feats[0].size();
  AliVadWrapper::engine.Infer(feats, &vad_probs, &in_cache_);
  // E2EVadModel vad_scorer = E2EVadModel();
  std::vector<std::vector<int>> vad_segments;
  vad_segments = vad_scorer(vad_probs, waves, true, false, vad_silence_duration_, vad_max_len_, vad_speech_noise_thres_,
                            vad_sample_rate_);
  for (int i = 0; i < vad_segments.size(); i++)
    result.push_back(std::make_pair(vad_segments[i][0], vad_segments[i][1]));
  return result;
}

std::vector<int16_t> AliVadWrapper::RemoveSilence(const std::vector<int16_t> &audio) {
  std::vector<int16_t> results;
  LOG(INFO) << "aliVad recive " << audio.size();
  auto segments = SplitAudio(audio);
  for (const auto &seg : segments) {
    results.insert(results.end(), audio.begin() + seg.first, audio.begin() + seg.second);
  }

  LOG(INFO) << "keep " << results.size() << " samples";
  return results;
}

// 返回结果单位为ms
std::vector<std::pair<long, long>> AliVadWrapper::OnlineSplitAudio(const std::vector<int16_t> &audio) {
  std::vector<std::pair<long, long>> result;
  int chunk_size = 800;
  int offset = 0;
  std::vector<std::vector<int>> vad_segments;
  while (offset < audio.size()) {
    std::vector<int16_t> chunk_audio(audio.begin() + offset, audio.begin() + offset + chunk_size);
    offset += chunk_size;
    if(offset < audio.size()){
      vad_segments = AcceptAudio(chunk_audio,false);
    }else{
      vad_segments = AcceptAudio(chunk_audio,true);
    }
    for(int i=0;i<vad_segments.size();i++){
      // 合并segment
      if(result.empty()){
        result.push_back(std::make_pair(vad_segments[i][0],vad_segments[i][1]));
      }else if(result[result.size()-1].second==-1 and vad_segments[i][0]==-1){
        result[result.size()-1].second = vad_segments[i][1];
      }else{
        result.push_back(std::make_pair(vad_segments[i][0],vad_segments[i][1]));
      }
        
      LOG(INFO)<<vad_segments[i][0]<<" "<<vad_segments[i][1];
    }
  }
  return result;
}

void AliVadWrapper::ResetCache() {
  reserve_waveforms_.clear();
  input_cache_.clear();
  lfr_splice_cache_.clear();
}

void AliVadWrapper::Reset() {
  in_cache_.clear();
  InitCache();
};

}  // namespace BASE_NAMESPACE