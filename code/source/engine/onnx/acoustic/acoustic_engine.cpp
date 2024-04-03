/*
 * @Author: dm.liujunshi
 * @Date: 2021-07-19 09:52:02
 * @Last Modified by: dm.liujunshi
 * @Last Modified time: 2021-09-07 20:37:01
 */

#include "acoustic_engine.h"

#include "utils/logging.h"
#include "utils/regex-util.h"
#include "utils/string-util.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <numeric>
#include <unordered_set>

namespace BASE_NAMESPACE {

const static std::vector<float> dur_embedding_weight{
  0.0174,  -0.0164, -0.0167, -0.0165, -0.0165, -0.9983, -0.0163, 0.0171,  -0.0164, -0.0164, 0.0164,  -1.3409, -0.0163,
  -0.0164, -0.0170, 0.0168,  -2.6578, -1.3277, -0.0163, -0.0163, -0.0163, 0.0166,  0.0163,  2.8813,  0.0165,  0.0168,
  0.0169,  0.0163,  0.0164,  0.0163,  -2.9538, 0.0164,  -0.0181, -0.0163, 0.0164,  2.8784,  2.7233,  0.7331,  -0.0164,
  0.0165,  -0.0166, -2.5067, -0.0163, -0.0164, 0.0165,  -0.0169, 0.0163,  3.1538,  1.7814,  0.0163,  -0.0164, -1.0314,
  1.7700,  0.0164,  0.0164,  0.0165,  0.0167,  0.0163,  0.0163,  0.0164,  -0.0168, -0.0163, -1.5425, -0.9971};
const static std::vector<float> dur_embedding_bias{
  0.0106,  -0.0091, -0.0098, -0.0107, -0.0094, 0.9645,  -0.0088, 0.0105,  -0.0090, -0.0090, 0.0093,  0.1967,  -0.0088,
  -0.0091, -0.0104, 0.0116,  0.2879,  0.1959,  -0.0089, -0.0088, -0.0088, 0.0096,  0.0088,  -0.3402, 0.0092,  0.0096,
  0.0102,  0.0087,  0.0099,  0.0089,  0.2450,  0.0093,  -0.0124, -0.0087, 0.0094,  -0.2904, -0.2409, -0.1121, -0.0092,
  0.0093,  -0.0099, 0.2931,  -0.0088, -0.0092, 0.0098,  -0.0114, 0.0088,  -0.2513, -0.2476, 0.0089,  -0.0096, 0.3742,
  -0.2478, 0.0096,  0.0105,  0.0101,  0.0092,  0.0088,  0.0087,  0.0089,  -0.0097, -0.0089, 0.2490,  0.1519};


static std::vector<float> dur_embedding(long dur){
  std::vector<float> dur_emb(dur_embedding_weight.size());
  for(int i=0;i<dur_emb.size();i++){
    dur_emb[i] = dur/100.0 * dur_embedding_weight[i] + dur_embedding_bias[i];
  }
  return dur_emb;
}

static int length_regulate(const CTensorfl &memories, const CTensorll &durs, CTensorfl *out) {
  // memories: [1, T, *]
  // durs: [1, T]
  CHECK_EQ(durs.shapes()[0], 1);
  CHECK_EQ(memories.shapes()[0], 1);
  CHECK_EQ(memories.shapes()[1], durs.shapes()[1]);
  int64_t sum = 0;
  for (int64_t i = 0; i < durs.shapes()[1]; i++) {
    sum += durs.at({0, i});
  }
  out->resize({1, sum, memories.shapes()[2]});
  int64_t repeat_size = memories.shapes()[2] * sizeof(float);
  auto strides = out->strides();
  int64_t s_index = 0, t_index = 0;
  for (int64_t i = 0; i < durs.shapes()[1]; i++) {
    s_index = i;
    for (int64_t j = 0; j < durs.at({0, i}); j++) {  // repeat data
      memcpy(out->data() + t_index * strides[1], memories.data() + s_index * strides[1], repeat_size);
      t_index++;
    }
  }
  return 0;
}

static int adapt_duration(const std::vector<std::string> &phonemes, CTensorll &durs) {
  // 局部语速调整，以sil、sp2、 sp1为界
  LOG(INFO) << "adapt duration start";
  std::unordered_set<std::string> silence_phoneme{"S_sil", "S_sp2", "S_sp1"};
  std::vector<int> sp_index;
  for (int i = 0; i < phonemes.size(); i++) {
    if (silence_phoneme.count(phonemes[i])) { sp_index.push_back(i); }
  }
  double MIN_AVG_PHONE_DUR = 8.5;
  for (int i = 0; i < sp_index.size() - 1; i++) {
    int start_index = sp_index[i] + 1;  // sil、sp2、sp1不计入dur统计
    int end_index = sp_index[i + 1];
    double sum = 0;
    for (int j = start_index; j < end_index; j++) {
      sum += durs.at({0, j});
    }
    if (sum / (end_index - start_index) < MIN_AVG_PHONE_DUR) {
      double ratio = MIN_AVG_PHONE_DUR * (end_index - start_index) / sum;
      int k = start_index;
      while (k < end_index) {
        // 当前音素和后一个音素属于一个汉字
        if (startswith(phonemes[k], std::string("ZH")) && REGEX::match(phonemes[k + 1], std::string("ZH.*[12345]"))) {
          // 该汉字的时长超过20帧，则不再放大
          if (durs.at({0, k}) + durs.at({0, k + 1}) > 20) {
            k += 2;
            continue;
          }
        }
        durs.at({0, k}) = std::ceil(durs.at({0, k}) * ratio);
        k++;
      }
    }
  }
  long MIN_SP2_DUR = 20;
  for (int i = 0; i < phonemes.size(); i++) {
    if (phonemes[i] == "S_sp2") { durs.at({0, i}) = std::max(durs.at({0, i}), MIN_SP2_DUR); }
  }

  //调整句尾sil时长
  int END_SIL_DUR = 20;
  durs.at({0, static_cast<int64_t>(phonemes.size() - 1)}) = END_SIL_DUR;
  //开头字、结尾字增加duration
  double START_WORD_DUR = 20;
  double END_WORD_DUR = 20;
  if (durs.shapes()[1] > 5) {
    if (durs.at({0, 1}) + durs.at({0, 2}) < START_WORD_DUR) {
      double ratio = START_WORD_DUR / (durs.at({0, 1}) + durs.at({0, 2}));
      durs.at({0, 1}) = static_cast<int64_t>(ratio * durs.at({0, 1}) + 0.5);
      durs.at({0, 2}) = static_cast<int64_t>(ratio * durs.at({0, 2}) + 0.5);
    }
    int len = durs.shapes()[1];
    if (durs.at({0, len - 2}) + durs.at({0, len - 3}) < END_WORD_DUR) {
      double ratio = END_WORD_DUR / (durs.at({0, len - 2}) + durs.at({0, len - 3}));
      durs.at({0, len - 2}) = static_cast<int64_t>(ratio * durs.at({0, len - 2}) + 0.5);
      durs.at({0, len - 3}) = static_cast<int64_t>(ratio * durs.at({0, len - 3}) + 0.5);
    }
  }
  // 限制duration的最小值
  int MIN_DUR = 6;
  for (int i = 1; i < durs.shapes()[1]; i++) {
    if (durs.at({0, i}) < MIN_DUR) durs.at({0, i}) = MIN_DUR;
  }
  LOG(INFO) << "adapt duration end";
  return 0;
}

int AcousticEngine::loadModel(const std::string &encoderPath, const std::string &decoderPath,
                              const std::string &postnetPath, int num_threads) {
  LOG(INFO) << "load acoustic model:" << encoderPath << ";" << decoderPath << ";" << postnetPath;
  Ort::Env &env = ONNXENV::getInstance();
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(num_threads);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  // encoder
  encoder_inputNames = {"seqs", "skips"};
  encoder_outputNames = {"memories", "durs"};
  encoder_session = std::make_unique<Ort::Session>(env, encoderPath.c_str(), session_options);

  // decoder
  decoder_inputNames = {"frame", "memory", "rnn_h0_in", "rnn_h1_in", "rnn_h2_in"};
  decoder_outputNames = {"mel", "rnn_h0_out", "rnn_h1_out", "rnn_h2_out"};
  decoder_session = std::make_unique<Ort::Session>(env, decoderPath.c_str(), session_options);

  // postnet
  postnet_inputNames = {"mels_in"};
  postnet_outputNames = {"mels_out"};
  postnet_session = std::make_unique<Ort::Session>(env, postnetPath.c_str(), session_options);
  LOG(INFO) << "load acoustic model success";
  is_init_ = true;
  return 0;
}

int AcousticEngine::encoderInfer(const CTensorll &seqs, const CTensorll &skips,
                                 const std::vector<std::string> &phonemes, CTensorfl *out, std::vector<int> *dur) {
  LOG(INFO) << "run encoder";
  try {
    std::vector<Ort::Value> encoder_inputTensors;
    Ort::MemoryInfo memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    encoder_inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
      memoryInfo, const_cast<int64_t *>(seqs.data()), seqs.size(), seqs.shapes().data(), seqs.shapes().size()));
    encoder_inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
      memoryInfo, const_cast<int64_t *>(skips.data()), skips.size(), skips.shapes().data(), skips.shapes().size()));
    auto encoder_out = encoder_session->Run(Ort::RunOptions{nullptr}, encoder_inputNames.data(),
                                            encoder_inputTensors.data(), 2, encoder_outputNames.data(), 2);
    std::vector<int64_t> memories_shape = encoder_out[0].GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> durs_shapes = encoder_out[1].GetTensorTypeAndShapeInfo().GetShape();
    CTensorfl memories(memories_shape);
    CTensorll durs(durs_shapes);
    memcpy(memories.data(), encoder_out[0].GetTensorMutableData<float>(), memories.byteSize());
    memcpy(durs.data(), encoder_out[1].GetTensorMutableData<float>(), durs.byteSize());
    if (phonemes.size() > 0) {
      adapt_duration(phonemes, durs);
      for(int i=0;i<durs.shapes()[1];i++){
        std::vector<float> dur_emb = dur_embedding(durs.at({0,i}));
        int j = memories.shapes()[2]-1;
        int k = dur_emb.size()-1;
        while(k>=0){
          memories.at({0,i,j}) = dur_emb[k];
          k--;
          j--;
        }
      }
    }

    if (dur) {
      // 将skip音素设置为0
      int i = 0, j = 0;
      while (i < skips.shapes()[0]) {
        if (skips.at({i}) == 0) {
          dur->push_back(0);
        } else {
          dur->push_back(durs.at({0, j}));
          j++;
        }
        i++;
      }
    }
    length_regulate(memories, durs, out);
    return 0;
  } catch (Ort::Exception &e) {
    LOG(ERROR) << e.what();
    return -1;
  }
}

int AcousticEngine::decoderInfer(const CTensorfl &memories, CTensorfl *out) {
  LOG(INFO) << "run decoder";
  try {
    CTensorfl frame({1, n_mel_channels});
    CTensorfl rnn_h0({1, decoder_rnn_dim});
    CTensorfl rnn_h1({1, decoder_rnn_dim});
    CTensorfl rnn_h2({1, decoder_rnn_dim});
    // 拼接相邻n_frames_per_step帧，求均值
    std::vector<CTensorfl> mem_mean;
    int64_t feature_dim = memories.shapes()[2];

    for (int i = 0; i + n_frames_per_step <= memories.shapes()[1]; i += n_frames_per_step) {
      CTensorfl mean({1, feature_dim});
      for (int j = 0; j < feature_dim; j++) {
        for (int k = 0; k < n_frames_per_step; k++) {
          mean.at({0, j}) += memories.at({0, i + k, j});
        }
        mean.at({0, j}) /= n_frames_per_step;
      }
      mem_mean.push_back(mean);
    }

    Ort::MemoryInfo memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    std::vector<CTensorfl> mels;
    for (auto &m : mem_mean) {
      std::vector<Ort::Value> decoder_inputTensors;
      decoder_inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, frame.data(), frame.size(),
                                                                     frame.shapes().data(), frame.shapes().size()));
      decoder_inputTensors.push_back(
        Ort::Value::CreateTensor<float>(memoryInfo, m.data(), m.size(), m.shapes().data(), m.shapes().size()));
      decoder_inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, rnn_h0.data(), rnn_h0.size(),
                                                                     rnn_h0.shapes().data(), rnn_h0.shapes().size()));
      decoder_inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, rnn_h1.data(), rnn_h1.size(),
                                                                     rnn_h1.shapes().data(), rnn_h1.shapes().size()));
      decoder_inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, rnn_h2.data(), rnn_h2.size(),
                                                                     rnn_h2.shapes().data(), rnn_h2.shapes().size()));
      auto decoder_out = decoder_session->Run(Ort::RunOptions{nullptr}, decoder_inputNames.data(),
                                              decoder_inputTensors.data(), 5, decoder_outputNames.data(), 4);
      // 获取输出结果，循环计算
      CTensorfl mel({1, n_frames_per_step, n_mel_channels});
      memcpy(mel.data(), decoder_out[0].GetTensorMutableData<float>(), mel.byteSize());
      memcpy(frame.data(), &mel.at({0, n_frames_per_step - 1, 0}), frame.byteSize());
      memcpy(rnn_h0.data(), decoder_out[1].GetTensorMutableData<float>(), rnn_h0.byteSize());
      memcpy(rnn_h1.data(), decoder_out[2].GetTensorMutableData<float>(), rnn_h1.byteSize());
      memcpy(rnn_h2.data(), decoder_out[3].GetTensorMutableData<float>(), rnn_h2.byteSize());
      mels.push_back(mel);
    }

    // concatenate and transpose mels
    out->resize({1, n_mel_channels, static_cast<int64_t>(mels.size() * n_frames_per_step)});
    for (int k = 0; k < mels.size(); k++) {
      for (int i = 0; i < n_frames_per_step; i++) {
        #pragma unroll n_mel_channels
        for (int j = 0; j < n_mel_channels; j++) {
          out->at({0, j, i + k * n_frames_per_step}) = mels[k].at({0, i, j});
        }
      }
    }
    return 0;
  } catch (Ort::Exception &e) {
    LOG(ERROR) << e.what();
    return -1;
  }
}

int AcousticEngine::postnetInfer(const CTensorfl &mels, CTensorfl *out) {
  LOG(INFO) << "run postnet";
  try {
    std::vector<Ort::Value> postnet_inputTensors;
    Ort::MemoryInfo memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    postnet_inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, mels.data(), mels.size(),
                                                                   mels.shapes().data(), mels.shapes().size()));

    auto postnet_out = postnet_session->Run(Ort::RunOptions{nullptr}, postnet_inputNames.data(),
                                            postnet_inputTensors.data(), 1, postnet_outputNames.data(), 1);
    out->resize(postnet_out[0].GetTensorTypeAndShapeInfo().GetShape());
    memcpy(out->data(), postnet_out[0].GetTensorMutableData<float>(), out->byteSize());
    return 0;
  } catch (Ort::Exception &e) {
    LOG(ERROR) << e.what();
    return -1;
  }
}

int AcousticEngine::infer(const CTensorll &seqs, const CTensorll &skips, const std::vector<std::string> &phonemes,
                          CTensorfl *out, std::vector<int> *dur) {
  LOG(INFO) << "AcousticEngine::infer";
  CTensorfl memories, mels;
  encoderInfer(seqs, skips, phonemes, &memories, dur);
  decoderInfer(memories, &mels);
  postnetInfer(mels, out);
  LOG(INFO) << "AcousticEngine::infer end";
  return 0;
}

};  // namespace BASE_NAMESPACE
