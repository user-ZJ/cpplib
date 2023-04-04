#include "opus/OpusWrapper.h"
#include "utils/AudioFeature.h"
#include "utils/audio-util.h"
#include "utils/file-util.h"
#include "utils/flags.h"
#include "utils/logging.h"
#include "utils/string-util.h"
#include "vad/FVadWrapper.h"
#include <cassert>
#include <complex>
#include <iostream>
#include <limits>
#include <string>
#include <torch/torch.h>
#include <vector>

using namespace BASE_NAMESPACE;

int main(int argc, char *argv[]) {
  std::vector<float> paded{4., 4., 3., 2., 1., 2., 3., 4., 4., 3., 2., 1., 2., 3., 4., 4.};
  

  // 输入信号
  torch::Tensor input = torch::full({1, 800}, 0.5);
//   std::cout<<input<<std::endl;

  // 窗口大小
  int window_size = 400;
  // 重叠大小
  int hop_size = 160;
  // FFT大小
  int fft_size = 400;

  // 窗口类型
  torch::Tensor window = torch::hann_window(fft_size);

  // 计算STFT
  torch::Tensor stft = torch::stft(input, fft_size, hop_size, window_size, window, false, false, true);
  for (int i = 0; i < stft.sizes().size(); i++) {
    std::cout << stft.sizes()[i] << "x";
  }
  std::cout << std::endl;
  auto stft1 = torch::slice(stft, 1, 0, fft_size / 2 + 1);
  stft1 = torch::abs(torch::slice(stft1, 2, 0, stft.sizes()[2] - 1));
  auto magnitudes = torch::mul(stft1, stft1);
  
  torch::Tensor filters = torch::ones({80, 201});

  auto mel_spec = torch::matmul(filters, magnitudes);
  std::cout<<mel_spec<<std::endl;
  auto log_spec = torch::log10(torch::clamp(mel_spec, 1e-10));
  
  log_spec = torch::maximum(log_spec, torch::max(log_spec) - 8.0);
  log_spec = (log_spec + 4.0) / 4.0;
  for (int i = 0; i < log_spec.sizes().size(); i++) {
    std::cout << log_spec.sizes()[i] << "x";
  }
  std::cout << std::endl;
  std::cout << log_spec << std::endl;
  log_spec = log_spec.transpose(2,1).contiguous();
  std::cout << log_spec << std::endl;

  return 0;
}