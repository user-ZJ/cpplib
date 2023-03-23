#include "utils/AudioFeature.h"
#include "utils/audio-util.h"
#include <complex>
#include <iostream>
#include <torch/torch.h>
#include "utils/logging.h"
#include "utils/flags.h"
#include "utils/string-util.h"
#include <iostream>
#include <string>
#include <cassert>
#include <vector>
#include <limits>
#include "vad/FVadWrapper.h"
#include "utils/audio-util.h"
#include "utils/file-util.h"
#include "utils/logging.h"
#include "utils/flags.h"
#include "opus/OpusWrapper.h"
#include "utils/AudioFeature.h"
#include "utils/string-util.h"

using namespace BASE_NAMESPACE;

int main(int argc, char *argv[]) {
  google::EnableLogCleaner(30);  // keep your logs for 30 days
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);  //初始化 glog
  google::SetStderrLogging(google::GLOG_ERROR);
  google::InstallFailureSignalHandler();


  std::vector<float> paded{4., 4., 3., 2., 1., 2., 3., 4., 4., 3., 2., 1., 2., 3., 4., 4.};

  // 输入信号
  //   torch::Tensor input = torch::rand({1, 1000});
  torch::Tensor input = torch::from_blob(paded.data(), at::IntArrayRef({1, 16}), torch::kFloat);
  std::cout << input << std::endl;

  std::vector<float> fwindow{0.5, 0.5, 0.5, 0.5};
  torch::Tensor window = torch::from_blob(fwindow.data(), at::IntArrayRef({4}), torch::kFloat);
  std::cout << input << std::endl;

  // 窗口大小
  int window_size = 4;

  // 窗口类型
  torch::Tensor window1 = torch::hann_window(400);
  std::cout << window1 << std::endl;

  // 重叠大小
  int hop_size = 2;

  // FFT大小
  int fft_size = 8;

  // 计算STFT
  torch::Tensor stft = torch::stft(input, fft_size, hop_size, window_size, window, false, false, true);
  std::cout << stft << std::endl;
  auto stft_accessor = stft.accessor<c10::complex<float>, 3>();
  for (int i = 0; i < stft.size(1); i++) {
    for (int j = 0; j < stft.size(2); j++) {
      std::cout << stft_accessor[0][i][j].real() << "+" << stft_accessor[0][i][j].imag() << "i ";
    }
    std::cout << std::endl;
  }
  return 0;
}