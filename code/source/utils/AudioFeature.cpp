#include "AudioFeature.h"
#include "fft.h"
#include "utils/logging.h"
#include "utils/string-util.h"
#include <algorithm>
#include <cstring>
#include <math.h>

namespace BASE_NAMESPACE {

constexpr double M_2PI = 2 * M_PI;

static int UpperPowerOfTwo(int n) {
  return static_cast<int>(pow(2, ceil(log(n) / log(2))));
}

void ApplyWindow(const std::vector<float> &window, std::vector<float> &frameData) {
  if (frameData.size() < window.size()) return;
  for (int i = 0; i < frameData.size(); i++)
    frameData[i] *= window[i];
}

void PreEmphasis(float coeff, std::vector<float> &data) {
  if (coeff == 0.0) return;
  for (int i = data.size() - 1; i > 0; i--)
    data[i] -= coeff * data[i - 1];
  data[0] -= coeff * data[0];
}

std::vector<float> PoveyWindow(int frame_length) {
  std::vector<float> window(frame_length);
  double a = M_2PI / (frame_length - 1);
  for (int i = 0; i < frame_length; ++i) {
    window[i] = std::pow(0.5 - 0.5 * cos(a * i), 0.85);
  }
  return window;
}

std::vector<float> HannWindow(int frame_length, bool periodic) {
  frame_length++;
  std::vector<float> window(frame_length);
  double a = M_2PI / (frame_length - 1);
  for (int i = 0; i < frame_length; ++i) {
    window[i] = 0.5 - 0.5 * cos(a * i);
  }
  if (periodic) window.resize(window.size() - 1);
  return window;
}

std::vector<float> HammingWindow(int frame_length) {
  std::vector<float> window(frame_length);
  double a = M_2PI / (frame_length - 1);
  for (int i = 0; i < frame_length; ++i) {
    window[i] = 0.54 - 0.46 * cos(a * i);
  }
  return window;
}

float MelScale(float freq) {
  return 1127.0f * logf(1.0f + freq / 700.0f);
}

float InverseMelScale(float mel_freq) {
  return 700.0f * (expf(mel_freq / 1127.0f) - 1.0f);
}

std::vector<std::vector<double>> GenMelFilter(int num_bin, int n_fft, int sample_rate) {
  int lowFreq = 0, highFreq = sample_rate / 2;
  double melLowFreq = MelScale(lowFreq);
  double melHighFreq = MelScale(highFreq);
  double melFreqDelta = (melHighFreq - melLowFreq) / (num_bin + 1);
  std::vector<double> melCenters(num_bin + 2);
  for (int i = 0; i < num_bin + 2; i++) {
    melCenters[i] = melLowFreq + i * melFreqDelta;
    melCenters[i] = InverseMelScale(melCenters[i]);
  }
  LOG(INFO)<<melCenters.size()<<" "<<printCollection(melCenters);
  std::vector<double> k(n_fft / 2 + 1);
  double FreqDelta = (highFreq - lowFreq) / (n_fft / 2);
  for (int i = 0; i < n_fft / 2 + 1; i++) {
    k[i] = lowFreq + i * FreqDelta;
  }
  LOG(INFO)<<k.size()<<" "<<printCollection(k);
  std::vector<std::vector<double>> filterBank(num_bin);
  for (int i = 1; i < num_bin + 1; i++) {
    filterBank[i-1].resize(k.size());
    for (int j = 0; j < k.size(); j++) {
      if (k[j] < melCenters[i - 1] || k[j] > melCenters[i + 1]) {
        filterBank[i-1][j] = 0;
      } else if (k[j] >= melCenters[i - 1] && k[j] < melCenters[i]) {
        filterBank[i-1][j] = (k[j] - melCenters[i - 1]) / (melCenters[i] - melCenters[i - 1]);
      } else if (k[j] > melCenters[i] && k[j] <= melCenters[i + 1]) {
        filterBank[i-1][j] = (melCenters[i + 1] - k[j]) / (melCenters[i + 1] - melCenters[i]);
      }
    }
  }
  return filterBank;
}

// 蝴蝶算法FFT实现
// 输入参数：实部向量，虚部向量，inverse=true表示进行IFFT，否则进行FFT
void butterflyFFT(std::vector<double> &real, std::vector<double> &imag, bool inverse) {
  int n = real.size();
  int levels = log2(n);  // FFT的层数

  // 交换实部和虚部向量，方便后续处理
  if (inverse) {
    for (int i = 0; i < n; i++) {
      imag[i] = -imag[i];
    }
  }
  std::vector<double> tmp(n);
  for (int i = 0; i < n; i++) {
    std::swap(real[i], imag[i]);
  }

  // 蝴蝶算法迭代
  int length = 2;
  for (int i = 0; i < levels; i++) {
    // 计算旋转因子
    double angle = 2.0 * M_PI / length;
    if (inverse) { angle = -angle; }
    double cosangle = cos(angle);
    double sinangle = sin(angle);

    for (int j = 0; j < n; j += length) {
      // 计算蝴蝶FFT
      double cos = 1.0;
      double sin = 0.0;
      for (int k = j; k < j + length / 2; k++) {
        double tempreal = real.at(k + length / 2) * cos - imag.at(k + length / 2) * sin;
        double tempimag = real.at(k + length / 2) * sin + imag.at(k + length / 2) * cos;
        real[k + length / 2] = real[k] - tempreal;
        imag[k + length / 2] = imag[k] - tempimag;
        real[k] += tempreal;
        imag[k] += tempimag;

        double cos_temp = cos * cosangle - sin * sinangle;
        double sin_temp = sin * cosangle + cos * sinangle;
        cos = cos_temp;
        sin = sin_temp;
      }
    }
    length *= 2;
  }

  // 对于IFFT，将输出除以n
  if (inverse) {
    for (int i = 0; i < n; i++) {
      real[i] /= n;
      imag[i] /= n;
    }
  }

  // 恢复实部和虚部向量顺序
  for (int i = 0; i < n; i++) {
    std::swap(real[i], imag[i]);
  }
}

std::vector<std::vector<std::complex<double>>> stft(const std::vector<float> &data, int n_fft, int hop_length,
                                                    const std::vector<float> &window) {
  auto num_samples = data.size();
  int win_length = n_fft;
  // padding center=true
  int pad = n_fft / 2;
  std::vector<float> paded_data(data.size() + 2 * pad);
  // reflect pad
  std::copy_n(data.begin(), data.size(), paded_data.begin() + pad);
  for (int i = 1; i <= pad; i++) {
    paded_data[pad - i] = paded_data[pad + i];
    paded_data[pad + num_samples - 1 - i] = paded_data[pad + num_samples - 1 + i];
  }

  int fft_points = UpperPowerOfTwo(win_length);

  LOG(INFO) << "fft_points:" << fft_points;
  int n = data.size();
  int num_frames = ceil((double)(n - win_length) / hop_length) + 1;  // 总帧数
  std::vector<std::vector<std::complex<double>>> stft_data(
    num_frames,
    std::vector<std::complex<double>>(n_fft / 2 + 1));  // STFT结果

  const int fft_points_4 = fft_points / 4;
  std::vector<int> bitrev(fft_points);
  std::vector<float> sintbl(fft_points + fft_points_4);
  make_sintbl(fft_points, sintbl.data());
  make_bitrev(fft_points, bitrev.data());

  std::vector<float> fft_real(fft_points, 0), fft_img(fft_points, 0);
  // 逐帧进行STFT
  for (int i = 0; i < num_frames; i++) {
    // 获取当前帧的起始位置
    int start = i * hop_length;

    // 加窗
    std::vector<float> frame(win_length);
    for (int j = 0; j < win_length; j++) {
      frame[j] = paded_data[start + j] * window[j];
    }

    memset(fft_img.data(), 0, sizeof(float) * fft_points);
    memset(fft_real.data() + win_length, 0, sizeof(float) * (fft_points - win_length));
    memcpy(fft_real.data(), frame.data(), sizeof(float) * win_length);
    fft(bitrev.data(), sintbl.data(), fft_real.data(), fft_img.data(), fft_points);

    // FFT计算
    // std::vector<double> real(fft_points, 0);
    // std::vector<double> imag(fft_points, 0);
    // for (int j = 0; j < frame_length; j++) {
    //   real[j] = frame[j];
    //   imag[j] = 0.0;
    // }
    // butterflyFFT(real, imag);

    for (int j = 0; j <= win_length / 2; j++) {
      stft_data[i][j] = std::complex<double>(fft_real[j], fft_img[j]);
    }
  }

  return stft_data;
}

}  // namespace BASE_NAMESPACE
