#ifndef BASE_AUDIO_FEATURE_H_
#define BASE_AUDIO_FEATURE_H_
#include <vector>
#include <complex>
namespace BASE_NAMESPACE {

// 预加重
void PreEmphasis(float coeff, std::vector<float> &data);

std::vector<float> PoveyWindow(int frame_length);
// periodic是否删除最后一个重复值
// hann_window(L, periodic=True) 等价于hann_window(L + 1, periodic=False)[:-1])
std::vector<float> HannWindow(int frame_length,bool periodic=true);
std::vector<float> HammingWindow(int frame_length);

float MelScale(float freq);
float InverseMelScale(float mel_freq);
std::vector<std::vector<double>> GenMelFilter(int num_bin, int n_fft,int sample_rate);

void butterflyFFT(std::vector<double>& real, std::vector<double>& imag, bool inverse=false);
std::vector<std::vector<std::complex<double>>> stft(const std::vector<float> &data,int n_fft, int hop_length,const std::vector<float> &window); 

}
#endif