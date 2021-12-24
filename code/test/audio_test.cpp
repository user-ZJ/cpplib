#include "sox.h"
#include <iostream>
#include <string>
#include <cassert>
#include <vector>
#include <limits>
#include "vad/FVadWrapper.h"


using namespace BASE_NAMESPACE;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "usage: audio_test audio_path" << std::endl;
    exit(1);
  }
  sox_format_t *in;
  size_t blocks, block_size;
  /* Period of audio over which we will measure its volume in order to
   * display the wave-form: */
  static const double block_period = 0.025; /* seconds */
  double start_secs = 0, period = 2;
  char dummy;
  uint64_t seek;

  /* All libSoX applications must start by initialising the SoX library */
  assert(sox_init() == SOX_SUCCESS);
  assert(in = sox_open_read(argv[1], NULL, NULL, NULL));
  int sample_rate = in->signal.rate;
  int channels = in->signal.channels;
  int64_t sample_num = in->signal.length;
  std::cout<<"sample_rate:"<<sample_rate<<" channels:"<<channels<<" sample_num:"<<sample_num<<std::endl;
  std::vector<sox_sample_t> audio_data(sample_num);
  std::vector<int16_t>  audio_int16(sample_num);
  assert(sox_read(in, audio_data.data(), sample_num)==sample_num);
  for(size_t i=0;i<audio_data.size();i++){
      //std::cout<<SOX_SAMPLE_TO_FLOAT_64BIT(audio_data[i],)<<" ";
      //std::cout<<SOX_SAMPLE_TO_SIGNED_32BIT(audio_data[i],)<<" ";
      int16_t d = SOX_SAMPLE_TO_FLOAT_64BIT(audio_data[i],)*(std::numeric_limits<int16_t>::max()+1.0);
      audio_int16[i] = d;
      // std::cout<<d<<" ";
  }
  std::cout<<std::endl;

  ////////////////////////////////////////////////
  FVadWrapper vad;
  vad.Init("");
  auto vadres = vad.SplitAudio(audio_int16,sample_rate);
  for(auto &p:vadres){
    std::cout<<p.first/8000.0<<","<<p.second/8000.0<<"\n";
  }
  ////////////////////////////////////////////////
  sox_close(in);
  sox_quit();
  return 0;
}