#include <iostream>
#include <string>
#include <cassert>
#include <vector>
#include <limits>
#include "vad/FVadWrapper.h"
#include "utils/sox-util.h"
#include "utils/file-util.h"


using namespace BASE_NAMESPACE;


int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "usage: audio_test audio_path" << std::endl;
    exit(1);
  }
  auto info = SoxUtil::instance().GetWavInfo(argv[1]);
  std::cout<<info.sample_rate<<" "<<info.channel<<" "<<info.sample_num<<" "<<info.precision<<std::endl;
  std::vector<double> data;
  SoxUtil::instance().GetData(argv[1],&data);
  std::cout<<"size:"<<data.size()<<std::endl;
  auto wav = SoxUtil::instance().ProcessWav(info,data,24000,/*volume*/1.0,/*speed*/1.5);
  write_to_file("out.wav",wav);
  auto mp3 = SoxUtil::instance().Wav2Mp3(wav);
  write_to_file("out.mp3",mp3);
  
  return 0;
}