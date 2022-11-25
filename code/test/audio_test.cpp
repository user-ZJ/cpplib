#include <iostream>
#include <string>
#include <cassert>
#include <vector>
#include <limits>
#include "vad/FVadWrapper.h"
#include "utils/sox-util.h"
#include "utils/file-util.h"
#include "utils/logging.h"
#include "utils/flags.h"
#include "opus/OpusWrapper.h"


using namespace BASE_NAMESPACE;


int main(int argc, char *argv[]) {
  FLAGS_log_dir = ".";
  google::EnableLogCleaner(30);  // keep your logs for 30 days
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);  //初始化 glog
  google::SetStderrLogging(google::GLOG_ERROR);
  if (argc < 2) {
    LOG(INFO) << "usage: audio_test audio_path" << std::endl;
    exit(1);
  }
  int res;
  WavInfo info;
  res = SoxUtil::instance().GetWavInfo(argv[1],info);
  LOG(INFO)<<info.sample_rate<<" "<<info.channel<<" "<<info.sample_num<<" "<<info.precision<<std::endl;
  std::vector<short> data;
  res = SoxUtil::instance().GetData(argv[1],data);
  LOG(INFO)<<"size:"<<data.size()<<std::endl;
  auto wav = SoxUtil::instance().ProcessWav(info,data,24000,/*volume*/1.0,/*speed*/0.5);
  write_to_file("out.wav",wav);
  auto mp3 = SoxUtil::instance().Wav2Mp3(wav);
  write_to_file("out.mp3",mp3);

  auto wav_buff = SoxUtil::instance().Mp3ToWav(mp3);
  write_to_file("out1.wav",wav_buff);


  std::vector<unsigned char> opus_buff;
  res = OpusEncode(data,info.sample_rate,opus_buff);
  LOG(INFO)<<"opus_buff:"<<opus_buff.size();
  std::vector<short> opus_decoded;
  // res = OpusDecode(opus_buff,info.sample_rate,opus_decoded);
  // LOG(INFO)<<"opus_decoded:"<<opus_decoded.size();
  // SoxUtil::instance().Write2File(info,opus_decoded,"opus_decoded.wav");
  
  return 0;
}
