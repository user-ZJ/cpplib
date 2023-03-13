#include <iostream>
#include <string>
#include <cassert>
#include <vector>
#include <limits>
#include "remove_noise/RnnoiseWrapper.h"
#include "utils/audio-util.h"
#include "utils/file-util.h"
#include "utils/logging.h"
#include "utils/flags.h"


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
  RnnoiseWrapper rmnoise;
  WavInfo info;
  res = SoxUtil::instance().GetWavInfo(argv[1],info);
  LOG(INFO)<<info.sample_rate<<" "<<info.channel<<" "<<info.sample_num<<" "<<info.precision<<std::endl;
  std::vector<int16_t> data;
  res = SoxUtil::instance().GetData(argv[1],data);
  LOG(INFO)<<"size:"<<data.size()<<std::endl;

  auto buff = SoxUtil::instance().Write2Buff(info,data);
  rmnoise.RemoveNoise(buff,"");
 
  
  writeBinaryFile("out.wav",buff);
  return 0;
}
