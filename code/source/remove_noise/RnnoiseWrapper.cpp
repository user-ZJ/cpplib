#include "RnnoiseWrapper.h"
#include "utils/file-util.h"
#include "utils/logging.h"
#include "utils/audio-util.h"
#include <string.h>
#include <vector>

namespace BASE_NAMESPACE {

int RnnoiseWrapper::RemoveNoise(const std::string &infile, const std::string &outfile, const std::string &taskid) {
  LOG(INFO) << "remove file noise. task_id:" << taskid;
  int res;
  WavInfo info;
  res = SoxUtil::instance().GetWavInfo(infile, info);
  info.channel = 1;
  int org_sr = info.sample_rate;
  if (res != 0) return res;
  std::vector<int16_t> data;
  res = SoxUtil::instance().GetData(infile, data);

  if (res != 0) return res;
  if (info.sample_rate != 48000) {
    LOG(INFO) << "resample audio to 48k";
    auto processed_buff = SoxUtil::instance().ProcessWav(info, data, 48000, 1.0, 1.0);
    res = SoxUtil::instance().GetWavInfo(processed_buff, info);
    res = SoxUtil::instance().GetData(processed_buff, data);
    if (res != 0) return res;
  }
  process(data, taskid);
  auto out_buff = SoxUtil::instance().ProcessWav(info, data, org_sr, 1.0, 1.0);
  LOG(INFO) << "write to file";
  writeBinaryFile(outfile.c_str(), out_buff);
  return 0;
}

int RnnoiseWrapper::RemoveNoise(std::vector<char> &buff, const std::string &taskid) {
  try {
    LOG(INFO) << "remove buff noise. task_id:" << taskid;
    int res;
    WavInfo info;
    res = SoxUtil::instance().GetWavInfo(buff, info);
    info.channel = 1;
    int org_sr = info.sample_rate;
    if (res != 0) return res;
    LOG(INFO) << "sample_rate:" << info.sample_rate << " sample_num:" << info.sample_num;
    if (info.sample_num <= 0) { return -1; }
    std::vector<int16_t> data;
    res = SoxUtil::instance().GetData(buff, data);
    if (res != 0) return res;
    if (info.sample_rate != 48000) {
      LOG(INFO) << "resample to 48k";
      auto processed_buff = SoxUtil::instance().ProcessWav(info, data, 48000, 1.0, 1.0);
      res = SoxUtil::instance().GetWavInfo(processed_buff, info);
      res = SoxUtil::instance().GetData(processed_buff, data);
      if (res != 0) return res;
    }
    process(data, taskid);
    buff = SoxUtil::instance().ProcessWav(info, data, org_sr, 1.0, 1.0);
    LOG(INFO) << "remove buff noise end. task_id:" << taskid;
    return 0;
  }
  catch (std::exception &ex) {
    LOG(INFO) << "rnnnoise exception:" << ex.what();
    return -1;
  }
}

void RnnoiseWrapper::process(std::vector<int16_t> &data, const std::string &taskid) {
  LOG(INFO) << "remove noise. task_id:" << taskid;
  DenoiseState *st = rnnoise_create(NULL);
  float x[FRAME_SIZE];
  size_t offset = 0;
  while (offset + FRAME_SIZE <= data.size()) {
    memset(x, 0, sizeof(float) * FRAME_SIZE);
    for (int i = 0; i < FRAME_SIZE; i++)
      x[i] = static_cast<float>(data[offset + i]);
    rnnoise_process_frame(st, x, x);
    for (int i = 0; i < FRAME_SIZE; i++)
      data[offset + i] = static_cast<short>(x[i]);
    offset += FRAME_SIZE;
  }
  rnnoise_destroy(st);
}

RnnoiseWrapper::RnnoiseWrapper() {
}
RnnoiseWrapper::~RnnoiseWrapper() {
}

}  // namespace BASE_NAMESPACE