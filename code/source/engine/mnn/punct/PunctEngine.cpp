#include "PunctEngine.h"
#include "utils/logging.h"
#include <map>
#include <sys/stat.h>

using namespace MNN;

namespace DMAI {

PunctEngine::PunctEngine() : is_init_(false) {}
PunctEngine::~PunctEngine() {
  net->releaseSession(session);
}

int PunctEngine::loadModel(const std::string &modelPath) {
  LOG(INFO) << "load model:" << modelPath;
  struct stat fs;
  if (stat(modelPath.c_str(), &fs) != 0) {
    LOG(ERROR)<<"model file error;"<<modelPath<<" not exist,please check!!!";
    return 1;
  }
  net.reset(Interpreter::createFromFile(modelPath.c_str()));
  if (!net) {
    LOG(ERROR)<<"load model "<<modelPath<<" error";
    return 1;
  }
  ScheduleConfig config;
  config.type = MNN_FORWARD_AUTO;
  session = net->createSession(config);
  net->resizeSession(session);
  LOG(INFO) << "load model success";
  is_init_ = true;
  return 0;
}

int PunctEngine::loadModel(const void *buffer, size_t size) {
  LOG(INFO) << "load model from buffer";
  net.reset(Interpreter::createFromBuffer(buffer, size));
  if (!net) {
    LOG(ERROR) << "load model error";
    return 1;
  }
  ScheduleConfig config;
  config.type = MNN_FORWARD_AUTO;
  session = net->createSession(config);
  net->resizeSession(session);
  LOG(INFO) << "load model success";
  is_init_ = true;
  return 0;
}

std::vector<float> PunctEngine::infer(const std::vector<float> &input) {
  LOG(INFO) << "PunctEngine::infer";
  std::vector<float> output;
  if (!is_init_) {
    LOG(ERROR) << "infer error;please call loadModel() before";
    return output;
  }
  std::vector<float> segs(input.size(), 0.0);
  std::map<std::string, Tensor *> model_inputs = net->getSessionInputAll(session);
  for (auto in : model_inputs) {
    int ids_len = input.size();
    net->resizeTensor(in.second, {1, ids_len});
    // in.second->printShape();
  }
  net->resizeSession(session);
  Tensor *t_token = net->getSessionInput(session, "input1");
  Tensor *t_segs = net->getSessionInput(session, "input2");
  memcpy(t_token->host<float>(), input.data(), t_token->size());
  memcpy(t_segs->host<float>(), segs.data(), t_segs->size());
  net->runSession(session);
  Tensor *model_output = net->getSessionOutput(session, NULL);
  int elementSize = model_output->elementSize();
  output.resize(elementSize);
  memcpy(output.data(), model_output->host<float>(), model_output->size());
  LOG(INFO) << "PunctEngine::infer end";
  return output;
}

};  // namespace DMAI
