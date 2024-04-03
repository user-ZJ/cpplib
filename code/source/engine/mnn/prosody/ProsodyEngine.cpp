#include "ProsodyEngine.h"
#include "utils/logging.h"
#include <map>
#include <sys/stat.h>

using namespace MNN;

namespace BASE_NAMESPACE {

ProsodyEngine::ProsodyEngine() : is_init_(false) {}
ProsodyEngine::~ProsodyEngine() {
  net->releaseSession(session);
}

int ProsodyEngine::loadModel(const std::string &modelPath) {
  LOG(INFO) << "load model:" << modelPath;
  struct stat fs;
  if (stat(modelPath.c_str(), &fs) != 0) {
    LOG(ERROR) << "model file error;" << modelPath << " not exist,please check!!!";
    return 1;
  }
  net.reset(Interpreter::createFromFile(modelPath.c_str()));
  if (!net) {
    LOG(ERROR) << "load model " << modelPath << " error";
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

int ProsodyEngine::loadModel(const void *buffer, size_t size) {
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

vector<vector<float>> ProsodyEngine::infer(const vector<int> &toks) {
  LOG(INFO) << "ProsodyEngine::infer";
  vector<vector<float>> outputs;
  if (!is_init_) {
    LOG(ERROR) << "infer error;please call loadModel() before";
    return outputs;
  }
  std::vector<int> segs(toks.size(), 0);
  std::map<std::string, Tensor *> model_inputs = net->getSessionInputAll(session);
  for (auto in : model_inputs) {
    auto shape = in.second->shape();
    shape[1] = toks.size();
    LOG(INFO) << "resize tensor:" << in.first;
    net->resizeTensor(in.second, shape);
    // in.second->printShape();
  }
  net->resizeSession(session);
  Tensor *t_token = net->getSessionInput(session, "toks");
  Tensor *t_segs = net->getSessionInput(session, "segs");
  memcpy(t_token->host<int>(), toks.data(), t_token->size());
  memcpy(t_segs->host<int>(), segs.data(), t_segs->size());
  net->runSession(session);
  std::map<std::string, Tensor *> model_outputs = net->getSessionOutputAll(session);
  for (auto out : model_outputs) {
    Tensor *t = out.second;
    int elementSize = t->elementSize();
    vector<float> o(elementSize);
    memcpy(o.data(), t->host<float>(), t->size());
    outputs.emplace_back(o);
  }
  LOG(INFO) << "ProsodyEngine::infer end";
  return outputs;
}

};  // namespace BASE_NAMESPACE
