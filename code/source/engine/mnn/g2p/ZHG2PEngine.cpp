#include "ZHG2PEngine.h"
#include "utils/logging.h"
#include <map>

using namespace MNN;

namespace BASE_NAMESPACE {

ZHG2PStartEngine::ZHG2PStartEngine() : is_init_(false) {}
ZHG2PStartEngine::~ZHG2PStartEngine() {
  net->releaseSession(session);
}

int ZHG2PStartEngine::loadModel(const std::string &modelPath) {
  LOG(INFO) << "load model:" << modelPath;
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

int ZHG2PStartEngine::loadModel(const char *buffer, size_t size) {
  LOG(INFO) << "load model from buffer";
  net.reset(Interpreter::createFromBuffer(buffer, size));
  if (!net) {
    LOG(ERROR) << "load model from buffer error";
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

vector<float> ZHG2PStartEngine::infer(const vector<int> &ids) {
  LOG(INFO) << "ZHG2PStartEngine::infer";
  vector<float> output;
  if (!is_init_) {
    LOG(ERROR) << "infer error;please call loadModel() before";
    return output;
  }
  std::vector<int> segs(ids.size(), 0);
  int id_num = ids.size();
  if (id_num == 0)
    return output;
  std::map<std::string, Tensor *> model_inputs = net->getSessionInputAll(session);
  net->resizeTensor(model_inputs["ids"], {1, id_num});
  net->resizeTensor(model_inputs["segs"], {1, id_num});
  // for(auto in:model_inputs){
  //     auto shape = in.second->shape();
  //     shape[1] = input.size();
  //     LOG(INFO)<<"resize tensor:"<<in.first;
  //     net->resizeTensor(in.second, shape);
  //     in.second->printShape();
  // }
  net->resizeSession(session);
  Tensor *t_token = net->getSessionInput(session, "ids");
  Tensor *t_segs = net->getSessionInput(session, "segs");
  memcpy(t_token->host<int>(), ids.data(), t_token->size());
  memcpy(t_segs->host<int>(), segs.data(), t_segs->size());
  net->runSession(session);
  auto model_out = net->getSessionOutput(session, NULL);
  // std::map<std::string,Tensor *> model_outputs = net->getSessionOutputAll(session);
  // for(auto out:model_outputs){
  //     Tensor *t = out.second;
  //     int elementSize = t->elementSize();
  //     vector<int> o(elementSize);
  //     memcpy(o.data(),t->host<int>(),t->size());
  //     //outputs.emplace_back(o);
  // }
  output.resize(model_out->elementSize());
  memcpy(output.data(), model_out->host<int>(), model_out->size());
  LOG(INFO) << "ZHG2PStartEngine::infer end";
  return output;
}

ZHG2PEndEngine::ZHG2PEndEngine() : is_init_(false) {}
ZHG2PEndEngine::~ZHG2PEndEngine() {
  net->releaseSession(session);
}

int ZHG2PEndEngine::loadModel(const std::string &modelPath) {
  LOG(INFO) << "load model:" << modelPath;
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

int ZHG2PEndEngine::loadModel(const char *buffer, size_t size) {
  LOG(INFO) << "load model from buffer";
  net.reset(Interpreter::createFromBuffer(buffer, size));
  if (!net) {
    LOG(ERROR) << "load model from buffer error";
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

vector<int> ZHG2PEndEngine::infer(const vector<float> &input, const vector<int> &x_ids, const vector<int> &x_cm) {
  LOG(INFO) << "ZHG2PEndEngine::infer";
  vector<int> output;
  if (!is_init_) {
    LOG(ERROR) << "infer error;please call loadModel() before";
    return output;
  }
  int id_num = input.size() / 256, dyz_num = x_ids.size();
  if (dyz_num == 0)
    return output;
  std::map<std::string, Tensor *> model_inputs = net->getSessionInputAll(session);
  net->resizeTensor(model_inputs["idx"], {1, dyz_num});
  net->resizeTensor(model_inputs["cm"], {1, dyz_num, 6});
  net->resizeTensor(model_inputs["bert_o"], {1, id_num, 256});
  // for(auto in:model_inputs){
  //     auto shape = in.second->shape();
  //     shape[1] = input.size();
  //     LOG(INFO)<<"resize tensor:"<<in.first;
  //     net->resizeTensor(in.second, shape);
  //     in.second->printShape();
  // }
  net->resizeSession(session);
  Tensor *t_ids = net->getSessionInput(session, "idx");
  Tensor *t_cm = net->getSessionInput(session, "cm");
  Tensor *t_bert = net->getSessionInput(session, "bert_o");
  memcpy(t_ids->host<int>(), x_ids.data(), t_ids->size());
  memcpy(t_cm->host<int>(), x_cm.data(), t_cm->size());
  memcpy(t_bert->host<float>(), input.data(), t_bert->size());
  net->runSession(session);
  auto model_out = net->getSessionOutput(session, NULL);
  // std::map<std::string,Tensor *> model_outputs = net->getSessionOutputAll(session);
  // for(auto out:model_outputs){
  //     Tensor *t = out.second;
  //     int elementSize = t->elementSize();
  //     vector<int> o(elementSize);
  //     memcpy(o.data(),t->host<int>(),t->size());
  //     //outputs.emplace_back(o);
  // }
  output.resize(model_out->elementSize());
  memcpy(output.data(), model_out->host<int>(), model_out->size());
  LOG(INFO) << "ZHG2PEndEngine::infer end";
  return output;
}

};  // namespace BASE_NAMESPACE
