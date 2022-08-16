#include "HumanVoiceDetectorEngine.h"
#include "utils/logging.h"
#include <string>
#include <vector>

using std::string;

namespace BASE_NAMESPACE {

void RegisterSelectedOps(::tflite::MutableOpResolver *resolver);
void ABSL_ATTRIBUTE_WEAK RegisterSelectedOps(::tflite::MutableOpResolver *resolver) {}

int GetNumElements(const TfLiteIntArray *dim_array) {
  int num_elements = 1;
  for (size_t i = 0; i < dim_array->size; i++) {
    num_elements *= dim_array->data[i];
  }
  return num_elements;
}

int HumanVoiceDetectorEngine::loadModel(string modelPath) {
  // Load the model
  LOG(INFO) << "load model:" << modelPath;
  model_ = std::move(tflite::FlatBufferModel::BuildFromFile(modelPath.c_str()));
  if (!model_) {
    LOG(INFO) << "load model " << modelPath << " error\n";
    return 1;
  }
  auto resolver = new tflite::ops::builtin::BuiltinOpResolver();
  RegisterSelectedOps(resolver);
  LOG(INFO) << "create interpreter";
  tflite::InterpreterBuilder(*model_, *resolver)(&interpreter_, 1);
  if (!interpreter_) {
    LOG(INFO) << "create interpreter error\n";
    return 1;
  }
  LOG(INFO) << "allocate tensor";
  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    LOG(INFO) << "Failed to allocate tensors!\n";
    return 1;
  }
  is_init_ = true;
  LOG(INFO) << "loadModel success";
  return 0;
}

std::vector<float> HumanVoiceDetectorEngine::infer(const std::vector<float> &input) {
  std::vector<float> output;
  if (!is_init_) {
    LOG(INFO) << "infer error;please call loadModel() before";
    return output;
  }
  if (input.size() != input_dim) {
    LOG(INFO) << "input error; input dim must be " << input_dim;
    return output;
  }
  auto interpreter_inputs = interpreter_->inputs();
  // LOG(INFO)<<"set input";
  std::memcpy(interpreter_->tensor(interpreter_inputs[0])->data.raw, input.data(), input.size() * sizeof(float));
  // float *input = interpreter_->typed_input_tensor<float>(0);
  //  LOG(INFO)<<"invoke";
  interpreter_->Invoke();
  // LOG(INFO)<<"get output";
  auto interpreter_outputs = interpreter_->outputs();
  int out_index = interpreter_outputs[0];
  output.resize(GetNumElements(interpreter_->tensor(out_index)->dims));
  std::memcpy(output.data(), interpreter_->tensor(out_index)->data.raw, output.size() * sizeof(float));
  return output;
}

};  // namespace BASE_NAMESPACE
