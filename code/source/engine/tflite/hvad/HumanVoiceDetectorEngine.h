#pragma once
#include<string>
#include<vector>
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "absl/base/attributes.h"
using std::string;

namespace BASE_NAMESPACE{

class HumanVoiceDetectorEngine
{
public:
    HumanVoiceDetectorEngine():input_dim(8000),is_init_(false) {}
    ~HumanVoiceDetectorEngine() {}
    int loadModel(string modelPath);
    std::vector<float> infer(const std::vector<float> &input);
private:
    bool is_init_;
    int input_dim;
    std::unique_ptr<tflite::FlatBufferModel> model_=nullptr;
    std::unique_ptr<tflite::Interpreter> interpreter_=nullptr; 
};

};

