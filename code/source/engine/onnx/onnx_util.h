/*
 * @Author: zack 
 * @Date: 2022-08-15 16:53:35 
 * @Last Modified by: zack
 * @Last Modified time: 2022-08-15 16:54:13
 */


#pragma once
#include "onnxruntime_cxx_api.h"

class ONNXENV {
 private:
  ONNXENV(){};
  ~ONNXENV(){};
  ONNXENV(const ONNXENV &);
  ONNXENV &operator=(const ONNXENV &);

 public:
  static Ort::Env &getInstance() {
    static Ort::Env env(/*envOpts,*/ ORT_LOGGING_LEVEL_WARNING, "test");
    //    Ort::Global<void>::api_.ReleaseThreadingOptions(envOpts);
    return env;
  }
};