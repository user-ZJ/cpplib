# tflite使用扩展算子--tensorflow算子

## 环境

```txt
ubuntu 18.04
tensorflow 2.4.1
python 3.7
gcc 7.5.0
```

## 配置步骤

1. 修改tensorflow/lite/BUILD，添加tensorflow/lite/BUILD//tensorflow/lite/delegates/flex:delegate依赖

   ```text
   tflite_cc_shared_object(
       name = "tensorflowlite",
       # Until we have more granular symbol export for the C++ API on Windows,
       # export all symbols.
       features = ["windows_export_all_symbols"],
       linkopts = select({
           "//tensorflow:macos": [
               "-Wl,-exported_symbols_list,$(location //tensorflow/lite:tflite_exported_symbols.lds)",
           ],
           "//tensorflow:windows": [], 
           "//conditions:default": [
               "-Wl,-z,defs",
               "-Wl,--version-script,$(location //tensorflow/lite:tflite_version_script.lds)",
           ],
       }), 
       per_os_targets = True,
       deps = [ 
           ":framework",
           ":tflite_exported_symbols.lds",
           ":tflite_version_script.lds",
           "//tensorflow/lite/kernels:builtin_ops_all_linked",
           "tensorflow/lite/BUILD//tensorflow/lite/delegates/flex:delegate",   --改行为添加项                                                                                                                                  
       ],  
   )
   ```

2. 编译libtensorflowlite.so

   ```sh
   bazel build -c opt  --config=monolithic tensorflow/lite:libtensorflowlite.so
   ```

3. 编写C++代码，调用libtensorflowlite.so加载模型，并调用推理，[代码参考](https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/lite/tools/benchmark/benchmark_plus_flex_main.cc)

   ```cpp
   #include <iostream>
   #include <random>
   #include "tensorflow/lite/model.h"
   #include "tensorflow/lite/profiling/profiler.h"
   #include "absl/base/attributes.h"
   #include "tensorflow/lite/op_resolver.h"
   #include "tensorflow/lite/kernels/register.h"
   #include "tensorflow/lite/tools/delegates/delegate_provider.h"
   
   
   using namespace std;
   
   
   void RegisterSelectedOps(::tflite::MutableOpResolver* resolver);
   
   // Version with Weak linker attribute doing nothing: if someone links this
   // library with another definition of this function (presumably to actually
   // register custom ops), that version will be used instead.
   void ABSL_ATTRIBUTE_WEAK
   RegisterSelectedOps(::tflite::MutableOpResolver* resolver) {}
   
   using VoidUniquePtr = std::unique_ptr<void, void (*)(void*)>;
   
   struct InputTensorData {
       InputTensorData() : data(nullptr, nullptr) {}
   
       VoidUniquePtr data;
       size_t bytes;
   };
   
   template <typename T, typename Distribution>
   inline InputTensorData CreateInputTensorData(int num_elements,
                                                Distribution distribution) {
     std::mt19937 random_engine_;
     InputTensorData tmp;
     tmp.bytes = sizeof(T) * num_elements;
     T* raw = new T[num_elements];
     std::generate_n(raw, num_elements, [&]() {
       return static_cast<T>(distribution(random_engine_));
     });
     tmp.data = VoidUniquePtr(static_cast<void*>(raw),
                              [](void* ptr) { delete[] static_cast<T*>(ptr); });
     return tmp;
   }
   
   int GetNumElements(const TfLiteIntArray* dim_array) {
     int num_elements = 1;
     //cout << "input dims:";
     for (size_t i = 0; i < dim_array->size; i++) {
       //cout<<dim_array->size<<",";
       num_elements *= dim_array->data[i];
     }
     //cout<<endl;
     return num_elements;
   }
   
   
   int main(int argc,char* argv[])
   {
       std::cout << "test tflite start " << std::endl;
       cout<<"load model:"<<argv[1]<<endl;
       std::unique_ptr<tflite::FlatBufferModel> model_ = tflite::FlatBufferModel::BuildFromFile(argv[1]);
       if (!model_) cout<<"load model error"<<endl;
       cout<<"build resolver"<<endl;
       auto resolver = new tflite::ops::builtin::BuiltinOpResolver();
       RegisterSelectedOps(resolver); 
       std::unique_ptr<tflite::Interpreter> interpreter_;
       cout<<"build Interpreter"<<endl;
       tflite::InterpreterBuilder(*model_, *resolver)(&interpreter_, 1);
       if (!interpreter_) cout<<"Failed to initialize the interpreter"<<endl;
       std::vector<tflite::Interpreter::TfLiteDelegatePtr> owned_delegates_;
       owned_delegates_.clear();
       for (const auto& delegate_provider : tflite::tools::GetRegisteredDelegateProviders()){
          cout<< "delegate_provider name "<<delegate_provider->GetName()<<std::endl;
       }
       cout<<"get inputs"<<endl;
       auto interpreter_inputs = interpreter_->inputs();
       cout<<"interpreter_inputs size"<<interpreter_inputs.size()<<endl;
       cout<<"allocate tensor"<<endl;
       if (interpreter_->AllocateTensors() != kTfLiteOk){
           cout<<"Failed to allocate tensors!"<<endl;
       }
       cout<<"random input data and set to model"<<endl;
       std::vector<InputTensorData> inputs_data_;
       for (int i = 0; i < interpreter_inputs.size(); ++i) {
           int tensor_index = interpreter_inputs[i];
           const TfLiteTensor& t = *(interpreter_->tensor(tensor_index));
           InputTensorData t_data;
           int num_elements = GetNumElements(t.dims);
           t_data = CreateInputTensorData<float>(num_elements, std::uniform_real_distribution<float>(-0.5f, 0.5f));
           inputs_data_.push_back(std::move(t_data));
           std::memcpy(t.data.raw, inputs_data_[i].data.get(),inputs_data_[i].bytes);
       }
       cout<<"run model"<<endl;
       interpreter_->Invoke();
       cout<<"get output"<<endl;
       auto interpreter_outputs = interpreter_->outputs();
       cout<<"interpreter_outputs size:"<<interpreter_outputs.size()<<endl;
       for(int i=0;i<interpreter_outputs.size();i++){
           int tensor_index = interpreter_outputs[i];
           const TfLiteTensor & t = *(interpreter_->tensor(tensor_index));
           //cout<<"output size:";
           //for(int j=0;j<t.dims->size;j++)
           //    cout<<t.dims->data[j]<<",";
           //cout<<endl;
           for(int j=0;j<GetNumElements(t.dims);j++)
               cout<<t.data.f[j]<<",";
           cout<<endl;
   
       }
       return 0;
   }
   ```

4. 编写CMakeLists.txt

   ```cmake
   cmake_minimum_required(VERSION 2.8)
   project(tflite_feat)
   
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
   include_directories(
   	${CMAKE_SOURCE_DIR}
       /home/zack/sourceCode/tensorflow/tensorflow-24/
       /home/zack/.cache/bazel/_bazel_zack/84b40c743e9b60541d123357b44e545f/external/flatbuffers/include
       /home/zack/.cache/bazel/_bazel_zack/84b40c743e9b60541d123357b44e545f/external/com_google_absl
   		 )
   
   link_directories(
           /usr/local/lib
           ${CMAKE_SOURCE_DIR}/tools/libs
           )
   
   add_executable(tflitetest tflitetest.cpp)
   target_link_libraries(tflitetest tensorflowlite)
   INSTALL(TARGETS tflitetest RUNTIME DESTINATION bins)
   ```

   编译完成后生成tflitetest的bin

5. 使用带有tensorflow算子的模型进行验证，tensorflow hub中提供了一个pitch的模型[spice](https://tfhub.dev/google/spice/2)可以进行此项验证，[下载模型](https://tfhub.dev/google/spice/2?tf-hub-format=compressed)，解压后保存到saved_model文件夹，使用如下脚本将模型转换为tflite

   ```python
   #!/usr/bin/env python
   # coding=utf-8
   import tensorflow_hub as hub
   import tensorflow as tf
   import numpy as np
   import matplotlib.pyplot as plt
   
   model = tf.saved_model.load("saved_model")  #加载出来的模型输入为None，需要设定输入的shape
   concrete_func = model.signatures[
     tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
   concrete_func.inputs[0].set_shape([2048])
   converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
   converter.target_spec.supported_ops = [
     tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
     tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
   ]
   tflite_model = converter.convert()
   with open('spice_model.tflite', 'wb') as f:
     f.write(tflite_model)
   
   interpreter = tf.lite.Interpreter("spice_model.tflite")
   input_details = interpreter.get_input_details()
   output_details = interpreter.get_output_details()
   print(input_details)
   interpreter.resize_tensor_input(0, [2048], strict=False)
   interpreter.allocate_tensors()
   input_details = interpreter.get_input_details()
   input_shape = input_details[0]['shape']
   print(input_shape)
   output_details = interpreter.get_output_details()
   print(output_details)
   input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
   interpreter.set_tensor(input_details[0]['index'], input_data)
   interpreter.invoke()
   ```

6. 验证

   ```shell
   ./tflitetest path/to/spice_model.tflite
   ```

   