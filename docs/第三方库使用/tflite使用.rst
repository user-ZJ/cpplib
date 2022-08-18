========================
tflite使用
========================

模型转换
=================

keras模型转tflite
----------------------

.. code-block:: python 

    import tensorflow as tf
    import os
    os.environ['TF_KERAS'] = 'true'
    tf.compat.v1.enable_eager_execution()

    model.input[0].set_shape((1,122))
    model.input[1].set_shape((1,122))
    model.save("mymodel")


    reconstructed_model = tf.keras.models.load_model("mymodel")
    reconstructed_model.input[0].set_shape((1,122))
    reconstructed_model.input[1].set_shape((1,122))
    converter = tf.lite.TFLiteConverter.from_keras_model(reconstructed_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

    interpreter = tf.lite.Interpreter("model.tflite")


tflite使用select算子
============================

`tf select 官方说明 <https://www.tensorflow.org/lite/guide/ops_select?hl=zh-cn>`_

官方配置
------------------
1. 修改tensorflow/lite/BUILD，添加tensorflow/lite/BUILD//tensorflow/lite/delegates/flex:delegate依赖

:: 

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
            "tensorflow/lite/BUILD//tensorflow/lite/delegates/flex:delegate",   --该行为添加项                                                                                                                                  
        ],  
    )

2. 编译libtensorflowlite.so

.. code-block:: shell

    bazel build -c opt  --config=monolithic tensorflow/lite:libtensorflowlite.so

3. 编写C++代码，调用libtensorflowlite.so加载模型，并调用推理， `代码参考 <https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/lite/tools/benchmark/benchmark_plus_flex_main.cc>`_

.. code-block:: cpp

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

4. 编写CMakeLists.txt

.. code-block:: cmake

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

编译完成后生成tflitetest的bin

5. 使用带有tensorflow算子的模型进行验证，tensorflow hub中提供了一个pitch的模型 `spice <https://tfhub.dev/google/spice/2>`_ 可以进行此项验证，
`下载模型 <https://tfhub.dev/google/spice/2?tf-hub-format=compressed>`_ ，解压后保存到saved_model文件夹，使用如下脚本将模型转换为tflite

.. code-block:: python

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

6. 验证

.. code-block:: shell

    ./tflitetest path/to/spice_model.tflite

tflite使用select算子后，Android库压缩
---------------------------------------

tflite使用Android库后，根据官方文档添加tensorflow/lite/delegates/flex:delegate依赖，会导致编译出来的libtensorflowlite.so库有150M，
因为在编译的时候把tensorflow的所有select算子都编译进去了

**修改方案：**

1. 在tensorflow源码的根目录下创建myflex目录
2. 在myflex目录下创建ops_list.txt，添加需要用到的select算子
    ["All","Erf"]
3. 将tflite模型拷贝到myflex目录
4. 添加BUILD文件

.. code-block:: 

    load("//tensorflow/lite/delegates/flex:build_def.bzl", "tflite_flex_cc_library")

    tflite_flex_cc_library(
        name = "my_tensorflowlite_flex",
        models=["addpunct_zh.tflite",],
        visibility = ["//visibility:public"],
    )

5. 在tensorflow/lite/BUILD中第778行添加依赖

.. code-block:: 

    flite_cc_shared_object(                                                                                                                                                                                                                                                           
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
            "//myflex:my_tensorflowlite_flex" ,         #添加内容
        ],  
    )

6. 重新编译Android库
.. code-block::
    
    编译64位库
    bazel build -c opt --cxxopt='--std=c++14' --config=monolithic --config=android_arm64 
          --host_crosstool_top=@bazel_tools//tools/cpp:toolchain //tensorflow/lite:libtensorflowlite.so
    编译32位库
    bazel build -c opt --cxxopt='--std=c++14' --config=monolithic --config=android_arm32 
          --host_crosstool_top=@bazel_tools//tools/cpp:toolchain //tensorflow/lite:libtensorflowlite.so
