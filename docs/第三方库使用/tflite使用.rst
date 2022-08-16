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
