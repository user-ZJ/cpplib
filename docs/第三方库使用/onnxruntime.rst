onnxruntime
===============

安装
-----------------

`参考 <https://onnxruntime.ai/docs/build/inferencing.html>`_

需要安装cmake 3.18以上

.. code-block:: shell

    git clone --recursive https://github.com/Microsoft/onnxruntime
    cd onnxruntime
    ./build.sh --config RelWithDebInfo --build_shared_lib --parallel  #编译debug版本
    ./build.sh --config Release --build_shared_lib --parallel  #编译release版本

    source /opt/intel/openvino_2021/bin/setupvars.sh
    ./build.sh --config RelWithDebInfo --use_openvino CPU_FP32 --build_shared_lib --parallel #编译openvino后端版本，需要先安装好openvino




openvino使用

https://github.com/yas-sim/openvino-ep-enabled-onnxruntime
