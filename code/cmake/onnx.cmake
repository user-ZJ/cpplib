include(local_library)
mbase_local_library(onnxruntime onnxruntime_cxx_api.h)
add_definitions(-DUSE_ONNX)
