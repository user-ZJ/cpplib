cmake_minimum_required(VERSION 3.14 FATAL_ERROR)


set(CMAKE_CXX_STANDARD 20)


find_package(CUDA REQUIRED)


include_directories(${CMAKE_CURRENT_SOURCE_DIR})
link_directories(/usr/local/cuda/lib64)

# 设置CUDA的编译选项
# set(CUDA_NVCC_FLAGS "-arch=sm_61")


aux_source_directory(. LAYER_SRC_LIST)
message("LAYER_SRC_LIST ${LAYER_SRC_LIST}")

cuda_add_library(culayer STATIC ${LAYER_SRC_LIST})
target_link_libraries(culayer ${CUDA_LIBRARIES} cudnn cublas)
