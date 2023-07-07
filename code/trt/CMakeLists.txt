cmake_minimum_required(VERSION 3.14)
project(trtcode)
set(CMAKE_CXX_STANDARD 17)


include_directories(
	/data/code/TensorRT-8.5.3.1/include/
		 )
link_directories(
	/data/code/TensorRT-8.5.3.1/lib
		)


find_package(CUDA REQUIRED)



set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)


cuda_add_executable(trt_demo trt_demo.cpp)
target_link_libraries(trt_demo nvinfer nvinfer_plugin nvparsers nvonnxparser)

