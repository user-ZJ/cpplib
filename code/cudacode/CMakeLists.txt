cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(mycuda)

set(CMAKE_CXX_STANDARD 17)

message(STATUS "opration system is ${CMAKE_SYSTEM_NAME}")
if(CMAKE_SYSTEM_NAME MATCHES "Android")
    set(TARGET_ARCH ${ANDROID_ABI})
    if(CMAKE_ANDROID_ARCH_ABI MATCHES "armeabi-v7a")
    elseif(CMAKE_ANDROID_ARCH_ABI MATCHES "arm64-v8a")
    endif()
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    if(ARCH_32)
        set(TARGET_ARCH x86)
    else()
        set(TARGET_ARCH x64)
    endif()
else()
    message(FATAL_ERROR "not support system ${CMAKE_SYSTEM_NAME}")
endif()


set(CUDA_NVCC_FLAGS "-G;-g")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DDEBUG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fPIC -Wfatal-errors")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -Wfatal-errors")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -Wfatal-errors -Wall -Wextra -Werror -Wconversion -Wshadow")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -Wfatal-errors")
    # -DNDEBUG 禁用assert
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wfatal-errors")
endif()

find_package(CUDA REQUIRED)

add_definitions(-DUSE_CUDA=1)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


include_directories(
    /usr/local/cuda/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
link_directories(/usr/local/cuda/lib64)

set(CMAKE_VERBOSE_MAKEFILE on)

enable_language(CUDA)

# 设置 NVCC 编译选项
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -m64 -lineinfo --resource-usage")


# 设置所需的计算能力
set(CMAKE_CUDA_ARCHITECTURES "61;75")


add_subdirectory(layers)

cuda_add_executable(test test.cu mykernel.cu)
target_link_libraries(test ${CUDA_LIBRARIES})
# target_compile_options(test  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#                         -m64 -lineinfo --resource-usage 
#                         -gencode arch=compute_52,code=sm_52 
#                         -gencode arch=compute_60,code=sm_60 
#                         -gencode arch=compute_61,code=sm_61 
#                         -gencode arch=compute_70,code=sm_70 
#                         -gencode arch=compute_75,code=sm_75 
#                         -gencode arch=compute_80,code=sm_80
#                        >)


cuda_add_executable(cudnn_dropout cudnn_dropout.cpp)
target_link_libraries(cudnn_dropout ${CUDA_LIBRARIES} cudnn)

cuda_add_executable(cublas_gemm cublas_gemm.cpp)
target_link_libraries(cublas_gemm ${CUDA_LIBRARIES} cublas)


cuda_add_executable(cutest cutest.cu)
target_link_libraries(cutest culayer)


add_subdirectory(base)
add_subdirectory(kernels)
add_subdirectory(unittest)
