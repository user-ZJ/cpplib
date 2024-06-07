cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(mbase)

option(BUILD_UNITTEST "build test with googletest" OFF)
option(BUILD_TEST "build test with googletest" ON)

set(BASE_NAMESPACE zack)

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

option(USE_MNN "whether to build MNN" OFF)
option(USE_TFLITE "whether to build tflite" OFF)
option(USE_ONNXRUNTIME "whether to build ONNXRUNTIME" ON)


# set(CMAKE_VERBOSE_MAKEFILE on)

# 设置所有编译出的库为静态库
# set(BUILD_SHARED_LIBS ON CACHE BOOL "global variable")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib_static)

include(FetchContent)
include(ExternalProject)
set(FETCHCONTENT_QUIET off)
# set(fc_base $ENV{HOME}/fc_base CACHE STRING "global variable")
set(FETCHCONTENT_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fc_base)
# set(FETCHCONTENT_BASE_DIR ${fc_base})
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


include(glog)
include(gflags)
include(openfst)
include(boost)
# include(opencc)
include(rapidjson)
include(eigen)
include(poco)
include(onnx)
# include(flatbuffers)
include(cereal)

execute_process(COMMAND unzip -n ${CMAKE_SOURCE_DIR}/3rdparty/grpc_ubuntu20_install.zip -d ${CMAKE_BINARY_DIR}/grpc)


include_directories(
    /opt/intel/mkl/include
    source
    source/kaldi
    3rdparty/poco/include 
    3rdparty/fvad/include
    3rdparty/rnnoise/include
    3rdparty/opus/include  
)
link_directories(
    # /usr/local/lib
    /opt/intel/mkl/lib/intel64
    /opt/intel/lib/intel64/
    # 3rdparty/poco/lib/${CMAKE_SYSTEM_NAME}/${TARGET_ARCH}
    3rdparty/fvad/lib/${CMAKE_SYSTEM_NAME}/${TARGET_ARCH}
    3rdparty/rnnoise/lib/${CMAKE_SYSTEM_NAME}/${TARGET_ARCH}
    3rdparty/opus/lib/${CMAKE_SYSTEM_NAME}/${TARGET_ARCH}
)


#链接静态库，需要使用-Wl,-whole-archive,其他动态库不需要，在退出时需要使用-Wl,--no-whole-archive还原
# 所以静态库链接形式为-Wl,-whole-archive static_libs -Wl,--no-whole-archive
# target_link_libraries(${PROJECT_NAME}
#         minizip
#         glog
#         gflags
#         pthread
# 		)
add_subdirectory(source/utils)
add_subdirectory(source/kaldi)
add_subdirectory(source/web)
add_subdirectory(source/grpc)
add_subdirectory(source/db)
add_subdirectory(source/vad)
add_subdirectory(source/remove_noise)
add_subdirectory(source/opus)
add_subdirectory(source/crypto)
add_subdirectory(source/bin)

if(BUILD_TEST)
    add_subdirectory(test)
endif()



if(BUILD_UNITTEST)
    FetchContent_Declare(googletest
	    URL      https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz
	    URL_HASH SHA256=b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5
	    # SOURCE_DIR ../thirdpart/googletest
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(GTEST_HAS_PTHREAD 1)
    FetchContent_MakeAvailable(googletest) #后续链接gtest_main即可
    enable_testing()
    add_subdirectory(unittest)
endif()




