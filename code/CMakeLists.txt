cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(mbase)

option(BUILD_UNITTEST "build test with googletest" ON)
option(BUILD_TEST "build module examples" ON)

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
# else()
#     message(FATAL_ERROR "not support system ${CMAKE_SYSTEM_NAME}")
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

set(MBASE_DEPS_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/.local-deps" CACHE PATH "Locally installed third-party dependencies")
list(PREPEND CMAKE_PREFIX_PATH "${MBASE_DEPS_PREFIX}" /opt/homebrew /usr/local)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


include(glog)
include(gflags)
# kaldi 依赖 /opt/intel/mkl（见 source/kaldi），找不到时跳过 openfst、kaldi 及依赖 kaldi 的 vad
find_path(MKL_INCLUDE_DIR mkl.h PATHS /opt/intel/mkl/include NO_DEFAULT_PATH)
if(MKL_INCLUDE_DIR)
    include(openfst)
else()
    message(STATUS "MKL not found in /opt/intel/mkl, skip openfst, source/kaldi and source/vad")
endif()
include(boost)
# include(opencc)
include(rapidjson)
include(eigen)
include(poco)
if(USE_ONNXRUNTIME)
    include(onnx)
endif()
# include(flatbuffers)
include(cereal)
include(sox)
include(rnnoise)


#链接静态库，需要使用-Wl,-whole-archive,其他动态库不需要，在退出时需要使用-Wl,--no-whole-archive还原
# 所以静态库链接形式为-Wl,-whole-archive static_libs -Wl,--no-whole-archive
# target_link_libraries(${PROJECT_NAME}
#         minizip
#         glog
#         gflags
#         pthread
# 		)
add_subdirectory(source/utils)
if(MKL_INCLUDE_DIR)
    add_subdirectory(source/kaldi)
endif()
if(TARGET Poco::DataMySQL)
    add_subdirectory(source/db)
else()
    message(STATUS "MySQL client not found, skip source/db and its tests")
endif()
if(MKL_INCLUDE_DIR)
    add_subdirectory(source/vad)
endif()
add_subdirectory(source/remove_noise)
add_subdirectory(source/opus)
add_subdirectory(source/crypto)
add_subdirectory(apps)

if(BUILD_TEST)
    add_subdirectory(examples)
endif()



if(BUILD_UNITTEST)
    find_package(GTest CONFIG REQUIRED)
    enable_testing()
    add_subdirectory(tests)
endif()





# Configure only first-party targets after their subdirectories are defined.
include(project_targets)
foreach(module utils kaldi db vad remove_noise opus crypto)
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/source/${module}")
        get_property(configured_dirs DIRECTORY PROPERTY SUBDIRECTORIES)
        set(module_dir "${CMAKE_CURRENT_SOURCE_DIR}/source/${module}")
        if(module_dir IN_LIST configured_dirs)
            mbase_configure_targets("${module_dir}")
        endif()
    endif()
endforeach()
mbase_configure_targets("${CMAKE_CURRENT_SOURCE_DIR}/apps")
if(BUILD_TEST)
    mbase_configure_targets("${CMAKE_CURRENT_SOURCE_DIR}/examples")
endif()
if(BUILD_UNITTEST)
    mbase_configure_targets("${CMAKE_CURRENT_SOURCE_DIR}/tests")
endif()
