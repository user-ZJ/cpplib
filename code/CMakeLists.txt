cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(mbase)

option(BUILD_UNITTEST "build test with googletest" OFF)
option(BUILD_TEST "build test with googletest" ON)

set(BASE_NAMESPACE zack)

set(CMAKE_CXX_STANDARD 17)

message(STATUS "opration system is ${CMAKE_SYSTEM_NAME}")
#${CMAKE_SYSTEM_NAME} "Windows", "Linux","Android" or "Darwin"
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
    if(MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /DEBUG")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DEBUG")
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
        add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fPIC -Wfatal-errors")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -Wfatal-errors")
        # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -Wfatal-errors -Wall -Wextra -Werror -Wconversion -Wshadow")
    endif()
else()
    if (MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
        add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -Wfatal-errors")
        # -DNDEBUG 禁用assert
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wfatal-errors")
    endif()
endif()

option(USE_MNN "whether to build MNN" OFF)
option(USE_TFLITE "whether to build tflite" OFF)
option(USE_ONNXRUNTIME "whether to build ONNXRUNTIME" ON)
# 设置所有编译出的库为静态库
set(BUILD_SHARED_LIBS OFF CACHE BOOL "global variable")

include(FetchContent)
include(ExternalProject)
set(FETCHCONTENT_QUIET off)
# set(fc_base $ENV{HOME}/fc_base CACHE STRING "global variable")
# set(FETCHCONTENT_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fc_base)
# set(FETCHCONTENT_BASE_DIR ${fc_base})
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/fc_base)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


include(glog)
include(gflags)

# third_party: boost
include(boost)

include(opencc)

include_directories(
    # 3rdparty/gflags/include
    # 3rdparty/glog/include
    3rdparty/poco/include 
    3rdparty/fvad/include
    3rdparty/rnnoise/include
    3rdparty/opus/include  
)
link_directories(
    # 3rdparty/gflags/lib/${CMAKE_SYSTEM_NAME}/${TARGET_ARCH}
    # 3rdparty/glog/lib/${CMAKE_SYSTEM_NAME}/${TARGET_ARCH}
    3rdparty/poco/lib/${CMAKE_SYSTEM_NAME}/${TARGET_ARCH}
    3rdparty/fvad/lib/${CMAKE_SYSTEM_NAME}/${TARGET_ARCH}
    3rdparty/rnnoise/lib/${CMAKE_SYSTEM_NAME}/${TARGET_ARCH}
    3rdparty/opus/lib/${CMAKE_SYSTEM_NAME}/${TARGET_ARCH}
)

# third_party: minizip
include(minizip)


# third_party: rapidjson
include(rapidjson)


# third_party: eigen
include(eigen)


# # third_party: cryptopp
# FetchContent_Declare(cryptopp
#   URL      https://cryptopp.com/cryptopp860.zip
#   URL_HASH SHA256=20aa413957d9c8ae353ee2f7747bd7ac392f933c60a53e3fd1e41cadbc48d193
# )
# FetchContent_MakeAvailable(cryptopp)
# include_directories(${cryptopp_BINARY_DIR}/include)

# set(cryptopp_SRC_DIR ${FETCHCONTENT_BASE_DIR}/cryptopp-src)
# set(cryptopp_BUILD_DIR ${FETCHCONTENT_BASE_DIR}/cryptopp-build)
# set(cryptopp_PREFIX_DIR ${FETCHCONTENT_BASE_DIR}/cryptopp-subbuild/cryptopp-populate-prefix)
# ExternalProject_Add(cryptopp
#     URL               https://cryptopp.com/cryptopp860.zip
#     URL_HASH          SHA256=20aa413957d9c8ae353ee2f7747bd7ac392f933c60a53e3fd1e41cadbc48d193
#     PREFIX            ${cryptopp_PREFIX_DIR}
#     # DOWNLOAD_DIR      ${cryptopp_PREFIX_DIR}
#     SOURCE_DIR        ${cryptopp_SRC_DIR}
#     BINARY_DIR        ${cryptopp_SRC_DIR}
#     CONFIGURE_COMMAND ""
#     BUILD_COMMAND     make -j 4
#     INSTALL_COMMAND   make install PREFIX=${cryptopp_BUILD_DIR}
# )
# include_directories(${cryptopp_SRC_DIR})
# link_directories(${cryptopp_SRC_DIR})


include_directories(
    source
	source/utils
		 )
link_directories(
		)


# aux_source_directory(source/tokenizer Tokenizer_SRC)
# aux_source_directory(source/utils UTILS_SOURCES)



# add_library(${PROJECT_NAME} OBJECT ${Tokenizer_SRC} ${UTILS_SOURCES})

#链接静态库，需要使用-Wl,-whole-archive,其他动态库不需要，在退出时需要使用-Wl,--no-whole-archive还原
# 所以静态库链接形式为-Wl,-whole-archive static_libs -Wl,--no-whole-archive
# target_link_libraries(${PROJECT_NAME}
#         minizip
#         glog
#         gflags
#         pthread
# 		)
# INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
# add_dependencies(${PROJECT_NAME} gflags_module glog_module)
add_subdirectory(source/utils)
add_subdirectory(source/web)
# add_subdirectory(source/grpc)
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




