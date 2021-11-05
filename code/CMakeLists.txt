cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(mbase)

option(BUILD_UNITTEST "build test with googletest" ON)
option(BUILD_TEST "build test with googletest" ON)

set(BASE_NAMESPACE zack)

set(CMAKE_CXX_STANDARD 11)

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
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fPIC -Wfatal-errors")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -Wfatal-errors")
    endif()
else()
    if (MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -Wfatal-errors")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wfatal-errors")
    endif()
endif()


include(FetchContent)
set(FETCHCONTENT_QUIET off)
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/thirdpart)


# third_party: boost
FetchContent_Declare(boost
  URL      https://boostorg.jfrog.io/artifactory/main/release/1.75.0/source/boost_1_75_0.tar.gz
  URL_HASH SHA256=aeb26f80e80945e82ee93e5939baebdca47b9dee80a07d3144be1e1a6a66dd6a
)
FetchContent_MakeAvailable(boost)
include_directories(${boost_SOURCE_DIR})


# third_party: gflags
FetchContent_Declare(gflags
  URL      https://github.com.cnpmjs.org/gflags/gflags/archive/v2.2.0.zip
  URL_HASH SHA256=99f9e8b63ea53e4b23f0fd0f68a6f1e397d5512be36716c17cc75966a90f0d57
)
FetchContent_MakeAvailable(gflags)
include_directories(${gflags_BINARY_DIR}/include)

# third_party: glog
FetchContent_Declare(glog
  URL      https://github.com/google/glog/archive/refs/tags/v0.5.0.zip
  URL_HASH SHA256=21bc744fb7f2fa701ee8db339ded7dce4f975d0d55837a97be7d46e8382dea5a
)
set(WITH_CUSTOM_PREFIX ON CACHE BOOL "Always with custom prefix" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "disable test" FORCE)
FetchContent_MakeAvailable(glog)
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})

# third_party: minizip
FetchContent_Declare(minizip
  GIT_REPOSITORY    https://github.com/zlib-ng/minizip-ng.git
  GIT_TAG   99d39015e29703af2612277180ea586805f655ea
)
#add_subdirectory(${minizip_SOURCE_DIR})
FetchContent_MakeAvailable(minizip)
include_directories(${minizip_SOURCE_DIR})



include_directories(
	source/utils
		 )
link_directories(
		)


aux_source_directory(source/Tokenizer Tokenizer_SRC)
aux_source_directory(source/utils UTILS_SOURCES)



add_library(${PROJECT_NAME} SHARED ${Tokenizer_SRC} ${UTILS_SOURCES})

target_link_libraries(${PROJECT_NAME}
		glog
        minizip
		)
INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

add_subdirectory(source/web)

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

