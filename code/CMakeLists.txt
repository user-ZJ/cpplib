cmake_minimum_required(VERSION 3.14)
project(mbase)

set(BASE_NAMESPACE zack)

if(CMAKE_BUILD_TYPE MATCHES Debug)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fPIC")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -s -fPIC")
endif()

set(CMAKE_CXX_STANDARD 11)


include(FetchContent)
set(FETCHCONTENT_QUIET off)
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/thirdpart)


# third_party: glog
FetchContent_Declare(glog
  URL      https://github.com/google/glog/archive/refs/tags/v0.5.0.zip
  URL_HASH SHA256=21bc744fb7f2fa701ee8db339ded7dce4f975d0d55837a97be7d46e8382dea5a
)
set(WITH_CUSTOM_PREFIX ON)
set(BUILD_TESTING OFF)
FetchContent_MakeAvailable(glog)
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})

include_directories(
	source/utils
		 )
link_directories(
		)


aux_source_directory(source/utils UTILS_SOURCES)


add_library(${PROJECT_NAME} SHARED ${UTILS_SOURCES})

target_link_libraries(${PROJECT_NAME}
		glog
		)
INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


add_subdirectory(test)

option(BUILD_UNITTEST "build test with googletest" ON)

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

