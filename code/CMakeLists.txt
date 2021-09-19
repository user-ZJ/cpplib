cmake_minimum_required(VERSION 3.14)
project(zack)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fPIC")

set(CMAKE_CXX_STANDARD 11)


include(FetchContent)
set(FETCHCONTENT_QUIET off)
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/thirdpart)

include_directories(
	source/utils
		 )
link_directories(
		)


aux_source_directory(source/utils UTILS_SOURCES)


add_library(${PROJECT_NAME} SHARED ${UTILS_SOURCES})

target_link_libraries(${PROJECT_NAME}
		)
INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


option(BUILD_TEST "build test with googletest" ON)

if(BUILD_TEST)
FetchContent_Declare(googletest
	URL      https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz
	URL_HASH SHA256=b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5
	# SOURCE_DIR ../thirdpart/googletest
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(GTEST_HAS_PTHREAD 1)
FetchContent_MakeAvailable(googletest) #后续链接gtest_main即可
enable_testing()
add_subdirectory(test)
endif()

