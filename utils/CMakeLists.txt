cmake_minimum_required(VERSION 3.14)
project(zack_utils)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fPIC")

include_directories(
		 )
link_directories(
		)

include(FetchContent)
set(FETCHCONTENT_QUIET off)
# set(FETCHCONTENT_BASE_DIR ../fethchs)
FetchContent_Declare(gtest
	URL      https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz
	URL_HASH SHA256=b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5
	SOURCE_DIR ../gtest
)
FetchContent_MakeAvailable(gtest)


set(UTILS_SOURCES 
	path.cpp
	text-utils.cpp
)


add_library(${PROJECT_NAME} SHARED ${UTILS_SOURCES})

target_link_libraries(${PROJECT_NAME}
		)
INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

set(TEST_SOURCES path_test.cpp)
add_executable(path_test ${TEST_SOURCES})
target_link_libraries(path_test zack_utils gtest)
INSTALL(TARGETS path_test RUNTIME DESTINATION bin)

set(TEXT_TEST_SOURCES text-utils-test.cpp)
add_executable(text_test ${TEXT_TEST_SOURCES})
target_link_libraries(text_test zack_utils)
INSTALL(TARGETS text_test RUNTIME DESTINATION bin)
